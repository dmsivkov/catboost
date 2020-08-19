#include "yetirank_helpers.h"

#include "approx_updater_helpers.h"

#include <catboost/private/libs/data_types/pair.h>
#include <catboost/private/libs/options/catboost_options.h>
#include <catboost/private/libs/options/loss_description.h>

#include <library/cpp/threading/local_executor/local_executor.h>
#include <library/cpp/threading/local_executor/omp_local_executor.h>

#include <util/generic/vector.h>


static void GenerateYetiRankPairsForQuery(
    const float* relevs,
    const double* expApproxes,
    float queryWeight,
    ui32 querySize,
    int permutationCount,
    double decaySpeed,
    ui64 randomSeed,
    TVector<TVector<TCompetitor>>* competitors
) {
    TFastRng64 rand(randomSeed);
    TVector<TVector<TCompetitor>>& competitorsRef = *competitors;
    competitorsRef.clear();
    competitorsRef.resize(querySize);

    TVector<int> indices(querySize);
    TVector<TVector<float>> competitorsWeights(querySize, TVector<float>(querySize));
    for (int permutationIndex = 0; permutationIndex < permutationCount; ++permutationIndex) {
        std::iota(indices.begin(), indices.end(), 0);
        TVector<double> bootstrappedApprox(expApproxes, expApproxes + querySize);
        for (ui32 docId = 0; docId < querySize; ++docId) {
            const float uniformValue = rand.GenRandReal1();
            // TODO(nikitxskv): try to experiment with different bootstraps.
            bootstrappedApprox[docId] *= uniformValue / (1.000001f - uniformValue);
        }

        Sort(
            indices,
            [&](int i, int j) {
                return bootstrappedApprox[i] > bootstrappedApprox[j];
            }
        );

        double decayCoefficient = 1;
        for (ui32 docId = 1; docId < querySize; ++docId) {
            const int firstCandidate = indices[docId - 1];
            const int secondCandidate = indices[docId];
            const double magicConst = 0.15; // Like in GPU

            const float pairWeight = magicConst * decayCoefficient
                * Abs(relevs[firstCandidate] - relevs[secondCandidate]);
            if (relevs[firstCandidate] > relevs[secondCandidate]) {
                competitorsWeights[firstCandidate][secondCandidate] += pairWeight;
            } else if (relevs[firstCandidate] < relevs[secondCandidate]) {
                competitorsWeights[secondCandidate][firstCandidate] += pairWeight;
            }
            decayCoefficient *= decaySpeed;
        }
    }

    // TODO(nikitxskv): Can be optimized
    for (ui32 winnerIndex = 0; winnerIndex < querySize; ++winnerIndex) {
        for (ui32 loserIndex = 0; loserIndex < querySize; ++loserIndex) {
            const float competitorsWeight
                = queryWeight * competitorsWeights[winnerIndex][loserIndex] / permutationCount;
            if (competitorsWeight != 0) {
                competitorsRef[winnerIndex].push_back({loserIndex, competitorsWeight});
            }
        }
    }
}

template <typename LocalExecutorType>
void UpdatePairsForYetiRank(
    TConstArrayRef<double> approxes,
    TConstArrayRef<float> relevances,
    const NCatboostOptions::TLossDescription& lossDescription,
    ui64 randomSeed,
    int queryBegin,
    int queryEnd,
    TVector<TQueryInfo>* queriesInfo,
    LocalExecutorType* localExecutor
) {
    const int permutationCount = NCatboostOptions::GetYetiRankPermutations(lossDescription);
    const double decaySpeed = NCatboostOptions::GetYetiRankDecay(lossDescription);

    typename LocalExecutorType::TExecRangeParams blockParams(queryBegin, queryEnd);
    blockParams.SetBlockCount(CB_THREAD_LIMIT);
    const int blockSize = blockParams.GetBlockSize();
    const ui32 blockCount = blockParams.GetBlockCount();
    const TVector<ui64> randomSeeds = GenRandUI64Vector(blockCount, randomSeed);
    common::ParallelFor(
        *localExecutor,
        0,
        blockCount,
        [&](int blockId) {
            TFastRng64 rand(randomSeeds[blockId]);
            const int from = queryBegin + blockId * blockSize;
            const int to = Min<int>(queryBegin + (blockId + 1) * blockSize, queryEnd);
            for (int queryIndex = from; queryIndex < to; ++queryIndex) {
                TQueryInfo& queryInfoRef = (*queriesInfo)[queryIndex];
                GenerateYetiRankPairsForQuery(
                    relevances.data() + queryInfoRef.Begin,
                    approxes.data() + queryInfoRef.Begin,
                    queryInfoRef.Weight,
                    queryInfoRef.End - queryInfoRef.Begin,
                    permutationCount,
                    decaySpeed,
                    rand.GenRand(),
                    &queryInfoRef.Competitors
                );
            }
        }
    );
}
template
void UpdatePairsForYetiRank<NPar::TLocalExecutor>(
    TConstArrayRef<double> approxes,
    TConstArrayRef<float> relevances,
    const NCatboostOptions::TLossDescription& lossDescription,
    ui64 randomSeed,
    int queryBegin,
    int queryEnd,
    TVector<TQueryInfo>* queriesInfo,
    NPar::TLocalExecutor* localExecutor
);
template
void UpdatePairsForYetiRank<OMPNPar::TLocalExecutor>(
    TConstArrayRef<double> approxes,
    TConstArrayRef<float> relevances,
    const NCatboostOptions::TLossDescription& lossDescription,
    ui64 randomSeed,
    int queryBegin,
    int queryEnd,
    TVector<TQueryInfo>* queriesInfo,
    OMPNPar::TLocalExecutor* localExecutor
);

template <typename LocalExecutorType>
void YetiRankRecalculation(
    const TFold& ff,
    const TFold::TBodyTail& bt,
    const NCatboostOptions::TCatBoostOptions& params,
    ui64 randomSeed,
    LocalExecutorType* localExecutor,
    TVector<TQueryInfo>* recalculatedQueriesInfo,
    TVector<float>* recalculatedPairwiseWeights
) {
    Y_ASSERT(ff.LearnTarget.size() == 1);
    *recalculatedQueriesInfo = ff.LearnQueriesInfo;
    UpdatePairsForYetiRank(
        bt.Approx[0],
        ff.LearnTarget[0],
        params.LossFunctionDescription.Get(),
        randomSeed,
        0,
        bt.TailQueryFinish,
        recalculatedQueriesInfo,
        localExecutor
    );
    recalculatedPairwiseWeights->resize(bt.PairwiseWeights.ysize());
    CalcPairwiseWeights(*recalculatedQueriesInfo, bt.TailQueryFinish, recalculatedPairwiseWeights);
}

template
void YetiRankRecalculation<NPar::TLocalExecutor>(
    const TFold& ff,
    const TFold::TBodyTail& bt,
    const NCatboostOptions::TCatBoostOptions& params,
    ui64 randomSeed,
    NPar::TLocalExecutor* localExecutor,
    TVector<TQueryInfo>* recalculatedQueriesInfo,
    TVector<float>* recalculatedPairwiseWeights
);
template
void YetiRankRecalculation<OMPNPar::TLocalExecutor>(
    const TFold& ff,
    const TFold::TBodyTail& bt,
    const NCatboostOptions::TCatBoostOptions& params,
    ui64 randomSeed,
    OMPNPar::TLocalExecutor* localExecutor,
    TVector<TQueryInfo>* recalculatedQueriesInfo,
    TVector<float>* recalculatedPairwiseWeights
);