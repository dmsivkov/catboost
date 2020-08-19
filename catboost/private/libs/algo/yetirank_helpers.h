#pragma once

#include "learn_context.h"

#include <util/generic/array_ref.h>


namespace NCatboostOptions {
    class TCatBoostOptions;
    class TLossDescription;
}

namespace NPar {
    class TLocalExecutor;
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
);
