#pragma once

#include "model_quantization_adapter.h"

#include <catboost/libs/data/data_provider.h>
#include <catboost/libs/data/objects.h>
#include <catboost/private/libs/options/restrictions.h>

#include <util/generic/hash.h>
#include <util/generic/vector.h>
#include <util/system/types.h>


class TFold;
struct TSplit;
struct TSplitTree;
struct TNonSymmetricTreeStructure;

namespace NPar {
    class TLocalExecutor;
}
namespace OMPNPar {
    class TLocalExecutor;
}

/*
 * for index calculation we need either TFullSubset or TIndexedSubset
 * so, for TBlockedSubset we need to make it TIndexedSubset
 */
using TIndexedSubsetCache = THashMap<const NCB::TFeaturesArraySubsetIndexing*, NCB::TIndexedSubset<ui32>>;

template <typename LocalExecutorType>
void GetObjectsDataAndIndexing(
    const NCB::TTrainingDataProviders& trainingData,
    const TFold& fold,
    bool isEstimated,
    bool isOnline,
    ui32 objectSubsetIdx, // 0 - learn, 1+ - test (subtract 1 for testIndex)
    TIndexedSubsetCache* indexedSubsetCache,
    LocalExecutorType* localExecutor,
    NCB::TQuantizedObjectsDataProviderPtr* objectsData,
    const ui32** columnIndexing // can return nullptr
);

template <typename LocalExecutorType>
void SetPermutedIndices(
    const TSplit& split,
    const NCB::TTrainingDataProviders& trainingData,
    int curDepth,
    const TFold& fold,
    TVector<TIndexType>* indices,
    LocalExecutorType* localExecutor);

TVector<bool> GetIsLeafEmpty(int curDepth, const TVector<TIndexType>& indices);

int GetRedundantSplitIdx(const TVector<bool>& isLeafEmpty);


enum class EBuildIndicesDataParts {
    All,
    LearnOnly,
    TestOnly
};

template <typename LocalExecutorType>
TVector<TIndexType> BuildIndices(
    const TFold& fold, // can be empty
    const TVariant<TSplitTree, TNonSymmetricTreeStructure>& tree,
    const NCB::TTrainingDataProviders& trainingData,
    EBuildIndicesDataParts dataParts,
    LocalExecutorType* localExecutor);

TVector<TIndexType> BuildIndicesForBinTree(
    const TFullModel& model,
    const NCB::NModelEvaluation::IQuantizedData* quantizedFeatures,
    size_t treeId);


