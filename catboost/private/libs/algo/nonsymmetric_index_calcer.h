#pragma once

#include <catboost/libs/data/data_provider.h>
#include <catboost/private/libs/options/restrictions.h>

#include <util/generic/fwd.h>
#include <util/generic/vector.h>


class TFold;
struct TSplitNode;
struct TNonSymmetricTreeStructure;
struct TOnlineCTR;

namespace NPar {
    class TLocalExecutor;
}

template <typename LocalExecutorType>
void UpdateIndices(
    const TSplitNode& node,
    const NCB::TTrainingDataProviders& trainingData,
    const NCB::TIndexedSubset<ui32>& docsSubset,
    const TFold& fold,
    LocalExecutorType* localExecutor,
    TArrayRef<TIndexType> indices
);

template <typename LocalExecutorType>
void BuildIndicesForDataset(
    const TNonSymmetricTreeStructure& tree,
    const NCB::TTrainingDataProviders& trainingData,
    const TFold& fold,
    ui32 sampleCount,
    const TVector<const TOnlineCTR*>& onlineCtrs,
    ui32 docOffset,
    ui32 objectSubsetIdx, // 0 - learn, 1+ - test (subtract 1 for testIndex)
    LocalExecutorType* localExecutor,
    TIndexType* indices);
