#pragma once

#include "map_merge.h"

#include <library/cpp/dot_product/dot_product.h>
#include <library/cpp/threading/local_executor/local_executor.h>
#include <library/cpp/threading/local_executor/omp_local_executor.h>

#include <util/generic/array_ref.h>
#include <util/generic/cast.h>
#include <util/generic/maybe.h>
#include <util/generic/vector.h>

#include <functional>

namespace common {

template <typename TBody>
inline void ParallelFor(NPar::TLocalExecutor& executor, ui32 from, ui32 to, TBody&& body) {
    NPar::ParallelFor(executor, from, to, std::forward<TBody>(body));
}

template <typename TBody>
inline void ParallelFor(OMPNPar::TLocalExecutor& executor, ui32 from, ui32 to, TBody&& body) {
    OMPNPar::ParallelFor(executor, from, to, std::forward<TBody>(body));
}

}

namespace NCB {

    // tasks is not const because elements of tasks are cleared after execution
    template <typename LocalExecutorType>
    void ExecuteTasksInParallel(TVector<std::function<void()>>* tasks, LocalExecutorType* localExecutor);

    template <class T, typename LocalExecutorType>
    void ParallelFill(
        const T& fillValue,
        TMaybe<int> blockSize,
        LocalExecutorType* localExecutor,
        TArrayRef<T> array) {

        typename LocalExecutorType::TExecRangeParams rangeParams(0, SafeIntegerCast<int>(array.size()));
        if (blockSize) {
            rangeParams.SetBlockSize(*blockSize);
        } else {
            rangeParams.SetBlockCountToThreadCount();
        }

        localExecutor->ExecRange(
            [=] (int i) { array[i] = fillValue; },
            rangeParams,
            LocalExecutorType::WAIT_COMPLETE);
    }

    template <typename TNumber>
    inline TNumber L2NormSquared(
        TConstArrayRef<TNumber> array,
        NPar::TLocalExecutor* localExecutor
    ) {
        TNumber result = 0;
        NCB::MapMerge(
            localExecutor,
            TSimpleIndexRangesGenerator<int>(TIndexRange<int>(array.size()), /*blockSize*/10000),
            /*mapFunc*/[&](NCB::TIndexRange<int> partIndexRange, TNumber* output) {
                Y_ASSERT(!partIndexRange.Empty());
                *output = DotProduct(
                    array.data() + partIndexRange.Begin,
                    array.data() + partIndexRange.Begin,
                    partIndexRange.GetSize()
                );
            },
            /*mergeFunc*/[](TNumber* output, TVector<TNumber>&& addVector) {
                for (TNumber addItem : addVector) {
                    *output += addItem;
                }
            },
            &result
        );
        return result;
    }

    template <typename TNumber, typename LocalExecutorType>
    inline void FillRank2(
        TNumber value,
        int rowCount,
        int columnCount,
        TVector<TVector<TNumber>>* dst,
        LocalExecutorType* localExecutor
    ) {
        constexpr int minimumElementCount = 1000;
        dst->resize(rowCount);
        if (rowCount * columnCount < minimumElementCount) {
            for (auto& dimension : *dst) {
                dimension.yresize(columnCount);
                Fill(dimension.begin(), dimension.end(), value);
            }
        } else if (columnCount < rowCount * minimumElementCount) {
            common::ParallelFor(
                *localExecutor,
                0,
                rowCount,
                [=] (int rowIdx) {
                    (*dst)[rowIdx].yresize(columnCount);
                    Fill((*dst)[rowIdx].begin(), (*dst)[rowIdx].end(), value);
                });
        } else {
            for (auto& dimension : *dst) {
                dimension.yresize(columnCount);
                ParallelFill(value, /*blockSize*/ Nothing(), localExecutor, MakeArrayRef(dimension));
            }
        }
    }
}
