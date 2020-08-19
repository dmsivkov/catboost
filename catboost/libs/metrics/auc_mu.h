#pragma once

#include <library/cpp/threading/local_executor/local_executor.h>
#include <library/cpp/threading/local_executor/omp_local_executor.h>

#include <util/generic/vector.h>
#include <util/generic/maybe.h>
template <typename LocalExecutorType>
double CalcMuAuc(
    const TVector<TVector<double>>& approx,
    const TConstArrayRef<float>& target,
    const TConstArrayRef<float>& weight,
    LocalExecutorType* localExecutor,
    const TMaybe<TVector<TVector<double>>>& misclassCostMatrix = Nothing()
);

double CalcMuAuc(
    const TVector<TVector<double>>& approx,
    const TConstArrayRef<float>& target,
    const TConstArrayRef<float>& weight,
    int threadCount = 1,
    const TMaybe<TVector<TVector<double>>>& misclassCostMatrix = Nothing()
);
