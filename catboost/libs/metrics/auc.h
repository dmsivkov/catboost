#pragma once

#include "sample.h"

#include <library/cpp/threading/local_executor/local_executor.h>
#include <library/cpp/threading/local_executor/omp_local_executor.h>
template <typename LocalExecutorType>
double CalcAUC(TVector<NMetrics::TSample>* samples, LocalExecutorType* localExecutor, double* outWeightSum = nullptr, double* outPairWeightSum = nullptr);
double CalcAUC(TVector<NMetrics::TSample>* samples, double* outWeightSum = nullptr, double* outPairWeightSum = nullptr, int threadCount = 1);

template <typename LocalExecutorType>
double CalcBinClassAuc(TVector<NMetrics::TBinClassSample>* positiveSamples, TVector<NMetrics::TBinClassSample>* negativeSamples, LocalExecutorType* localExecutor);
double CalcBinClassAuc(TVector<NMetrics::TBinClassSample>* positiveSamples, TVector<NMetrics::TBinClassSample>* negativeSamples, int threadCount = 1);
