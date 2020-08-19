#pragma once

#include <util/generic/fwd.h>
#include <util/system/types.h>

struct TSum;
struct TSumMulti;

namespace NPar {
    class TLocalExecutor;
}

template <typename LocalExecutorType>
void AddLangevinNoiseToDerivatives(
    float diffusionTemperature,
    float learningRate,
    ui64 randomSeed,
    TVector<TVector<double>>* derivatives,
    LocalExecutorType* localExecutor
);

void AddLangevinNoiseToLeafDerivativesSum(
    float diffusionTemperature,
    float learningRate,
    double scaledL2Regularizer,
    ui64 randomSeed,
    TVector<TSum>* leafDersSum
);

void AddLangevinNoiseToLeafDerivativesSum(
    float diffusionTemperature,
    float learningRate,
    double scaledL2Regularizer,
    ui64 randomSeed,
    TVector<TSumMulti>* leafDersSum
);
