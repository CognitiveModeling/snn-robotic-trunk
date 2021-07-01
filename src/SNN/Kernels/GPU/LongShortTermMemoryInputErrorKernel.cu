#ifndef __LONG_SHORT_TERM_MEMORY_INPUT_ERROR_KERNEL__
#define __LONG_SHORT_TERM_MEMORY_INPUT_ERROR_KERNEL__
#include "FullyConnectedLeakyIntegrateAndFireKernel.cu"
#include "FullyConnectedAdaptiveLeakyIntegrateAndFireKernel.cu"
#include "FullyConnectedInputOutputKernel.cu"
#include "FullyConnectedInputSpikePropagationKernel.cu"
#include "FullyConnectedHiddenSpikePropagationKernel.cu"
#include "LongShortTermMemoryLearnSignalKernel.cu"
#include "LongShortTermMemoryEligibilityGradientKernel.cu"
#include "LongShortTermMemoryLeakyReadoutGradientKernel.cu"
#include "LongShortTermMemoryInputEligibilityKernel.cu"
/**
 * Parallel kernel for a Long Short Term Spiking Network
 */
namespace SNN {

    namespace Kernels {

        namespace GPU {

            __device__ void longShortTermMemoryInputErrorKernel(

                /* the number of input neurons */
                unsigned numInputs,

                /* the number of (leaky integrate and fire) hiddem neurons */
                unsigned numStandartHidden,

                /* the number of (adaptive leaky integrate and fire) hidden neurons */
                unsigned numAdaptiveHidden,

                /* the number of simmulation time steps */
                unsigned numSimulationTimesteps,

                /* the simulation start and end time */
                int startTime, 
                int endTime,

                /* the synaptic input weights */
                FloatType *inputWeights,

                /* the network derivatives fore one simulation run */
                FloatType *derivativesOverTime,

                /* the input errors fore one simmulation run */
                FloatType *inputErrorsOverTime,

                /* the learn signals for ea.hiden neuron over time in 
                 * order to compute the input errors */ 
                FloatType *inputErrorLearnSignals

            ) {
                const unsigned numHidden = numStandartHidden + numAdaptiveHidden;
                cudaAssert(numInputs == blockDim.x || numHidden == blockDim.x);

                inputErrorsOverTime           += blockIdx.x * numInputs * numSimulationTimesteps;
                inputErrorLearnSignals        += blockIdx.x * numHidden * numSimulationTimesteps;
                derivativesOverTime           += blockIdx.x * numHidden * numSimulationTimesteps;

                if (startTime < 0) startTime = 0;
                if (endTime   < 0) endTime   = numSimulationTimesteps;

                /* clear values */
                const int i = threadIdx.x;
                if (i >= numInputs) return;
                __syncthreads();

                for (unsigned t = startTime; t < endTime; t++) {

                    FloatType inputError = 0;
                    for (unsigned h = 0; h < numHidden; h++) {
                        const unsigned index = i * numHidden + h;
                        inputError += 
                            inputErrorLearnSignals[t * numHidden + h] * 
                            derivativesOverTime[t * numHidden + h]    * 
                            inputWeights[index];
                    }

                    inputErrorsOverTime[t * numInputs + i] = inputError;
                }
            }
        }
    }
}
#endif /* __LONG_SHORT_TERM_MEMORY_INPUT_ERROR_KERNEL__ */
