#ifndef __SPARSE_DELTA_ERROR_KERNEL__
#define __SPARSE_DELTA_ERROR_KERNEL__
#include "CudaUtils.h"
/**
 * Parallel GPU Kernel for a fully connected Network of leaky integrate and fire neurons
 * combining eligibility computation with gradient computation 
 */
namespace SNN {

    namespace Kernels {

        namespace GPU {
            
            __device__ void sparseDeltaErrorKernel(

                /* the number of hidden neurons */
                unsigned numHidden,

                /* the number of (leaky integrate and fire) hiddem neurons */
                unsigned numStandart,

                /* the number synaptic input weights */
                unsigned numHiddenWeights,

                /* the number synaptic input weights */
                unsigned numOutputWeights,

                /* the firing threshold */
                FloatType spikeThreshold,

                /* the factor about which the base threshold increases */
                FloatType thresholdIncrease,

                /* the decay factor for the adaptive threshold */
                FloatType adaptionDecay,

                /* the decay factor for the voltage of the neurons */
                FloatType voltageDecay,

                /* the synaptic input weights */
                FloatType *hiddenWeights,

                /* the synaptic input weights */
                FloatType *outputWeights,

                /* the synaptic input weight input indices */
                unsigned *hiddenWeightsIn,

                /* the synaptic input weight input indices */
                unsigned *outputWeightsIn,

                /* the synaptic input weight output indices */
                unsigned *hiddenWeightsOut,

                /* the synaptic input weight output indices */
                unsigned *outputWeightsOut,

                /* the filtered (back propagated) output errors */
                FloatType *filteredOutputErrors,

                /* the voltage component of the delta errors */
                FloatType *voltageError,

                /* the adaption component of the delta errors */
                FloatType *adaptionError,

                /* the hidden derivatives */
                FloatType *derivatives,

                /* the learnsignals for ea.hidden neuron */
                FloatType *learnSignals

            ) {
                const int outWeightsPerThread     = ceilf(numOutputWeights / float(blockDim.x));
                const int hiddenWeightsPerThread  = ceilf(numHiddenWeights / float(blockDim.x));
                const int outStart                = threadIdx.x * outWeightsPerThread;
                const int outEnd                  = min(numOutputWeights, (1 + threadIdx.x) * outWeightsPerThread);
                const int hiddenStart             = threadIdx.x * hiddenWeightsPerThread;
                const int hiddenEnd               = min(numHiddenWeights, (1 + threadIdx.x) * hiddenWeightsPerThread);

                for (int i = threadIdx.x; i < numHidden; i += blockDim.x) {
                    learnSignals[i] = 0;
                }

                __syncthreads();

                for (int i = outStart; i < outEnd; i++)
                    atomicAdd(learnSignals + outputWeightsIn[i], filteredOutputErrors[outputWeightsOut[i]] * outputWeights[i]);

                for (int i = hiddenStart; i < hiddenEnd; i++)
                    atomicAdd(learnSignals + hiddenWeightsIn[i], voltageError[hiddenWeightsOut[i]] *  hiddenWeights[i]);

                /* add own error from futur timestep */
                for (int i = threadIdx.x; i < numHidden; i += blockDim.x) {
                    const int ai = i - numStandart;;

                    atomicAdd(learnSignals + i,  -1 * spikeThreshold * voltageError[i]);

                    /* add own error from futur timestep (adaptive part) */
                    if (i >= numStandart) 
                        atomicAdd(learnSignals + i, adaptionError[ai]);
                }
                __syncthreads();

                /* compute new delta errors */
                for (int i = threadIdx.x; i < numHidden; i += blockDim.x) {
                    const int ai = i - numStandart;

                    if (i >= numStandart) {
                        adaptionError[ai] = 
                            adaptionDecay * adaptionError[ai] - 
                            learnSignals[i] * derivatives[i] * thresholdIncrease;
                    }

                    voltageError[i] = voltageError[i] * voltageDecay + learnSignals[i] * derivatives[i];
                }

                __syncthreads();

                for (int i = outStart; i < outEnd; i++)
                    atomicAdd(learnSignals + outputWeightsIn[i], -1 * filteredOutputErrors[outputWeightsOut[i]] * outputWeights[i]);

                __syncthreads();
            }
        }
    }
}
#endif /* __SPARSE_DELTA_ERROR_KERNEL__ */
