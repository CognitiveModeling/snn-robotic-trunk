#ifndef __SPARSE_FAST_FIRING_RATE_KERNEL__
#define __SPARSE_FAST_FIRING_RATE_KERNEL__
#include "CudaUtils.h"
/**
 * Parallel GPU Kernel for a fully connected Network of leaky integrate and fire neurons
 * combining eligibility computation with gradient computation 
 */
namespace SNN {

    namespace Kernels {

        namespace GPU {
            
            __device__ void sparseFastFiringRateKernel(

                /* the number synaptic input weights */
                unsigned numInputWeights,

                /* the number synaptic input weights */
                unsigned numHiddenWeights,

                /* the target firing rate */
                FloatType targetFiringRate,

                /* the firing rate scalling factor */
                FloatType firingRateScallingFactor,

                /* the synaptic input weights */
                FloatType *inputWeights,

                /* the synaptic input weights */
                FloatType *hiddenWeights,

                /* the synaptic input weight input indices */
                unsigned *inputWeightsIn,

                /* the synaptic input weight output indices */
                unsigned *inputWeightsOut,

                /* the synaptic input weight input indices */
                unsigned *hiddenWeightsIn,

                /* the synaptic input weight output indices */
                unsigned *hiddenWeightsOut,

                /* the number of input spikes */
                FloatType *numInputSpikes,

                /* the number of hidden spikes */
                FloatType *numHiddenSpikes,

                /* the neurons firing rates */
                FloatType *firingRates,

                /* the input firing rate gradients */
                FloatType *inputFiringRateGradients,

                /* the hidden firing rate gradients */
                FloatType *hiddenFiringRateGradients

            ) {
                /* add errors to input gradients */
                for (unsigned i = threadIdx.x; i < numInputWeights; i += blockDim.x) {
                    inputFiringRateGradients[i] += 
                        firingRateScallingFactor * 
                        (firingRates[inputWeightsOut[i]] - targetFiringRate) *
                        (numInputSpikes[inputWeightsIn[i]] + numHiddenSpikes[inputWeightsOut[i]]);
                }

                /* add errors to hidden gradients */
                for (unsigned i = threadIdx.x; i < numHiddenWeights; i += blockDim.x) {
                    hiddenFiringRateGradients[i] += 
                        firingRateScallingFactor * 
                        (firingRates[hiddenWeightsOut[i]] - targetFiringRate) *
                        (numHiddenSpikes[hiddenWeightsIn[i]] + numHiddenSpikes[hiddenWeightsOut[i]]);
                }
            }
        }
    }
}
#endif /* __SPARSE_FAST_FIRING_RATE_KERNEL__ */
