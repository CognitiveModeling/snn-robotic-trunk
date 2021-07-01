#ifndef __FULLY_CONNECTED_FAST_FIRING_RATE_KERNEL__
#define __FULLY_CONNECTED_FAST_FIRING_RATE_KERNEL__
#include "CudaUtils.h"
/**
 * Parallel GPU Kernel for a fully connected Network of leaky integrate and fire neurons
 * combining eligibility computation with gradient computation 
 */
namespace SNN {

    namespace Kernels {

        namespace GPU {
            
            __device__ void fullyConnectedFastFiringRateKernel(

                /* the number of input neurons */
                unsigned numInputs,

                /* the number of output neurons */
                unsigned numOutputs,

                /* the target firing rate */
                FloatType targetFiringRate,

                /* the firing rate scalling factor */
                FloatType firingRateScallingFactor,

                /* the number of input spikes */
                FloatType *numInputSpikes,

                /* the number of output spikes */
                FloatType *numOutputSpikes,

                /* the neurons firing rates */
                FloatType *firingRates,

                /* the firing rate gradients */
                FloatType *firingRateGradients

            ) {
                const unsigned i = threadIdx.x;

                for (unsigned index = i; index < numInputs * numOutputs; index += blockDim.x) {
                    const unsigned o = index / numInputs; 
                    const unsigned i = index % numInputs; 

                    /* compute the firing rate gradients */
                    firingRateGradients[index] += 
                        firingRateScallingFactor * 
                        (firingRates[o] - targetFiringRate) *
                        (numInputSpikes[i] + numOutputSpikes[o]);
                }
            }
        }
    }
}
#endif /* __FULLY_CONNECTED_FAST_FIRING_RATE_KERNEL__ */
