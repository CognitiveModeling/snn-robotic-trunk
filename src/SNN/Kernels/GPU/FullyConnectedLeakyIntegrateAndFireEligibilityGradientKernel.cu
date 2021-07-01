#ifndef __FULLY_CONNECTED_LEAKY_INTEGRATE_AND_FIRE_ELIGIBILITY_GRADIENT_KERNEL__
#define __FULLY_CONNECTED_LEAKY_INTEGRATE_AND_FIRE_ELIGIBILITY_GRADIENT_KERNEL__
#include "CudaUtils.h"
/**
 * Parallel GPU Kernel for a fully connected Network of leaky integrate and fire neurons
 * combining eligibility computation with gradient computation 
 */
namespace SNN {

    namespace Kernels {

        namespace GPU {
            
            __device__ void fullyConnectedLeakyIntegrateAndFireEligibilityGradientKernel(

                /* the number of input neurons */
                unsigned numInputs,

                /* the number of output neurons */
                unsigned numOutputs,

                /* the number of network output neurons */
                unsigned numNetworkOutputs,

                /* the target firing rate */
                FloatType targetFiringRate,

                /* the firing rate scalling factor */
                FloatType firingRateScallingFactor,

                /* the readout voltage decay factor */
                FloatType decayFactor,

                /* array of filtered spikes */
                FloatType *filteredSpikes,

                /* the neurons firing rates */
                FloatType *firingRates,

                /* array of derivatives */
                FloatType *derivatives,

                /* the feedback weights */
                FloatType *feedbackWeights,

                /* the network outputs */
                FloatType *outputs,

                /* the network targets */
                FloatType *targets,

                /* the filtered eligibility traces */
                FloatType *filteredEligibilityTraces,

                /* the firing rate gradients */
                FloatType *firingRateGradients,

                /* the fixed braodcast gradients */
                FloatType *fixedBroadcastGradients
            ) {
                cudaAssert(numOutputs == blockDim.x);
                const int o = threadIdx.x;

                /* compute learn signals */
                FloatType learnSignal = 0;
                for (unsigned k = 0; k < numNetworkOutputs; k++) {
                    learnSignal += feedbackWeights[o * numNetworkOutputs + k] * 
                                   (outputs[k] - targets[k]);
                }
                
                for (unsigned i = 0; i < numInputs; i++) {
                    const unsigned index = i * numOutputs + o;

                    const FloatType eligibilityTrace = derivatives[o] * filteredSpikes[i];

                    /* compute the firing rate gradients */
                    firingRateGradients[index] += 
                        firingRateScallingFactor * 
                        (firingRates[o] - targetFiringRate) *
                        eligibilityTrace;
                
                    /* compute fixed broadcast gradient */
                    const FloatType filteredEligibilityTrace = 
                        filteredEligibilityTraces[index] * decayFactor +
                        eligibilityTrace;

                    fixedBroadcastGradients[index] += learnSignal * filteredEligibilityTrace;

                    /* set values afterwards */
                    filteredEligibilityTraces[index] = filteredEligibilityTrace;
                }
            }
        }
    }
}
#endif /* __FULLY_CONNECTED_LEAKY_INTEGRATE_AND_FIRE_ELIGIBILITY_GRADIENT_KERNEL__ */
