#ifndef __LONG_SHORT_TERM_MEMORY_ELIGIBILITY_GRADIENT_KERNEL__
#define __LONG_SHORT_TERM_MEMORY_ELIGIBILITY_GRADIENT_KERNEL__
#include "CudaUtils.h"
/**
 * Parallel GPU Kernel for a fully connected Network of leaky integrate and fire neurons
 * combining eligibility computation with gradient computation 
 */
namespace SNN {

    namespace Kernels {

        namespace GPU {

            __device__ void longShortTermMemoryEligibilityGradientKernel(

                /* the number of input neurons */
                unsigned numInputs,

                /* the number of (leaky integrate and fire) output neurons */
                unsigned numStandartOutputs,

                /* the number of (adaptive leaky integrate and fire) output neurons */
                unsigned numAdaptiveOutputs,

                /* the target firing rate */
                FloatType targetFiringRate,

                /* the firing rate scalling factor */
                FloatType firingRateScallingFactor,

                /* the readout voltage decay factor */
                FloatType decayFactor,
                
                /* the decay factor for the adaptive threshold */
                FloatType adaptationDecay,
                
                /* the factor about which the base threshold increases */
                FloatType thresholdIncreaseConstant,

                /* array of filtered spikes */
                FloatType *filteredSpikes,

                /* the neurons firing rates */
                FloatType *firingRates,

                /* array of derivatives */
                FloatType *derivatives,

                /* the learnSignals for each output neuron */
                FloatType *learnSignals,

                /* the adaption eligibility part */
                FloatType *adaptionEligibility,

                /* the filtered eligibility traces */
                FloatType *filteredEligibilityTraces,

                /* the firing rate gradients */
                FloatType *firingRateGradients,

                /* the fixed braodcast gradients */
                FloatType *fixedBroadcastGradients
            ) {
                const int id = threadIdx.x;

                for (unsigned offset = 0; offset + id < numInputs * numStandartOutputs; offset += blockDim.x) {
                    const unsigned index = offset + id; // o * numInputs + i
                    const unsigned o = index / numInputs; 
                    const unsigned i = index % numInputs; 

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

                    fixedBroadcastGradients[index] += learnSignals[o] * filteredEligibilityTrace;

                    /* set values afterwards */
                    filteredEligibilityTraces[index] = filteredEligibilityTrace;
                }

                for (unsigned offset = 0; offset + id < numInputs * numAdaptiveOutputs; offset += blockDim.x) {
                    const unsigned adaptionIndex = offset + id; 
                    const unsigned index = adaptionIndex + numInputs * numStandartOutputs;
                    const unsigned o = index / numInputs; 
                    const unsigned i = index % numInputs; 

                    const FloatType eligibilityTrace = derivatives[o] * (
                        filteredSpikes[i] - thresholdIncreaseConstant * adaptionEligibility[adaptionIndex]
                    );

                    adaptionEligibility[adaptionIndex] = 
                        derivatives[o] * filteredSpikes[i] +
                        (adaptationDecay - derivatives[o] * thresholdIncreaseConstant) *
                        adaptionEligibility[adaptionIndex];

                    /* compute the firing rate gradients */
                    firingRateGradients[index] += 
                        firingRateScallingFactor * 
                        (firingRates[o] - targetFiringRate) *
                        eligibilityTrace;
                
                    /* compute fixed broadcast gradient */
                    const FloatType filteredEligibilityTrace = 
                        filteredEligibilityTraces[index] * decayFactor +
                        eligibilityTrace;

                    fixedBroadcastGradients[index] += learnSignals[o] * filteredEligibilityTrace;

                    /* set values afterwards */
                    filteredEligibilityTraces[index] = filteredEligibilityTrace;
                }
            }
        }
    }
}
#endif /* __LONG_SHORT_TERM_MEMORY_ELIGIBILITY_GRADIENT_KERNEL__ */
