#ifndef __LONG_SHORT_TERM_MEMORY_SPARSE_ELIGIBILITY_BACK_PROPAGATION_GRADIENT_KERNEL__
#define __LONG_SHORT_TERM_MEMORY_SPARSE_ELIGIBILITY_BACK_PROPAGATION_GRADIENT_KERNEL__
#include "CudaUtils.h"
/**
 * Parallel GPU Kernel for a fully connected Network of leaky integrate and fire neurons
 * combining eligibility computation with gradient computation 
 */
namespace SNN {

    namespace Kernels {

        namespace GPU {

            __device__ void longShortTermMemorySparseEligibilityBackPropagationGradientKernel(

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

                /* the number synaptes to leaky integrate and fire neurons */
                unsigned numStandartSynapses,

                /* the number synaptes to adaptive leaky integrate and fire neurons */
                unsigned numAdaptiveSynapses,

                /* the synaptic weight input indices */
                unsigned *synapseIn,

                /* the synaptic weight output indices */
                unsigned *synapseOut,

                /* the adaption eligibility part */
                FloatType *adaptionEligibility,

                /* the firing rate gradients */
                FloatType *firingRateGradients,

                /* the fixed braodcast gradients */
                FloatType *eligibilityGradients
            ) {
                for (int index = threadIdx.x; index < numStandartSynapses; index += blockDim.x) {
                    const unsigned i = synapseIn[index];
                    const unsigned o = synapseOut[index];

                    const FloatType eligibilityTrace = derivatives[o] * filteredSpikes[i];

                    /* compute the firing rate gradients */
                    firingRateGradients[index] += 
                        firingRateScallingFactor * 
                        (firingRates[o] - targetFiringRate) *
                        eligibilityTrace;
                

                    eligibilityGradients[index] += learnSignals[o] * eligibilityTrace;
                }

                for (int index = numStandartSynapses + threadIdx.x; 
                     index < numStandartSynapses + numAdaptiveSynapses; 
                     index += blockDim.x) {

                    const unsigned i = synapseIn[index];
                    const unsigned o = synapseOut[index];
                    const unsigned adaptionIndex = index - numStandartSynapses;

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
                
                    eligibilityGradients[index] += learnSignals[o] * eligibilityTrace;
                }
            }
        }
    }
}
#endif /* __LONG_SHORT_TERM_MEMORY_SPARSE_ELIGIBILITY_BACK_PROPAGATION_GRADIENT_KERNEL__ */
