#ifndef __ADAPTIVE_LEAKY_INTEGRATE_AND_FIRE_KERNEL__
#define __ADAPTIVE_LEAKY_INTEGRATE_AND_FIRE_KERNEL__
#include "CudaUtils.h"
/**
 * Parallel GPU Kernel for a fully connected Network of LeakyIntegrateAndFireNeurons
 */
namespace SNN {

    namespace Kernels {

        namespace GPU {
                
            /**
             */
            __device__ void adaptiveLeakyIntegrateAndFireKernel(

                /* the number of neurons */
                unsigned numNeurons,
                
                /* neuron spike threshold */
                FloatType baseThreshold,

                /* neuron voltage decay factor */
                FloatType voltageDecay,

                /* neuron adaption decay factor */
                FloatType adaptationDecay,

                /* neuron threhsold increase factor */
                FloatType thresholdIncreaseConstant,

                /* neuron refactory period */
                FloatType refactoryPeriod,

                /* the derivative dumping factor */
                FloatType derivativeDumpingFactor,

                /* the neurons spikes */
                FloatType *spikes,

                /* the neurons adaptation values */
                FloatType *thresholdAdaptation,

                /* array of filtered spikes */
                FloatType *filteredSpikes,

                /* the number of neuron spikes */
                FloatType *numSpikes,

                /* the neuron input currents */
                FloatType *I,

                /* the neuron voltage */
                FloatType *v,

                /* the neurons pseudo derivative */
                FloatType *pseudoDerivative,

                /* the last spike time */
                FloatType *timeStepsSinceLastSpike
            ) {

                for (int i = threadIdx.x; i < numNeurons; i += blockDim.x) {

                    /* update voltage */
                    const FloatType v_ = v[i] * voltageDecay + I[i] - spikes[i] * baseThreshold;

                    /* update adaptive threshold */
                    const FloatType thresholdAdaptation_ = thresholdAdaptation[i] * adaptationDecay + spikes[i];
                    const FloatType adaptiveThreshold = baseThreshold + thresholdIncreaseConstant * thresholdAdaptation_;

                    /* update number of timesteps that have pased since the last spike */
                    const FloatType timeStepsSinceLastSpike_ = spikes[i] ? 1 : timeStepsSinceLastSpike[i] + 1;

                    /* update filtered spikes */
                    filteredSpikes[i] = filteredSpikes[i] * voltageDecay + spikes[i];

                    /* pseudo-derivative is fixed to zero within the refactory period */
                    if (timeStepsSinceLastSpike_ > refactoryPeriod) {
#if FloatTypeSize == 32
                        pseudoDerivative[i] = derivativeDumpingFactor * fmaxf(
                            0.0, 
                            1.0 - fabsf((v_ - adaptiveThreshold) / baseThreshold)
                        );
#else
                        pseudoDerivative[i] = derivativeDumpingFactor * fmax(
                            0.0, 
                            1.0 - fabs((v_ - adaptiveThreshold) / baseThreshold)
                        );
#endif
                    } else 
                        pseudoDerivative[i] = 0;
                    
                    /* set spike status */
                    if (timeStepsSinceLastSpike_ > refactoryPeriod && v_ > adaptiveThreshold) {
                        spikes[i] = 1;
                        numSpikes[i]++;
                    } else 
                        spikes[i] = 0;
                    
                    /* set values afterwards */
                    timeStepsSinceLastSpike[i] = timeStepsSinceLastSpike_;
                    thresholdAdaptation[i]     = thresholdAdaptation_;
                    v[i] = v_;
                    I[i] = 0;
                }
            }
        }
    }
}
#endif /* __ADAPTIVE_LEAKY_INTEGRATE_AND_FIRE_KERNEL__ */
