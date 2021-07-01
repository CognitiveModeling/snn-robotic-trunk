#ifndef __LEAKY_INTEGRATE_AND_FIRE_KERNEL__
#define __LEAKY_INTEGRATE_AND_FIRE_KERNEL__
#include "CudaUtils.h"
/**
 * Parallel GPU Kernel for a fully connected Network of LeakyIntegrateAndFireNeurons
 */
namespace SNN {

    namespace Kernels {

        namespace GPU {
                
            /**
             */
            __device__ void leakyIntegrateAndFireKernel(

                /* the number of neurons */
                unsigned numNeurons,
                
                /* neuron spike threshold */
                FloatType spikeThreshold,

                /* neuron voltage decay factor */
                FloatType voltageDecay,

                /* neuron refactory period */
                FloatType refactoryPeriod,

                /* the derivative dumping factor */
                FloatType derivativeDumpingFactor,

                /* the neurons spikes */
                FloatType *spikes,

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
                    const FloatType v_ = v[i] * voltageDecay + I[i] - spikes[i] * spikeThreshold;

                    /* update number of timesteps that have pased since the last spike */
                    const FloatType timeStepsSinceLastSpike_ = spikes[i] ? 1 : timeStepsSinceLastSpike[i] + 1;

                    /* update filtered spikes */
                    filteredSpikes[i] = filteredSpikes[i] * voltageDecay + spikes[i];

                    /* pseudo-derivative is fixed to zero within the refactory period */
                    if (timeStepsSinceLastSpike_ > refactoryPeriod) {
#if FloatTypeSize == 32
                        pseudoDerivative[i] = derivativeDumpingFactor * fmaxf(
                            0.0, 
                            1.0 - fabsf((v_ - spikeThreshold) / spikeThreshold)
                        );
#else
                        pseudoDerivative[i] = derivativeDumpingFactor * fmax(
                            0.0, 
                            1.0 - fabs((v_ - spikeThreshold) / spikeThreshold)
                        );
#endif
                    } else 
                        pseudoDerivative[i] = 0;
                    
                    /* set spike status */
                    if (timeStepsSinceLastSpike_ > refactoryPeriod && v_ > spikeThreshold) {
                        spikes[i] = 1;
                        numSpikes[i]++;
                    } else 
                        spikes[i] = 0;
                    
                    /* set values afterwards */
                    timeStepsSinceLastSpike[i] = timeStepsSinceLastSpike_;
                    v[i] = v_;
                    I[i] = 0;
                }
            }
        }
    }
}
#endif /* __LEAKY_INTEGRATE_AND_FIRE_KERNEL__ */
