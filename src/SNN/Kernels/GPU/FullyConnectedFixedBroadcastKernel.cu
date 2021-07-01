#ifndef __FULLY_CONNECTED_FIXED_BROADCAST_KERNEL__
#define __FULLY_CONNECTED_FIXED_BROADCAST_KERNEL__
#include "FullyConnectedLeakyIntegrateAndFireFixedBroadcastKernel.cu"
/**
 * Parallel FULLY_CONNECTED Kernel for a fully connected Network of neurons
 * SpikePropagation version
 */
namespace SNN {

    namespace Kernels {

        namespace GPU {

            __global__ void fullyConnectedFixedBroadcastKernel(

                /* the number of input neurons */
                unsigned *numInputs,

                /* the number of hidden neurons */
                unsigned *numHidden,

                /* the number of output neurons */
                unsigned *numOutputs,

                /* the number of simmulation time steps */
                unsigned *numSimulationTimesteps,

                /* the simulation timestep length */
                FloatType *timeStepLength,

                /* neuron spike threshold */
                FloatType *spikeThreshold,

                /* neuron refactory period */
                FloatType *refactoryPeriod,

                /* the hidden voltage decay factor */
                FloatType *hiddenDecayFactor,

                /* the readout voltage decay factor */
                FloatType *readoutDecayFactor,

                /* the target firing rate */
                FloatType *targetFiringRate,

                /* the firing rate gradient scalling factor */
                FloatType *firingRateScallingFactor,

                /* the derivative dumping factor */
                FloatType *derivativeDumpingFactor,

                /* the input neuron spikes over one simulation run */
                FloatType *inputSpikesOverTime,

                /* the hidden neurons firing rates */
                FloatType *firingRates,

                /* the hidden neurons number of spikes */
                FloatType *numSpikes,

                /* the synaptic input weights */
                FloatType *inputWeights,

                /* the synaptic input weights */
                FloatType *hiddenWeights,

                /* the synaptic input weights */
                FloatType *outputWeights,

                /* the feedback weights */
                FloatType *feedbackWeights,

                /* the network targets fore one simulation run */
                FloatType *targetsOverTime,

                /* the fixed braodcast gradients for input synapses */
                FloatType *inputFixedBroadcastGradients,

                /* the firing rate gradients for input synapses  */
                FloatType *inputFiringRateGradients,

                /* the fixed braodcast gradients for hidden synapses */
                FloatType *hiddenFixedBroadcastGradients,

                /* the firing rate gradients for hidden synapses */
                FloatType *hiddenFiringRateGradients,

                /* the leaky readout gradients */
                FloatType *leakyReadoutGradients,

                /* the networks squared summed error */
                FloatType *networkError,

                /***** content managed by kernel ******/

                /* the filtered eligibility traces */
                FloatType *filteredEligibilityTraces,

                /* the filtered hidden spikes */
                FloatType *filteredSpikes,

                /* hidden derivatives */
                FloatType *derivatives,

                /* input current for hidden and output neurons */
                FloatType *I,

                /* hidden and readout voltage */
                FloatType *v,

                /* hidden spikes */
                FloatType *hiddenSpikes,

                /* time since last spike for hidden neurons */
                FloatType *timeStepsSinceLastSpike
            ) {

                fullyConnectedLeakyIntegrateAndFireFixedBroadcastKernel(
                    *numInputs,
                    *numHidden,
                    *numOutputs,
                    *numSimulationTimesteps,
                    *timeStepLength,
                    *spikeThreshold,
                    *refactoryPeriod,
                    *hiddenDecayFactor,
                    *readoutDecayFactor,
                    *targetFiringRate,
                    *firingRateScallingFactor,
                    *derivativeDumpingFactor,
                    inputSpikesOverTime,
                    firingRates,
                    numSpikes,
                    inputWeights,
                    hiddenWeights,
                    outputWeights,
                    feedbackWeights,
                    targetsOverTime,
                    inputFixedBroadcastGradients,
                    inputFiringRateGradients,
                    hiddenFixedBroadcastGradients,
                    hiddenFiringRateGradients,
                    leakyReadoutGradients,
                    networkError,
                    filteredEligibilityTraces,
                    filteredSpikes,
                    derivatives,
                    I,
                    v,
                    hiddenSpikes,
                    timeStepsSinceLastSpike
                );
            }
        }
    }
}
#endif /* __FULLY_CONNECTED_FIXED_BROADCAST_KERNEL__ */
