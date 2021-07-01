#ifndef __LONG_SHORT_TERM_MEMORY_SPARSE_KERNEL__
#define __LONG_SHORT_TERM_MEMORY_SPARSE_KERNEL__
#include "LongShortTermMemorySparseForwardPassKernel.cu"
/**
 * Parallel kernel for a Long Short Term Spiking Network
 */
namespace SNN {

    namespace Kernels {

        namespace GPU {

            __global__ void longShortTermMemorySparseKernel(

                /* flags defining execution modules */
                unsigned *flags,

                /* the number of input neurons */
                unsigned *numInputs,

                /* the number of (leaky integrate and fire) hiddem neurons */
                unsigned *numStandartHidden,

                /* the number of (adaptive leaky integrate and fire) hidden neurons */
                unsigned *numAdaptiveHidden,

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

                /* the decay factor for the adaptive threshold */
                FloatType *adaptationDecayFactor,
                
                /* the factor about which the base threshold increases */
                FloatType *thresholdIncreaseConstant,

                /* the target firing rate */
                FloatType *targetFiringRate,

                /* the firing rate gradient scalling factor */
                FloatType *firingRateScallingFactor,

                /* the derivative dumping factor */
                FloatType *derivativeDumpingFactor,

                /* the input neuron spikes over one simulation run */
                FloatType *inputSpikesOverTime,

               /* the input and hidden neuron spikes over one simulation run */
                FloatType *spikesOverTime,

                /* the hidden neurons firing rates */
                FloatType *firingRates,

                /* the hidden neurons number of spikes */
                FloatType *numSpikes,

                /* the number input synaptes to leaky integrate and fire neurons */
                unsigned *numStandartInputSynapses,

                /* the number hidden synaptes to leaky integrate and fire neurons */
                unsigned *numStandartHiddenSynapses,

                /* the number input synaptes to adaptive leaky integrate and fire neurons */
                unsigned *numAdaptiveInputSynapses,

                /* the number hidden synaptes to adaptive leaky integrate and fire neurons */
                unsigned *numAdaptiveHiddenSynapses,

                /* the number synaptic input weights */
                unsigned *numInputWeight,

                /* the number synaptic hidden weights */
                unsigned *numHiddenWeights,

                /* the number synaptic output weights */
                unsigned *numOutputWeights,

                /* the number of feedback weights */
                unsigned *numFeedbackWeights,

                /* the synaptic input weights */
                FloatType *inputWeights,

                /* the synaptic hidden weights */
                FloatType *hiddenWeights,

                /* the synaptic output weights */
                FloatType *outputWeights,

                /* the feedback weights */
                FloatType *feedbackWeights,

                /* the synaptic input weight input indices */
                unsigned *inputWeightIn,

                /* the synaptic hidden weight input indices */
                unsigned *hiddenWeightsIn,

                /* the synaptic output weight input indices */
                unsigned *outputWeightsIn,

                /* the feedback weight input indices */
                unsigned *feedbackWeightsIn,

                /* the synaptic input weight output indices */
                unsigned *inputWeightOut,

                /* the synaptic hidden weight output indices */
                unsigned *hiddenWeightsOut,

                /* the synaptic output weight output indices */
                unsigned *outputWeightsOut,

                /* the feedback weight output indices */
                unsigned *feedbackWeightsOut,

                /* the network target weights */
                FloatType *targetWeights,

                /* the network targets fore one simulation run */
                FloatType *targetsOverTime,

                /* the network outputs fore one simulation run */
                FloatType *outputsOverTime,

                /* the network derivatives fore one simulation run */
                FloatType *derivativesOverTime,

                /* the networks delta errors for one simulation run */
                FloatType *deltaErrorsOverTime,

                /* the network error mask for one simulation run */
                FloatType *errorMaskOverTime,

                /* the network error factors for one simulation run */
                FloatType *outputErrorFactorOverTime,

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

                /* the network summed target for each output */
                FloatType *summedTargets,

                /* the network squared summed target for each output */
                FloatType *squaredSummedTargets,

                /* the number of values summed for each output */
                FloatType *numSummedValues,

                /* the networks classification accuracy error */
                FloatType *classificationAccuracy,

                /* the networks number of classification samples */
                FloatType *classificationSamples,

                /* summed network output for classification */
                FloatType *summedActivation,

                /***** content managed by kernel ******/

                /* the filtered eligibility traces */
                FloatType *filteredEligibilityTraces,

                /* the filtered hidden spikes */
                FloatType *filteredSpikes,

                /* the neurons adaptation values */
                FloatType *thresholdAdaptation,

                /* the adaption eligibility part */
                FloatType *adaptionEligibility,

                /* hidden derivatives */
                FloatType *derivatives,

                /* input current for hidden and output neurons */
                FloatType *I,

                /* hidden and readout voltage */
                FloatType *v,

                /* hidden spikes */
                FloatType *hiddenSpikes,

                /* time since last spike for hidden neurons */
                FloatType *timeStepsSinceLastSpike,

                /* hidden neuron learn signals */
                FloatType *learnSignals,

                /* the filtered (back propagated) output errors */
                FloatType *filteredOutputErrors,

                /* hidden neuron delta errors (voltage component) */
                FloatType *deltaErrorsVoltage,

                /* hidden neuron delta errors (adaptation component) */
                FloatType *deltaErrorsAdaption,

                /* the input errors fore one simmulation run */
                FloatType *inputErrorsOverTime,

                /* the network output errors for one simulation run */
                FloatType *outputErrorsOverTime

            ) {

                longShortTermMemorySparseForwardPassKernel(
                    *flags,
                    *numInputs,
                    *numStandartHidden,
                    *numAdaptiveHidden,
                    *numOutputs,
                    *numSimulationTimesteps,
                    *timeStepLength,
                    *spikeThreshold,
                    *refactoryPeriod,
                    *hiddenDecayFactor,
                    *readoutDecayFactor,
                    *adaptationDecayFactor,
                    *thresholdIncreaseConstant,
                    *targetFiringRate,
                    *firingRateScallingFactor,
                    *derivativeDumpingFactor,
                    inputSpikesOverTime,
                    spikesOverTime,
                    firingRates,
                    numSpikes,
                    *numStandartInputSynapses,
                    *numStandartHiddenSynapses,
                    *numAdaptiveInputSynapses,
                    *numAdaptiveHiddenSynapses,
                    *numInputWeight,
                    *numHiddenWeights,
                    *numOutputWeights,
                    *numFeedbackWeights,
                    inputWeights,
                    hiddenWeights,
                    outputWeights,
                    feedbackWeights,
                    inputWeightIn,
                    hiddenWeightsIn,
                    outputWeightsIn,
                    feedbackWeightsIn,
                    inputWeightOut,
                    hiddenWeightsOut,
                    outputWeightsOut,
                    feedbackWeightsOut,
                    targetWeights,
                    targetsOverTime,
                    outputsOverTime,
                    derivativesOverTime,
                    deltaErrorsOverTime,
                    errorMaskOverTime,
                    outputErrorFactorOverTime,
                    inputFixedBroadcastGradients,
                    inputFiringRateGradients,
                    hiddenFixedBroadcastGradients,
                    hiddenFiringRateGradients,
                    leakyReadoutGradients,
                    networkError,
                    summedTargets,
                    squaredSummedTargets,
                    numSummedValues,
                    classificationAccuracy,
                    classificationSamples,
                    summedActivation,
                    filteredEligibilityTraces,
                    filteredSpikes,
                    thresholdAdaptation,
                    adaptionEligibility,
                    derivatives,
                    I,
                    v,
                    hiddenSpikes,
                    timeStepsSinceLastSpike,
                    learnSignals,
                    filteredOutputErrors,
                    deltaErrorsVoltage,
                    deltaErrorsAdaption,
                    inputErrorsOverTime,
                    outputErrorsOverTime
                );
            }
        }
    }
}
#endif /* __LONG_SHORT_TERM_MEMORY_SPARSE_KERNEL__ */
