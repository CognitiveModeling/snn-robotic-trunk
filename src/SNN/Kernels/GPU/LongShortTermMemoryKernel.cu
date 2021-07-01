#ifndef __LONG_SHORT_TERM_MEMORY_KERNEL__
#define __LONG_SHORT_TERM_MEMORY_KERNEL__
#include "LongShortTermMemoryFixedBroadcastKernel.cu"
#include "LongShortTermMemoryBackPropagationForwardPassKernel.cu"
#include "LongShortTermMemoryBackPropagationBackwardPassKernel.cu"
#include "LongShortTermMemoryInputErrorKernel.cu"
/**
 * Parallel kernel for a Long Short Term Spiking Network
 */
namespace SNN {

    namespace Kernels {

        namespace GPU {

            __global__ void longShortTermMemoryKernel(

                /* wether to use back propagation or e-prop 1 */
                int *useBackPropagation,

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

                /* the simulation start and end time */
                int *startTime, 
                int *endTime,

                /* the error mode of this */
                unsigned *errorMode,

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

                /* the synaptic input weights */
                FloatType *inputWeights,

                /* the synaptic hidden weights */
                FloatType *hiddenWeights,

                /* the synaptic output weights */
                FloatType *outputWeights,

                /* the feedback weights */
                FloatType *feedbackWeights,

                /* the network target weights */
                FloatType *targetWeights,

                /* the network targets fore one simulation run */
                FloatType *targetsOverTime,

                /* the network outputs fore one simulation run */
                FloatType *outputsOverTime,

                /* the network output errors for one simulation run */
                FloatType *outputErrorsOverTime,

                /* the network derivatives for one simulation run */
                FloatType *derivativesOverTime,

                /* the network derivatives for the last simulation run */
                FloatType *oldDerivativesOverTime,

                /* the network hidden voltage for the last simulation run */
                FloatType *voltageOverTime,

                /* time since last spike for hidden neurons over time */
                FloatType *timeStepsSinceLastSpikeOverTime,

                /* the neurons adaptation values over time */
                FloatType *thresholdAdaptationOverTime,

                /* the network error mask for one simulation run */
                FloatType *errorMaskOverTime,

                /* the network error factors for one simulation run */
                FloatType *outputErrorFactorOverTime,

                /* the gradients for input synapses */
                FloatType *inputGradients,

                /* the firing rate gradients for input synapses  */
                FloatType *inputFiringRateGradients,

                /* the gradients for hidden synapses */
                FloatType *hiddenGradients,

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

                /***** content managed by kernel ******/

                /* the filtered eligibility traces */
                FloatType *filteredEligibilityTraces,

                /* the filtered hidden spikes */
                FloatType *filteredSpikes,

                /* the filtered hidden spikes (by the readout decay factor) */
                FloatType *readoutDecayFilteredSpikes,

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

                /* hidden neuron delta errors (voltage component) */
                FloatType *deltaErrorsVoltage,

                /* hidden neuron delta errors (adaptation component) */
                FloatType *deltaErrorsAdaption,

                /* the input errors fore one simmulation run */
                FloatType *inputErrorsOverTime,

                /* the input errors fore one simmulation run (for all batches) */
                FloatType *allInputErrorsOverTime,

                /* the filtered (back propagated) output errors */
                FloatType *filteredOutputErrors,

                /* summed network output for classification */
                FloatType *summedActivation

            ) {

                if (*useBackPropagation >= 1) {

                    if (*useBackPropagation == 1 || *useBackPropagation == 2) {
                        longShortTermMemoryBackPropagationForwardPassKernel(
                            *numInputs,
                            *numStandartHidden,
                            *numAdaptiveHidden,
                            *numOutputs,
                            *numSimulationTimesteps,
                            *startTime,
                            *endTime,
                            *errorMode,
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
                            inputWeights,
                            hiddenWeights,
                            outputWeights,
                            targetWeights,
                            targetsOverTime,
                            outputsOverTime,
                            derivativesOverTime,
                            oldDerivativesOverTime,
                            voltageOverTime,
                            timeStepsSinceLastSpikeOverTime,
                            thresholdAdaptationOverTime,
                            errorMaskOverTime,
                            outputErrorFactorOverTime,
                            inputGradients,
                            inputFiringRateGradients,
                            hiddenGradients,
                            hiddenFiringRateGradients,
                            leakyReadoutGradients,
                            networkError,
                            summedTargets,
                            squaredSummedTargets,
                            numSummedValues,
                            classificationAccuracy,
                            classificationSamples,
                            filteredSpikes,
                            readoutDecayFilteredSpikes,
                            thresholdAdaptation,
                            derivatives,
                            I,
                            v,
                            hiddenSpikes,
                            timeStepsSinceLastSpike,
                            deltaErrorsVoltage,
                            deltaErrorsAdaption,
                            allInputErrorsOverTime,
                            filteredOutputErrors,
                            summedActivation
                        );
                    }
                    if (*useBackPropagation == 1 || *useBackPropagation == 3) {
                        longShortTermMemoryBackPropagationBackwardPassKernel(
                            *numInputs,
                            *numStandartHidden,
                            *numAdaptiveHidden,
                            *numOutputs,
                            *numSimulationTimesteps,
                            *startTime,
                            *endTime,
                            *errorMode,
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
                            inputWeights,
                            hiddenWeights,
                            outputWeights,
                            feedbackWeights,
                            targetWeights,
                            targetsOverTime,
                            outputsOverTime,
                            outputErrorsOverTime,
                            derivativesOverTime,
                            oldDerivativesOverTime,
                            voltageOverTime,
                            timeStepsSinceLastSpikeOverTime,
                            thresholdAdaptationOverTime,
                            errorMaskOverTime,
                            outputErrorFactorOverTime,
                            inputGradients,
                            inputFiringRateGradients,
                            hiddenGradients,
                            hiddenFiringRateGradients,
                            leakyReadoutGradients,
                            networkError,
                            summedTargets,
                            squaredSummedTargets,
                            numSummedValues,
                            classificationAccuracy,
                            classificationSamples,
                            filteredSpikes,
                            readoutDecayFilteredSpikes,
                            thresholdAdaptation,
                            derivatives,
                            I,
                            v,
                            hiddenSpikes,
                            timeStepsSinceLastSpike,
                            deltaErrorsVoltage,
                            deltaErrorsAdaption,
                            allInputErrorsOverTime,
                            filteredOutputErrors,
                            summedActivation
                        );
                    }
                } else {

                    longShortTermMemoryFixedBroadcastKernel(
                        *numInputs,
                        *numStandartHidden,
                        *numAdaptiveHidden,
                        *numOutputs,
                        *numSimulationTimesteps,
                        *startTime,
                        *endTime,
                        *errorMode,
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
                        inputWeights,
                        hiddenWeights,
                        outputWeights,
                        feedbackWeights,
                        targetWeights,
                        targetsOverTime,
                        outputsOverTime,
                        oldDerivativesOverTime,
                        voltageOverTime,
                        timeStepsSinceLastSpikeOverTime,
                        thresholdAdaptationOverTime,
                        errorMaskOverTime,
                        outputErrorFactorOverTime,
                        inputGradients,
                        inputFiringRateGradients,
                        hiddenGradients,
                        hiddenFiringRateGradients,
                        leakyReadoutGradients,
                        networkError,
                        summedTargets,
                        squaredSummedTargets,
                        numSummedValues,
                        classificationAccuracy,
                        classificationSamples,
                        filteredEligibilityTraces,
                        filteredSpikes,
                        readoutDecayFilteredSpikes,
                        thresholdAdaptation,
                        adaptionEligibility,
                        allInputErrorsOverTime,
                        derivatives,
                        I,
                        v,
                        hiddenSpikes,
                        timeStepsSinceLastSpike,
                        learnSignals,
                        summedActivation
                    );
                }
            }
        }
    }
}
#endif /* __LONG_SHORT_TERM_MEMORY_KERNEL__ */
