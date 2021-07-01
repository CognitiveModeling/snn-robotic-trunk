#ifndef __LONG_SHORT_TERM_MEMORY_BACK_PROPAGATION_BACKWARD_PASS_KERNEL__
#define __LONG_SHORT_TERM_MEMORY_BACK_PROPAGATION_BACKWARD_PASS_KERNEL__
#include "FullyConnectedLeakyIntegrateAndFireKernel.cu"
#include "FullyConnectedAdaptiveLeakyIntegrateAndFireKernel.cu"
#include "FullyConnectedBackPropagationInputOutputKernel.cu"
#include "FullyConnectedInputSpikePropagationKernel.cu"
#include "FullyConnectedHiddenSpikePropagationKernel.cu"
#include "LongShortTermMemoryLearnSignalKernel.cu"
#include "FullyConnectedDeltaErrorKernel.cu"
#include "FullyConnectedFastFiringRateKernel.cu"
#include "LongShortTermMemoryLeakyReadoutGradientKernel.cu"
#include "LongShortTermMemoryInputEligibilityLearnSignalErrorKernel.cu"
/**
 * Parallel kernel for a Long Short Term Spiking Network
 */
namespace SNN {

    namespace Kernels {

        namespace GPU {

#define ERROR_MODE_REGRESSION 0
#define ERROR_MODE_CLASSIFICATION 1
#define ERROR_MODE_INFINITY 2
#define ERROR_MODE_LEARNSIGNAL 3

            __device__ void longShortTermMemoryBackPropagationBackwardPassKernel(

                /* the number of input neurons */
                unsigned numInputs,

                /* the number of (leaky integrate and fire) hiddem neurons */
                unsigned numStandartHidden,

                /* the number of (adaptive leaky integrate and fire) hidden neurons */
                unsigned numAdaptiveHidden,

                /* the number of output neurons */
                unsigned numOutputs,

                /* the number of simmulation time steps */
                unsigned numSimulationTimesteps,

                /* the simulation start and end time */
                int startTime, 
                int endTime,

                /* the error mode of this */
                unsigned errorMode,

                /* the simulation timestep length */
                FloatType timeStepLength,

                /* neuron spike threshold */
                FloatType spikeThreshold,

                /* neuron refactory period */
                FloatType refactoryPeriod,

                /* the hidden voltage decay factor */
                FloatType hiddenDecayFactor,

                /* the readout voltage decay factor */
                FloatType readoutDecayFactor,

                /* the decay factor for the adaptive threshold */
                FloatType adaptationDecayFactor,
                
                /* the factor about which the base threshold increases */
                FloatType thresholdIncreaseConstant,

                /* the target firing rate */
                FloatType targetFiringRate,

                /* the firing rate gradient scalling factor */
                FloatType firingRateScallingFactor,

                /* the derivative dumping factor */
                FloatType derivativeDumpingFactor,

                /* the input neuron spikes over one simulation run */
                FloatType *inputSpikesOverTime,

                /* the input and hidden neuron spikes over one simulation run */
                FloatType *spikesOverTime,

                /* the hidden neurons firing rates */
                FloatType *firingRates,

                /* the number of spikes per neuron */
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

                /* the back propagation gradients for input synapses */
                FloatType *inputBackPropagationGradients,

                /* the firing rate gradients for input synapses  */
                FloatType *inputFiringRateGradients,

                /* the back propagation gradients for hidden synapses */
                FloatType *hiddenBackPropagationGradients,

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

                /* the filtered hidden spikes */
                FloatType *filteredSpikes,

                /* the filtered hidden spikes (by the readout decay factor) */
                FloatType *readoutDecayFilteredSpikes,

                /* the neurons adaptation values */
                FloatType *thresholdAdaptation,

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

                /* hidden neuron delta errors (voltage component) */
                FloatType *deltaErrorsVoltage,

                /* hidden neuron delta errors (adaptation component) */
                FloatType *deltaErrorsAdaption,

                /* the input errors fore one simmulation run */
                FloatType *inputErrorsOverTime,

                /* the filtered (back propagated) output errors */
                FloatType *filteredOutputErrors,

                /* summed network output for classification */
                FloatType *summedActivation

            ) {

                const unsigned numHidden = numStandartHidden + numAdaptiveHidden;
                cudaAssert(max(max(numInputs, numHidden), numOutputs) == blockDim.x);

                inputSpikesOverTime             += blockIdx.x * numInputs * numSimulationTimesteps;
                spikesOverTime                  += blockIdx.x * (numInputs + numHidden) * numSimulationTimesteps;
                numSpikes                       += blockIdx.x * (numHidden + numInputs);
                targetsOverTime                 += blockIdx.x * numOutputs * numSimulationTimesteps;
                outputsOverTime                 += blockIdx.x * numOutputs * numSimulationTimesteps;
                outputErrorsOverTime            += blockIdx.x * numOutputs * numSimulationTimesteps;
                derivativesOverTime             += blockIdx.x * numHidden * numSimulationTimesteps;
                oldDerivativesOverTime          += blockIdx.x * numHidden * numSimulationTimesteps;
                voltageOverTime                 += blockIdx.x * numHidden * numSimulationTimesteps;
                timeStepsSinceLastSpikeOverTime += blockIdx.x * numHidden * numSimulationTimesteps;
                thresholdAdaptationOverTime     += blockIdx.x * numAdaptiveHidden * numSimulationTimesteps;
                errorMaskOverTime               += blockIdx.x * numSimulationTimesteps;
                outputErrorFactorOverTime       += blockIdx.x * numOutputs * numSimulationTimesteps;
                inputBackPropagationGradients   += blockIdx.x * numInputs * numHidden;
                inputFiringRateGradients        += blockIdx.x * numInputs * numHidden;
                hiddenBackPropagationGradients  += blockIdx.x * numHidden * numHidden;
                hiddenFiringRateGradients       += blockIdx.x * numHidden * numHidden;
                leakyReadoutGradients           += blockIdx.x * numHidden * numOutputs;
                networkError                    += blockIdx.x;
                summedTargets                   += blockIdx.x;
                squaredSummedTargets            += blockIdx.x;
                numSummedValues                 += blockIdx.x;
                readoutDecayFilteredSpikes      += blockIdx.x * numHidden;
                thresholdAdaptation             += blockIdx.x * numAdaptiveHidden;
                derivatives                     += blockIdx.x * numHidden;
                I                               += blockIdx.x * (numHidden + numOutputs);
                v                               += blockIdx.x * (numHidden + numOutputs);
                hiddenSpikes                    += blockIdx.x * numHidden;
                timeStepsSinceLastSpike         += blockIdx.x * numHidden;
                deltaErrorsVoltage              += blockIdx.x * numHidden;
                deltaErrorsAdaption             += blockIdx.x * numAdaptiveHidden;
                inputErrorsOverTime             += blockIdx.x * numInputs * numSimulationTimesteps;
                filteredOutputErrors            += blockIdx.x * numOutputs;
                summedActivation                += blockIdx.x * numOutputs;
                classificationAccuracy          += blockIdx.x;
                classificationSamples           += blockIdx.x;

                if (startTime < 0) startTime = 0;
                if (endTime   < 0) endTime   = numSimulationTimesteps;

                __syncthreads();
                const int i = threadIdx.x;

                for (unsigned index = 0; index + i < numInputs * numHidden; index += blockDim.x) {
                    inputFiringRateGradients[index + i]      = 0;
                    inputBackPropagationGradients[index + i] = 0;
                }
                for (unsigned index = 0; index + i < numHidden * numHidden; index += blockDim.x) {
                    hiddenFiringRateGradients[index + i]      = 0;
                    hiddenBackPropagationGradients[index + i] = 0;
                }
                for (unsigned index = 0; index + i < numHidden * numOutputs; index += blockDim.x) {
                    leakyReadoutGradients[index + i] = 0;
                }

                if (startTime > 0) {
                    startTime     = startTime       % numSimulationTimesteps;
                    endTime       = ((endTime - 1)  % numSimulationTimesteps) + 1;
                }

                /* backpropagation loop */
                for (int t = endTime - 1; t >= startTime; t--) {

                    if (i < numOutputs) {
                        if (errorMode != ERROR_MODE_LEARNSIGNAL) {
                            filteredOutputErrors[i] = filteredOutputErrors[i] * readoutDecayFactor + 
                                (outputsOverTime[t * numOutputs + i] - targetsOverTime[t * numOutputs + i]) *
                                errorMaskOverTime[t];
                        } else {
                            filteredOutputErrors[i] = filteredOutputErrors[i] * readoutDecayFactor + 
                                outputErrorsOverTime[t * numOutputs + i];
                        }
                    }
                    __syncthreads();

                    fullyConnectedDeltaErrorKernel(
                        numInputs,
                        numHidden,
                        numOutputs,
                        numStandartHidden,
                        numAdaptiveHidden,
                        spikeThreshold,
                        thresholdIncreaseConstant,
                        adaptationDecayFactor,
                        hiddenDecayFactor,
                        timeStepLength,
                        filteredOutputErrors,
                        targetWeights,
                        outputErrorFactorOverTime + t * numOutputs,
                        deltaErrorsVoltage,
                        deltaErrorsAdaption,
                        spikesOverTime + t * (numInputs + numHidden) + numInputs,
                        derivativesOverTime + t * numHidden,
                        voltageOverTime + t * numHidden,
                        hiddenWeights,
                        outputWeights
                    );

                    __syncthreads();

                    if (i < numInputs) {

                        FloatType inputError = 0;

                        /* compute input errors */
                        for (unsigned h = 0; h < numHidden; h++) 
                            inputError += inputWeights[i * numHidden + h] * deltaErrorsVoltage[h];
                        
                        inputErrorsOverTime[t * numInputs + i] = inputError;
                    }

                    /* add errors to input gradients */
                    for (unsigned index = i; index < numInputs * numHidden; index += blockDim.x) {
                        const unsigned h = index / numInputs; 
                        const unsigned i = index % numInputs; 

                        inputBackPropagationGradients[index] += 
                            deltaErrorsVoltage[h] * inputSpikesOverTime[t * numInputs + i];
                    }

                    /* add errors to hidden gradients */
                    if (t > 0) {
                        for (unsigned index = i; index < numHidden * numHidden; index += blockDim.x) {
                            const unsigned ho = index / numHidden; 
                            const unsigned hi = index % numHidden; 

                            if (spikesOverTime[(t - 1) * (numInputs + numHidden) + numInputs + hi] != 0)
                                hiddenBackPropagationGradients[index] += deltaErrorsVoltage[ho];
                        }
                    }

                    /* add errors to output gradients */
                    for (unsigned index = i; index < numHidden * numOutputs; index += blockDim.x) {
                        const unsigned o = index / numHidden; 
                        const unsigned h = index % numHidden; 

                        if (spikesOverTime[t * (numInputs + numHidden) + numInputs + h] != 0)
                            leakyReadoutGradients[index] += filteredOutputErrors[o];
                    }

                    __syncthreads();
                }

                fullyConnectedFastFiringRateKernel(
                    numInputs,
                    numHidden,
                    targetFiringRate,
                    firingRateScallingFactor,
                    numSpikes,
                    numSpikes + numInputs,
                    firingRates,
                    inputFiringRateGradients
                );
                fullyConnectedFastFiringRateKernel(
                    numHidden,
                    numHidden,
                    targetFiringRate,
                    firingRateScallingFactor,
                    numSpikes + numInputs,
                    numSpikes + numInputs,
                    firingRates,
                    hiddenFiringRateGradients
                );

                __syncthreads();
            }
        }
    }
}
#endif /* __LONG_SHORT_TERM_MEMORY_BACK_PROPAGATION_BACKWARD_PASS_KERNEL__ */
