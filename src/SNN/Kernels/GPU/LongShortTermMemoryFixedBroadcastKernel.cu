#ifndef __LONG_SHORT_TERM_MEMORY_FIXED_BROADCAST_KERNEL__
#define __LONG_SHORT_TERM_MEMORY_FIXED_BROADCAST_KERNEL__
#include "FullyConnectedLeakyIntegrateAndFireKernel.cu"
#include "FullyConnectedAdaptiveLeakyIntegrateAndFireKernel.cu"
#include "FullyConnectedInputOutputKernel.cu"
#include "FullyConnectedInputSpikePropagationKernel.cu"
#include "FullyConnectedHiddenSpikePropagationKernel.cu"
#include "LongShortTermMemoryLearnSignalKernel.cu"
#include "LongShortTermMemoryEligibilityGradientKernel.cu"
#include "LongShortTermMemoryLeakyReadoutGradientKernel.cu"
#include "LongShortTermMemoryInputEligibilityKernel.cu"
/**
 * Parallel kernel for a Long Short Term Spiking Network
 */
namespace SNN {

    namespace Kernels {

        namespace GPU {

#define ERROR_MODE_REGRESSION 0
#define ERROR_MODE_CLASSIFICATION 1
#define ERROR_MODE_INFINITY 2

            __device__ void longShortTermMemoryFixedBroadcastKernel(

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

                /* the network target weights */
                FloatType *targetWeights,

                /* the network targets fore one simulation run */
                FloatType *targetsOverTime,

                /* the network outputs fore one simulation run */
                FloatType *outputsOverTime,

                /* the network derivatives fore one simulation run */
                FloatType *derivativesOverTime,

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

                /* the input errors fore one simmulation run */
                FloatType *inputErrorsOverTime,

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

                /* summed network output for classification */
                FloatType *summedActivation

            ) {
                const unsigned numHidden = numStandartHidden + numAdaptiveHidden;
                cudaAssert(numInputs == blockDim.x || numHidden == blockDim.x);

                inputSpikesOverTime             += blockIdx.x * numInputs * numSimulationTimesteps;
                spikesOverTime                  += blockIdx.x * (numInputs + numHidden) * numSimulationTimesteps;
                numSpikes                       += blockIdx.x * (numHidden + numInputs) + numInputs;
                targetsOverTime                 += blockIdx.x * numOutputs * numSimulationTimesteps;
                outputsOverTime                 += blockIdx.x * numOutputs * numSimulationTimesteps;
                derivativesOverTime             += blockIdx.x * numHidden * numSimulationTimesteps;
                voltageOverTime                 += blockIdx.x * numHidden * numSimulationTimesteps;
                timeStepsSinceLastSpikeOverTime += blockIdx.x * numHidden * numSimulationTimesteps;
                thresholdAdaptationOverTime     += blockIdx.x * numAdaptiveHidden * numSimulationTimesteps;
                errorMaskOverTime               += blockIdx.x * numSimulationTimesteps;
                outputErrorFactorOverTime       += blockIdx.x * numOutputs * numSimulationTimesteps;
                inputFixedBroadcastGradients    += blockIdx.x * numInputs * numHidden;
                inputFiringRateGradients        += blockIdx.x * numInputs * numHidden;
                hiddenFixedBroadcastGradients   += blockIdx.x * numHidden * numHidden;
                hiddenFiringRateGradients       += blockIdx.x * numHidden * numHidden;
                leakyReadoutGradients           += blockIdx.x * numHidden * numOutputs;
                networkError                    += blockIdx.x;
                summedTargets                   += blockIdx.x;
                squaredSummedTargets            += blockIdx.x;
                numSummedValues                 += blockIdx.x;
                filteredEligibilityTraces       += blockIdx.x * (numInputs + numHidden) * numHidden;
                filteredSpikes                  += blockIdx.x * (numInputs + numHidden);
                readoutDecayFilteredSpikes      += blockIdx.x * numHidden;
                thresholdAdaptation             += blockIdx.x * numAdaptiveHidden;
                adaptionEligibility             += blockIdx.x * (numInputs + numHidden) * numAdaptiveHidden;
                inputErrorsOverTime             += blockIdx.x * numInputs * numSimulationTimesteps;
                derivatives                     += blockIdx.x * numHidden;
                I                               += blockIdx.x * (numHidden + numOutputs);
                v                               += blockIdx.x * (numHidden + numOutputs);
                hiddenSpikes                    += blockIdx.x * numHidden;
                timeStepsSinceLastSpike         += blockIdx.x * numHidden;
                learnSignals                    += blockIdx.x * numHidden;
                summedActivation                += blockIdx.x * numOutputs;
                classificationAccuracy          += blockIdx.x;
                classificationSamples           += blockIdx.x;

                if (startTime < 0) startTime = 0;
                if (endTime   < 0) endTime   = numSimulationTimesteps;

                /* clear values */
                const int i = threadIdx.x;
                if (startTime == 0) {
                    if (i < numHidden) {
                        filteredSpikes[i] = 0;
                        readoutDecayFilteredSpikes[i] = 0;
                    }
                }
                if (i < numHidden) {
                    numSpikes[i]      = 0;
                    hiddenSpikes[i]   = 0;
                    I[i]              = 0;
                    v[i]              = 0;
                    timeStepsSinceLastSpike[i] = 2 * refactoryPeriod;
                }

                if (i == 0) {
                    networkError[0] = 0;
                    summedTargets[0] = 0;
                    squaredSummedTargets[0] = 0;
                    numSummedValues[0] = 0;
                    classificationAccuracy[0] = 0;
                    classificationSamples[0] = 0;
                }

                if (startTime == 0 && i < numInputs) {
                    filteredSpikes[i + numHidden] = 0;
                }
                if (i < numOutputs) {
                    I[i + numHidden] = 0;
                    v[i + numHidden] = 0;
                }

                if (i < numAdaptiveHidden)
                    thresholdAdaptation[i] = 0;

                for (unsigned index = 0; index + i < numInputs * numHidden; index += blockDim.x) {
                    inputFiringRateGradients[index + i]     = 0;
                    inputFixedBroadcastGradients[index + i] = 0;
                }
                for (unsigned index = 0; index + i < numHidden * numHidden; index += blockDim.x) {
                    hiddenFiringRateGradients[index + i]     = 0;
                    hiddenFixedBroadcastGradients[index + i] = 0;
                }
                if (startTime == 0) {
                    for (unsigned index = 0; 
                        index + i < (numInputs + numHidden) * numAdaptiveHidden; 
                        index += blockDim.x) {

                        adaptionEligibility[index + i] = 0;
                    }
                    for (unsigned index = 0; 
                        index + i < (numInputs + numHidden) * numHidden; 
                        index += blockDim.x) {

                        filteredEligibilityTraces[index + i] = 0;
                    }
                }
                for (unsigned index = 0; index + i < numHidden * numOutputs; index += blockDim.x) {
                    leakyReadoutGradients[index + i] = 0;
                }

                if (i < numOutputs)
                    summedActivation[i] = 0;

                FloatType adaptiveThreadOffset = 0;
                if (wrapThreads(numStandartHidden) + wrapThreads(numAdaptiveHidden) <= blockDim.x)
                    adaptiveThreadOffset = wrapThreads(numStandartHidden);

                if (startTime > 0) {
                    int lastStart = (startTime - 1) % numSimulationTimesteps;
                    startTime     = startTime       % numSimulationTimesteps;
                    endTime       = ((endTime - 1)  % numSimulationTimesteps) + 1;
                    if (i < numHidden) {
                        hiddenSpikes[i] = spikesOverTime[lastStart * (numInputs + numHidden) + numInputs + i];
                        v[i]            = voltageOverTime[lastStart * numHidden + i];
                        timeStepsSinceLastSpike[i] = timeStepsSinceLastSpikeOverTime[lastStart * numHidden + i];
                    }
                    if (i < numOutputs) 
                        v[numHidden + i] = outputsOverTime[lastStart * numOutputs + i];
                    if (i < numAdaptiveHidden)
                        thresholdAdaptation[i] = thresholdAdaptationOverTime[lastStart * numAdaptiveHidden + i];

                    __syncthreads();
                    fullyConnectedHiddenSpikePropagationKernel(
                        numHidden,
                        numOutputs,
                        hiddenSpikes,
                        I,
                        hiddenWeights,
                        outputWeights
                    );
                }

                __syncthreads();

                for (unsigned t = startTime; t < endTime; t++) {

                    fullyConnectedInputSpikePropagationKernel(
                        numInputs,
                        numHidden,
                        inputSpikesOverTime + t * numInputs,
                        I,
                        inputWeights
                    );
                    __syncthreads();

                    fullyConnectedLeakyIntegrateAndFireKernel(
                        0,
                        numStandartHidden,
                        spikeThreshold,
                        hiddenDecayFactor,
                        refactoryPeriod,
                        derivativeDumpingFactor,
                        hiddenSpikes,
                        filteredSpikes + numInputs,
                        numSpikes,
                        I,
                        v,
                        derivatives,
                        timeStepsSinceLastSpike
                    );
                    fullyConnectedAdaptiveLeakyIntegrateAndFireKernel(
                        adaptiveThreadOffset,
                        numAdaptiveHidden,
                        spikeThreshold,
                        hiddenDecayFactor,
                        adaptationDecayFactor,
                        thresholdIncreaseConstant,
                        refactoryPeriod,
                        derivativeDumpingFactor,
                        hiddenSpikes + numStandartHidden,
                        thresholdAdaptation,
                        filteredSpikes + numInputs + numStandartHidden,
                        numSpikes + numStandartHidden,
                        I + numStandartHidden,
                        v + numStandartHidden,
                        derivatives + numStandartHidden,
                        timeStepsSinceLastSpike + numStandartHidden
                    );

                    __syncthreads();

                    fullyConnectedHiddenSpikePropagationKernel(
                        numHidden,
                        numOutputs,
                        hiddenSpikes,
                        I,
                        hiddenWeights,
                        outputWeights
                    );

                    __syncthreads();

                    fullyConnectedInputOutputKernel(
                        numInputs, 
                        numOutputs,
                        hiddenDecayFactor,
                        readoutDecayFactor,
                        I + numHidden, 
                        v + numHidden, 
                        inputSpikesOverTime + t * numInputs,
                        filteredSpikes
                    );

                    __syncthreads();

                    if (i < numOutputs)
                        outputsOverTime[t * numOutputs + i] = v[numHidden + i];

                    if (i < numHidden) {
                        derivativesOverTime[t * numHidden + i] = derivatives[i];
                        voltageOverTime[t * numHidden + i] = v[i];
                        spikesOverTime[t * (numInputs + numHidden) + numInputs + i] = hiddenSpikes[i];
                        timeStepsSinceLastSpikeOverTime[t * numHidden + i] = timeStepsSinceLastSpike[i];
                    }

                    if (i < numAdaptiveHidden)
                        thresholdAdaptationOverTime[t * numAdaptiveHidden + i] = thresholdAdaptation[i];

                    if (i < numInputs)
                        spikesOverTime[t * (numInputs + numHidden) + i] = inputSpikesOverTime[t * numInputs + i];

                    if (i < numHidden)
                        spikesOverTime[t * (numInputs + numHidden) + numInputs + i] = hiddenSpikes[i];

                    if (i == 0 && errorMaskOverTime[t] != 0) {
                        for (unsigned o = 0; o < numOutputs; o++) {
                            summedTargets[0] += targetsOverTime[t * numOutputs + o];
                            squaredSummedTargets[0] += pow(targetsOverTime[t * numOutputs + o], 2);
                            numSummedValues[0] += 1;
                        }
                        if (errorMode == ERROR_MODE_INFINITY) {
                            for (unsigned o = 0; o < numOutputs; o++) {
                                networkError[0] += exp(
                                    -1.0 * targetsOverTime[t * numOutputs + o] * v[numHidden + o]
                                );
                            }

                            for (unsigned o = 0; o < numOutputs; o++) 
                                summedActivation[o] += v[numHidden + o];

                            if (t + 1 == numSimulationTimesteps || errorMaskOverTime[t + 1] == 0) {

                                for (unsigned o = 0; o < numOutputs; o++) {
                                    const FloatType target = targetsOverTime[t * numOutputs + o];

                                    classificationSamples[0] += fabs(target);
                                    if (summedActivation[o] * target > 0)
                                        classificationAccuracy[0] += fabs(target);
                                }
                                    
                                for (unsigned o = 0; o < numOutputs; o++) 
                                    summedActivation[o] = 0;
                            }
                        } else if (errorMode == ERROR_MODE_CLASSIFICATION) {
                            FloatType expSum = 0;
                            for (unsigned o = 0; o < numOutputs; o++) 
                                expSum += exp(v[numHidden + o]);

                            for (unsigned o = 0; o < numOutputs; o++) {
                                const FloatType softmax = exp(v[numHidden + o]) / expSum;
                                networkError[0] -= targetsOverTime[t * numOutputs + o] *
                                                   log(softmax);
                            }

                            for (unsigned o = 0; o < numOutputs; o++) 
                                summedActivation[o] += v[numHidden + o];

                            if (t + 1 == numSimulationTimesteps || errorMaskOverTime[t + 1] == 0) {
                                unsigned maxNeuron = 0;

                                for (unsigned o = 1; o < numOutputs; o++) 
                                    if (summedActivation[o] > summedActivation[maxNeuron])
                                        maxNeuron = o;
                                    
                                classificationAccuracy[0] += targetsOverTime[t * numOutputs + maxNeuron];
                                classificationSamples[0]++;

                                for (unsigned o = 0; o < numOutputs; o++) 
                                    summedActivation[o] = 0;
                            }
                        } else if (errorMode == ERROR_MODE_REGRESSION) {
                            for (unsigned o = 0; o < numOutputs; o++) {
                                networkError[0] += pow(
                                    v[numHidden + o] - 
                                    targetsOverTime[t * numOutputs + o], 
                                    2
                                );
                            }
                        }
                    }

                    longShortTermMemoryLearnSignalKernel(
                        numHidden,
                        numOutputs,
                        errorMode,
                        v + numHidden,
                        targetWeights,
                        targetsOverTime + t * numOutputs,
                        errorMaskOverTime[t],
                        outputErrorFactorOverTime + t * numOutputs,
                        learnSignals,
                        feedbackWeights
                    );

                    __syncthreads();

                    longShortTermMemoryEligibilityGradientKernel(
                        numInputs,
                        numStandartHidden,
                        numAdaptiveHidden,
                        targetFiringRate,
                        firingRateScallingFactor,
                        readoutDecayFactor,
                        adaptationDecayFactor,
                        thresholdIncreaseConstant,
                        filteredSpikes,
                        firingRates,
                        derivatives,
                        learnSignals,
                        adaptionEligibility,
                        filteredEligibilityTraces,
                        inputFiringRateGradients,
                        inputFixedBroadcastGradients
                    );
                    longShortTermMemoryEligibilityGradientKernel(
                        numHidden,
                        numStandartHidden,
                        numAdaptiveHidden,
                        targetFiringRate,
                        firingRateScallingFactor,
                        readoutDecayFactor,
                        adaptationDecayFactor,
                        thresholdIncreaseConstant,
                        filteredSpikes + numInputs,
                        firingRates,
                        derivatives,
                        learnSignals,
                        adaptionEligibility + numInputs * numAdaptiveHidden,
                        filteredEligibilityTraces + numInputs * numHidden,
                        hiddenFiringRateGradients,
                        hiddenFixedBroadcastGradients
                    );
                    longShortTermMemoryLeakyReadoutGradientKernel(
                        numHidden,
                        numOutputs,
                        errorMode,
                        readoutDecayFactor,
                        readoutDecayFilteredSpikes, 
                        hiddenSpikes,
                        v + numHidden,
                        targetsOverTime + t * numOutputs,
                        errorMaskOverTime[t],
                        leakyReadoutGradients
                    );

                    __syncthreads();
                }
            }
        }
    }
}
#endif /* __LONG_SHORT_TERM_MEMORY_FIXED_BROADCAST_KERNEL__ */
