#ifndef __LONG_SHORT_TERM_MEMORY_SPARSE_FORWARD_PASS_KERNEL__
#define __LONG_SHORT_TERM_MEMORY_SPARSE_FORWARD_PASS_KERNEL__
#include "LeakyIntegrateAndFireKernel.cu"
#include "AdaptiveLeakyIntegrateAndFireKernel.cu"
#include "InputOutputKernel.cu"
#include "SparseConnectedInputSpikePropagationKernel.cu"
#include "SparseConnectedHiddenSpikePropagationKernel.cu"
#include "LongShortTermMemorySparseLearnSignalKernel.cu"
#include "LongShortTermMemorySparseEligibilityGradientKernel.cu"
#include "LongShortTermMemorySparseLeakyReadoutGradientKernel.cu"
#include "LongShortTermMemorySparseEligibilityBackPropagationGradientKernel.cu"
#include "LongShortTermMemorySparseKernelDefines.h"
#include "SparseDeltaErrorKernel.cu"
#include "SparseInputErrorKernel.cu"
#include "SparseBackPropagatedGradients.cu"
#include "SparseFastFiringRateKernel.cu"
#include "SparseOutputErrorKernel.cu"
/**
 * Parallel kernel for a Long Short Term Spiking Network
 */
namespace SNN {

    namespace Kernels {

        namespace GPU {

            __device__ void longShortTermMemorySparseForwardPassKernel(

                /* flags defining execution modules */
                unsigned flags,

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

                /* the number input synaptes to leaky integrate and fire neurons */
                unsigned numStandartInputSynapses,

                /* the number hidden synaptes to leaky integrate and fire neurons */
                unsigned numStandartHiddenSynapses,

                /* the number input synaptes to adaptive leaky integrate and fire neurons */
                unsigned numAdaptiveInputSynapses,

                /* the number hidden synaptes to adaptive leaky integrate and fire neurons */
                unsigned numAdaptiveHiddenSynapses,

                /* the number synaptic input weights */
                unsigned numInputWeights,

                /* the number synaptic hidden weights */
                unsigned numHiddenWeights,

                /* the number synaptic output weights */
                unsigned numOutputWeights,

                /* the number of feedback weights */
                unsigned numFeedbackWeights,

                /* the synaptic input weights */
                FloatType *inputWeights,

                /* the synaptic hidden weights */
                FloatType *hiddenWeights,

                /* the synaptic output weights */
                FloatType *outputWeights,

                /* the feedback weights */
                FloatType *feedbackWeights,

                /* the synaptic input weight input indices */
                unsigned *inputWeightsIn,

                /* the synaptic hidden weight input indices */
                unsigned *hiddenWeightsIn,

                /* the synaptic output weight input indices */
                unsigned *outputWeightsIn,

                /* the feedback weight input indices */
                unsigned *feedbackWeightsIn,

                /* the synaptic input weight output indices */
                unsigned *inputWeightsOut,

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
                FloatType *inputGradients,

                /* the firing rate gradients for input synapses  */
                FloatType *inputFiringRateGradients,

                /* the fixed braodcast gradients for hidden synapses */
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

                /* hidden neuron learn signals for one simulation run */
                FloatType *learnSignalsOverTime,

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
                const unsigned numHidden = numStandartHidden + numAdaptiveHidden;

                inputSpikesOverTime           += blockIdx.x * numInputs * numSimulationTimesteps;
                spikesOverTime                += blockIdx.x * (numInputs + numHidden) * numSimulationTimesteps;
                numSpikes                     += blockIdx.x * (numInputs + numHidden);
                targetsOverTime               += blockIdx.x * numOutputs * numSimulationTimesteps;
                outputsOverTime               += blockIdx.x * numOutputs * numSimulationTimesteps;
                derivativesOverTime           += blockIdx.x * numHidden * numSimulationTimesteps;
                deltaErrorsOverTime           += blockIdx.x * (numHidden + numAdaptiveHidden + numOutputs) * numSimulationTimesteps;
                errorMaskOverTime             += blockIdx.x * numSimulationTimesteps;
                outputErrorFactorOverTime     += blockIdx.x * numOutputs * numSimulationTimesteps;
                inputGradients                += blockIdx.x * numInputWeights;
                inputFiringRateGradients      += blockIdx.x * numInputWeights;
                hiddenGradients               += blockIdx.x * numHiddenWeights;
                hiddenFiringRateGradients     += blockIdx.x * numHiddenWeights;
                leakyReadoutGradients         += blockIdx.x * numOutputWeights;
                networkError                  += blockIdx.x * numOutputs;
                summedTargets                 += blockIdx.x * numOutputs;
                squaredSummedTargets          += blockIdx.x * numOutputs;
                numSummedValues               += blockIdx.x * numOutputs;
                classificationAccuracy        += blockIdx.x * numOutputs;
                classificationSamples         += blockIdx.x * numOutputs;
                summedActivation              += blockIdx.x * numOutputs;
                filteredEligibilityTraces     += blockIdx.x * (numInputWeights + numHiddenWeights);
                filteredSpikes                += blockIdx.x * (numInputs + numHidden);
                thresholdAdaptation           += blockIdx.x * numAdaptiveHidden;
                adaptionEligibility           += blockIdx.x * (numAdaptiveInputSynapses + numAdaptiveHiddenSynapses);
                derivatives                   += blockIdx.x * numHidden;
                I                             += blockIdx.x * (numHidden + numOutputs);
                v                             += blockIdx.x * (numHidden + numOutputs);
                hiddenSpikes                  += blockIdx.x * numHidden;
                timeStepsSinceLastSpike       += blockIdx.x * numHidden;
                learnSignalsOverTime          += blockIdx.x * numHidden * numSimulationTimesteps;
                deltaErrorsVoltage            += blockIdx.x * numHidden;
                deltaErrorsAdaption           += blockIdx.x * numAdaptiveHidden;
                inputErrorsOverTime           += blockIdx.x * numInputs * numSimulationTimesteps;
                filteredOutputErrors          += blockIdx.x * numOutputs;

                /* clear values */
                if (CONTAINS(flags, LSNN_INITIALIZE_FORWARD_PASS)) {
                    for (int i = threadIdx.x; i < numHidden; i += blockDim.x) {
                        filteredSpikes[i] = 0;
                        numSpikes[i]      = 0;
                        hiddenSpikes[i]   = 0;
                        I[i]              = 0;
                        v[i]              = 0;
                        timeStepsSinceLastSpike[i] = 2 * refactoryPeriod;
                    }

                    for (int i = threadIdx.x; i < numInputs; i += blockDim.x) {
                        filteredSpikes[i + numHidden] = 0;
                        numSpikes[i + numHidden]      = 0;
                    }
                    for (int i = threadIdx.x; i < numOutputs; i += blockDim.x) {
                        I[i + numHidden] = 0;
                        v[i + numHidden] = 0;
                    }
                    for (int i = threadIdx.x; i < numAdaptiveHidden; i += blockDim.x) {
                        thresholdAdaptation[i] = 0;
                    }
                    for (unsigned i = threadIdx.x; 
                        i < numAdaptiveInputSynapses + numAdaptiveHiddenSynapses;
                        i += blockDim.x) {

                        adaptionEligibility[i] = 0;
                    }
                    for (unsigned i = threadIdx.x;
                        i < numInputWeights + numHiddenWeights;
                        i += blockDim.x) {

                        filteredEligibilityTraces[i] = 0;
                    }

                    for (unsigned i = threadIdx.x; i < numHidden * numSimulationTimesteps; i += blockDim.x) 
                        learnSignalsOverTime[i] = 0;
                }

                if (CONTAINS(flags, LSNN_INITIALIZE_GRADIENTS)) {
                    for (unsigned i = threadIdx.x; i < numInputWeights; i += blockDim.x) {
                        inputFiringRateGradients[i] = 0;
                        inputGradients[i]           = 0;
                    }
                    for (unsigned i = threadIdx.x; i < numHiddenWeights; i += blockDim.x) {
                        hiddenFiringRateGradients[i] = 0;
                        hiddenGradients[i]           = 0;
                    }
                    for (unsigned i = threadIdx.x; i < numOutputWeights; i += blockDim.x) {
                        leakyReadoutGradients[i] = 0;
                    }
                }

                if (CONTAINS(flags, LSNN_INITIALIZE_BACKWARD_PASS)) {
                    for (int i = threadIdx.x; i < numOutputs; i += blockDim.x) {
                        filteredOutputErrors[i] = 0;
                    }
                    for (int i = threadIdx.x; i < numHidden; i += blockDim.x) {
                        deltaErrorsVoltage[i] = 0;
                    }
                    for (int i = threadIdx.x; i < numAdaptiveHidden; i += blockDim.x) {
                        deltaErrorsAdaption[i] = 0;
                    }
                    for (int i = threadIdx.x; i < numInputs * numSimulationTimesteps; i += blockDim.x) {
                        inputErrorsOverTime[i] = 0;
                    }
                }

                __syncthreads();
                    
                if (CONTAINS(flags, LSNN_FORWARD_PASS)) {
                    for (unsigned t = 0; t < numSimulationTimesteps; t++) {

                        sparseConnectedInputSpikePropagationKernel(
                            inputSpikesOverTime + t * numInputs,
                            I,
                            numInputWeights,
                            inputWeights,
                            inputWeightsIn,
                            inputWeightsOut
                        );

                        __syncthreads();
                       
                        leakyIntegrateAndFireKernel(
                            numStandartHidden,
                            spikeThreshold,
                            hiddenDecayFactor,
                            refactoryPeriod,
                            derivativeDumpingFactor,
                            hiddenSpikes,
                            filteredSpikes + numInputs,
                            numSpikes + numInputs,
                            I,
                            v,
                            derivatives,
                            timeStepsSinceLastSpike
                        );
                        adaptiveLeakyIntegrateAndFireKernel(
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
                            numSpikes + numInputs + numStandartHidden,
                            I + numStandartHidden,
                            v + numStandartHidden,
                            derivatives + numStandartHidden,
                            timeStepsSinceLastSpike + numStandartHidden
                        );

                        __syncthreads();

                        sparseConnectedHiddenSpikePropagationKernel(
                            numHidden,
                            hiddenSpikes,
                            I,
                            numHiddenWeights,
                            numOutputWeights,
                            hiddenWeights,
                            outputWeights,
                            hiddenWeightsIn,
                            outputWeightsIn,
                            hiddenWeightsOut,
                            outputWeightsOut
                        );

                        __syncthreads();

                        inputOutputKernel(
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

                        for (int i = threadIdx.x; i < numOutputs; i += blockDim.x)
                            outputsOverTime[t * numOutputs + i] = v[numHidden + i];

                        for (int i = threadIdx.x; i < numHidden; i += blockDim.x)
                            derivativesOverTime[t * numHidden + i] = derivatives[i];

                        for (int i = threadIdx.x; i < numInputs; i += blockDim.x) {
                            spikesOverTime[t * (numInputs + numHidden) + i] = inputSpikesOverTime[t * numInputs + i];
                            numSpikes[i] += inputSpikesOverTime[t * numInputs + i];
                        }

                        for (int i = threadIdx.x; i < numHidden; i += blockDim.x)
                            spikesOverTime[t * (numInputs + numHidden) + numInputs + i] = hiddenSpikes[i];


                        if (CONTAINS(flags, LSNN_FIXED_BROADCAST_LEARNSIGNAL)) {
                            longShortTermMemorySparseLearnSignalKernel(
                                flags,
                                numOutputs,
                                v + numHidden,
                                targetWeights,
                                targetsOverTime + t * numOutputs,
                                errorMaskOverTime[t],
                                outputErrorFactorOverTime + t * numOutputs,
                                learnSignalsOverTime + t * numHidden,
                                numFeedbackWeights,
                                CONTAINS(flags, LSNN_SYMETRIC_EPROP) ? outputWeights : feedbackWeights,
                                CONTAINS(flags, LSNN_SYMETRIC_EPROP) ? outputWeightsIn : feedbackWeightsIn,
                                CONTAINS(flags, LSNN_SYMETRIC_EPROP) ? outputWeightsOut : feedbackWeightsOut
                            );
                            
                            __syncthreads();
                        }

                        if (CONTAINS(flags, LSNN_ELIGIBILTY_GRADIENTS)) {
                            longShortTermMemorySparseEligibilityGradientKernel(
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
                                learnSignalsOverTime + t * numHidden,
                                numStandartInputSynapses,
                                numAdaptiveInputSynapses,
                                inputWeightsIn,
                                inputWeightsOut,
                                adaptionEligibility,
                                filteredEligibilityTraces,
                                inputFiringRateGradients,
                                inputGradients
                            );
                            longShortTermMemorySparseEligibilityGradientKernel(
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
                                learnSignalsOverTime + t * numHidden,
                                numStandartHiddenSynapses,
                                numAdaptiveHiddenSynapses,
                                hiddenWeightsIn,
                                hiddenWeightsOut,
                                adaptionEligibility + numAdaptiveInputSynapses,
                                filteredEligibilityTraces + numInputWeights,
                                hiddenFiringRateGradients,
                                hiddenGradients
                            );
                        }
                        if (CONTAINS(flags, LSNN_READOUT_FORWARD_GRAIENTS)) {
                            longShortTermMemorySparseLeakyReadoutGradientKernel(
                                flags,
                                numOutputs,
                                numOutputWeights,
                                outputWeightsIn,
                                outputWeightsOut,
                                filteredSpikes + numInputs,
                                hiddenSpikes,
                                readoutDecayFactor,
                                v + numHidden,
                                targetsOverTime + t * numOutputs,
                                errorMaskOverTime[t],
                                leakyReadoutGradients
                            );
                        }

                        __syncthreads();
                    }
                }

                for (int i = threadIdx.x; i < numOutputs; i += blockDim.x) {
                    networkError[i]           = 0;
                    summedTargets[i]          = 0;
                    squaredSummedTargets[i]   = 0;
                    numSummedValues[i]        = 0;
                    classificationAccuracy[i] = 0;
                    classificationSamples[i]  = 0;
                    summedActivation[i]       = 0;
                }
                __syncthreads();
                for (unsigned t = 0; t < numSimulationTimesteps; t++) {
                    if (errorMaskOverTime[t] != 0) {
                        FloatType *outputs = outputsOverTime + t * numOutputs;
                        FloatType *targets = targetsOverTime + t * numOutputs;
                        if (CONTAINS(flags, LSNN_REGRESSION_ERROR)) {
                            for (int i = threadIdx.x; i < numOutputs; i += blockDim.x) {
                                atomicAdd(networkError + i, pow(outputs[i] - targets[i], 2));
                                atomicAdd(summedTargets + i, targets[i]);
                                atomicAdd(squaredSummedTargets + i, pow(targets[i], 2));
                                atomicAdd(numSummedValues + i, 1);
                            }
                        } else if (CONTAINS(flags, LSNN_CLASSIFICATION_ERROR)) {
                            if (threadIdx.x == 0) {
                                FloatType expSum = 1e-9;
                                for (unsigned o = 0; o < numOutputs; o++) 
                                    expSum += exp(outputs[o]);

                                for (int i = 0; i < numOutputs; i++) {
                                    const FloatType softmax = exp(outputs[i]) / expSum;
                                    atomicAdd(networkError, targets[i] * log(softmax));
                                }
                                atomicAdd(numSummedValues, 1);

                                for (int i = 0; i < numOutputs; i++)
                                    atomicAdd(summedActivation + i, outputs[i]);

                                if (t + 1 == numSimulationTimesteps || errorMaskOverTime[t + 1] == 0) {
                                    unsigned maxNeuron = 0;

                                    for (unsigned o = 1; o < numOutputs; o++) 
                                        if (summedActivation[o] > summedActivation[maxNeuron])
                                            maxNeuron = o;
                                        
                                    atomicAdd(classificationAccuracy, targets[maxNeuron]);
                                    atomicAdd(classificationSamples, 1);

                                    for (int i = 0; i < numOutputs; i++)
                                        summedActivation[i] = 0;
                                }
                            }
                        }
                    }
                }

                if (CONTAINS(flags, LSNN_BACKWARD_PASS)) {
                    for (int t = numSimulationTimesteps - 1; t >= 0; t--) {
                        
                        __syncthreads();

                        /* output error calculation */
                        int offset  = t * (numHidden + numAdaptiveHidden + numOutputs);
                        offset     +=      numHidden + numAdaptiveHidden;

                        sparseOutputErrorKernel(
                            flags,
                            numOutputs,
                            readoutDecayFactor,
                            errorMaskOverTime[t],
                            outputErrorFactorOverTime + t * numOutputs,
                            targetWeights,
                            filteredOutputErrors,
                            outputErrorsOverTime + t * numOutputs,
                            outputsOverTime + t * numOutputs,
                            targetsOverTime + t * numOutputs
                        );
                        __syncthreads();
                        
                        /* hidden delta error calculation */
                        sparseDeltaErrorKernel(
                            numHidden,
                            numStandartHidden,
                            numHiddenWeights,
                            numOutputWeights,
                            spikeThreshold,
                            thresholdIncreaseConstant,
                            adaptationDecayFactor,
                            hiddenDecayFactor,
                            hiddenWeights,
                            outputWeights,
                            hiddenWeightsIn,
                            outputWeightsIn,
                            hiddenWeightsOut,
                            outputWeightsOut,
                            filteredOutputErrors,
                            deltaErrorsVoltage,
                            deltaErrorsAdaption,
                            derivativesOverTime + t * numHidden,
                            learnSignalsOverTime + t * numHidden
                        ); 

                        __syncthreads();

                        if (CONTAINS(flags, LSNN_INPUT_ERRORS)) {
                            sparseInputErrorKernel(
                                numInputWeights,
                                inputWeights,
                                inputWeightsIn,
                                inputWeightsOut,
                                deltaErrorsVoltage,
                                inputErrorsOverTime + t * numInputs
                            );
                        }

                        if (CONTAINS(flags, LSNN_BACKPROPAGATED_GRADIENTS)) {
                            sparseBackPropagatedGradients(
                                numInputWeights,
                                numHiddenWeights,
                                numOutputWeights,
                                inputWeights,
                                hiddenWeights,
                                outputWeights,
                                inputWeightsIn,
                                inputWeightsOut,
                                hiddenWeightsIn,
                                outputWeightsIn,
                                hiddenWeightsOut,
                                outputWeightsOut,
                                deltaErrorsVoltage,
                                filteredOutputErrors,
                                inputGradients,
                                hiddenGradients,
                                leakyReadoutGradients, 
                                inputSpikesOverTime + t * numInputs,
                                spikesOverTime + t * (numInputs + numHidden) + numInputs,
                                (t > 0) ? spikesOverTime + (t - 1) * (numInputs + numHidden) + numInputs : NULL
                            );
                        }
                    }

                    if(CONTAINS(flags, LSNN_BACKPROPAGATED_ELIGIBILITY_GRADIENTS)) {
                        for (int i = threadIdx.x; i < numInputs + numHidden; i += blockDim.x) {
                            filteredSpikes[i] = 0;
                        }
                        for (unsigned i = threadIdx.x; 
                            i < numAdaptiveInputSynapses + numAdaptiveHiddenSynapses;
                            i += blockDim.x) {

                            adaptionEligibility[i] = 0;
                        }
                        for (unsigned t = 0; t < numSimulationTimesteps; t++) {

                            for (int i = threadIdx.x; i < numInputs; i += blockDim.x) {
                                filteredSpikes[i] = 
                                    filteredSpikes[i] * hiddenDecayFactor + 
                                    spikesOverTime[t * (numInputs + numHidden) + i];
                            }
                            __syncthreads();

                            longShortTermMemorySparseEligibilityBackPropagationGradientKernel(
                                numInputs,
                                numStandartHidden,
                                numAdaptiveHidden,
                                targetFiringRate,
                                firingRateScallingFactor,
                                adaptationDecayFactor,
                                thresholdIncreaseConstant,
                                filteredSpikes,
                                firingRates,
                                derivativesOverTime + t * numHidden,
                                learnSignalsOverTime + t * numHidden,
                                numStandartInputSynapses,
                                numAdaptiveInputSynapses,
                                inputWeightsIn,
                                inputWeightsOut,
                                adaptionEligibility,
                                inputFiringRateGradients,
                                inputGradients
                            );
                            longShortTermMemorySparseEligibilityBackPropagationGradientKernel(
                                numHidden,
                                numStandartHidden,
                                numAdaptiveHidden,
                                targetFiringRate,
                                firingRateScallingFactor,
                                adaptationDecayFactor,
                                thresholdIncreaseConstant,
                                filteredSpikes + numInputs,
                                firingRates,
                                derivativesOverTime + t * numHidden,
                                learnSignalsOverTime + t * numHidden,
                                numStandartHiddenSynapses,
                                numAdaptiveHiddenSynapses,
                                hiddenWeightsIn,
                                hiddenWeightsOut,
                                adaptionEligibility + numAdaptiveInputSynapses,
                                hiddenFiringRateGradients,
                                hiddenGradients
                            );
                            __syncthreads();

                            for (int i = numInputs + threadIdx.x; i < numInputs + numHidden; i += blockDim.x) {
                                filteredSpikes[i] = 
                                    filteredSpikes[i] * hiddenDecayFactor + 
                                    spikesOverTime[t * (numInputs + numHidden) + i];
                            }
                        }
                    }

                    if (CONTAINS(flags, LSNN_FAST_FRINIG_RATE)) {
                        sparseFastFiringRateKernel(
                            numInputWeights,
                            numHiddenWeights,
                            targetFiringRate,
                            firingRateScallingFactor,
                            inputWeights,
                            hiddenWeights,
                            inputWeightsIn,
                            inputWeightsOut,
                            hiddenWeightsIn,
                            hiddenWeightsOut,
                            numSpikes, 
                            numSpikes + numInputs,
                            firingRates,
                            inputFiringRateGradients,
                            hiddenFiringRateGradients
                        );
                    }
                }
            }
        }
    }
}
#endif /* __LONG_SHORT_TERM_MEMORY_SPARSE_FORWARD_PASS_KERNEL__ */
