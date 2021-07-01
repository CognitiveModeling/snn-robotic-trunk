#ifndef __LONG_SHORT_TERM_MEMORY_SPARSE_KERNEL_CALLER__
#define __LONG_SHORT_TERM_MEMORY_SPARSE_KERNEL_CALLER__
#include "BasicGPUArray.h"
#include "LongShortTermMemorySparseKernelDefines.h"
#include <vector>
/**
 * Parallel LONG_SHORT_TERM_MEMORY Kernel for a fully connected Network of neurons
 * SpikePropagation version
 */
namespace SNN {

    namespace Kernels {

        namespace GPU {

            class LongShortTermMemorySparseKernelCaller {

                private:

                    /* execution flags */
                    unsigned *executionFlags;

                    /* the number of GPU blocks (also batchsize) */
                    unsigned numBlocks;

                    /* the number of GPU threads */
                    unsigned numThreads;

                    /* the number of hidden neurons */
                    unsigned numHidden;

                    /* the number of output neurons */
                    unsigned numOutputNeurons;

                    /* the network errors for each output */
                    FloatType *outputError;

                    /* the network summed target for each output */
                    FloatType *summedTargets;

                    /* the network squared summed target for each output */
                    FloatType *squaredSummedTargets;

                    /* the number of values summed for each output */
                    FloatType *numSummedValues;

                    /* the networks classification accuracy error */
                    FloatType *classificationAccuracyCPU;

                    /* the networks number of classification samples */
                    FloatType *classificationSamplesCPU;

                    /* flags defining execution modules */
                    BasicGPUArray *flags;

                    /* the number of input neurons */
                    BasicGPUArray *numInputs;

                    /* the number of (leaky integrate and fire) hiddem neurons */
                    BasicGPUArray *numStandartHidden;

                    /* the number of (adaptive leaky integrate and fire) hidden neurons */
                    BasicGPUArray *numAdaptiveHidden;

                    /* the number of output neurons */
                    BasicGPUArray *numOutputs;

                    /* the batch size */
                    BasicGPUArray *batchSize;

                    /* the number of simmulation time steps */
                    BasicGPUArray *numSimulationTimesteps;

                    /* the simulation timestep length */
                    BasicGPUArray *timeStepLength;

                    /* neuron spike threshold */
                    BasicGPUArray *spikeThreshold;

                    /* neuron refactory period */
                    BasicGPUArray *refactoryPeriod;

                    /* the hidden voltage decay factor */
                    BasicGPUArray *hiddenDecayFactor;

                    /* the readout voltage decay factor */
                    BasicGPUArray *readoutDecayFactor;

                    /* the decay factor for the adaptive threshold */
                    BasicGPUArray *adaptationDecayFactor;
                    
                    /* the factor about which the base threshold increases */
                    BasicGPUArray *thresholdIncreaseConstant;

                    /* the target firing rate */
                    BasicGPUArray *targetFiringRate;

                    /* the firing rate gradient scalling factor */
                    BasicGPUArray *firingRateScallingFactor;

                    /* the derivative dumping factor */
                    BasicGPUArray *derivativeDumpingFactor;

                    /* the input neuron spikes over one simulation run */
                    BasicGPUArray *inputSpikesOverTime;

                   /* the input and hidden neuron spikes over one simulation run */
                    BasicGPUArray *spikesOverTime;

                    /* the hidden neurons firing rates */
                    BasicGPUArray *firingRates;

                    /* the hidden neurons number of spikes */
                    BasicGPUArray *numSpikes;

                    /* the number input synaptes to leaky integrate and fire neurons */
                    BasicGPUArray *numStandartInputSynapses;

                    /* the number hidden synaptes to leaky integrate and fire neurons */
                    BasicGPUArray *numStandartHiddenSynapses;

                    /* the number input synaptes to adaptive leaky integrate and fire neurons */
                    BasicGPUArray *numAdaptiveInputSynapses;

                    /* the number hidden synaptes to adaptive leaky integrate and fire neurons */
                    BasicGPUArray *numAdaptiveHiddenSynapses;

                    /* the number synaptic input weights */
                    BasicGPUArray *numInputWeights;

                    /* the number synaptic hidden weights */
                    BasicGPUArray *numHiddenWeights;

                    /* the number synaptic output weights */
                    BasicGPUArray *numOutputWeights;

                    /* the number of feedback weights */
                    BasicGPUArray *numFeedbackWeights;

                    /* the synaptic input weights */
                    BasicGPUArray *inputWeights;

                    /* the synaptic input weights */
                    BasicGPUArray *hiddenWeights;

                    /* the synaptic input weights */
                    BasicGPUArray *outputWeights;

                    /* the feedback weights */
                    BasicGPUArray *feedbackWeights;

                    /* the synaptic input weight input indices */
                    BasicGPUArray *inputWeightsIn;

                    /* the synaptic hidden weight input indices */
                    BasicGPUArray *hiddenWeightsIn;

                    /* the synaptic output weight input indices */
                    BasicGPUArray *outputWeightsIn;

                    /* the feedback weight input indices */
                    BasicGPUArray *feedbackWeightsIn;

                    /* the synaptic input weight output indices */
                    BasicGPUArray *inputWeightsOut;

                    /* the synaptic hidden weight output indices */
                    BasicGPUArray *hiddenWeightsOut;

                    /* the synaptic output weight output indices */
                    BasicGPUArray *outputWeightsOut;

                    /* the feedback weight output indices */
                    BasicGPUArray *feedbackWeightsOut;

                    /* the network target weights */
                    BasicGPUArray *targetWeights;

                    /* the network targets fore one simulation run */
                    BasicGPUArray *targetsOverTime;

                    /* the network outputs fore one simulation run */
                    BasicGPUArray *outputsOverTime;

                    /* the network derivatives fore one simulation run */
                    BasicGPUArray *derivativesOverTime;

                    /* the networks delta errors for one simulation run */
                    BasicGPUArray *deltaErrorsOverTime;

                    /* the network error mask for one simulation run */
                    BasicGPUArray *errorMaskOverTime;

                    /* the network error factors for one simulation run */
                    BasicGPUArray *outputErrorFactorOverTime;

                    /* the fixed braodcast gradients for input synapses */
                    BasicGPUArray *inputFixedBroadcastGradients;

                    /* the firing rate gradients for input synapses  */
                    BasicGPUArray *inputFiringRateGradients;

                    /* the fixed braodcast gradients for hidden synapses */
                    BasicGPUArray *hiddenFixedBroadcastGradients;

                    /* the firing rate gradients for hidden synapses */
                    BasicGPUArray *hiddenFiringRateGradients;

                    /* the leaky readout gradients */
                    BasicGPUArray *leakyReadoutGradients;

                    /* the networks summed error */
                    BasicGPUArray *networkError;

                    /* the networks summed network targets */
                    BasicGPUArray *networkTargets;

                    /* the networks squared summed network targets */
                    BasicGPUArray *networkSquaredTargets;

                    /* the number of values summed for each output */
                    BasicGPUArray *summedValues;

                    /* the networks classification accuracy error */
                    BasicGPUArray *classificationAccuracy;

                    /* the networks number of classification samples */
                    BasicGPUArray *classificationSamples;

                    /* summed network output for classification */
                    BasicGPUArray *summedActivation;

                    /***** content managed by kernel ******/

                    /* the filtered eligibility traces */
                    BasicGPUArray *filteredEligibilityTraces;

                    /* the filtered hidden spikes */
                    BasicGPUArray *filteredSpikes;

                    /* the neurons adaptation values */
                    BasicGPUArray *thresholdAdaptation;

                    /* the adaption eligibility part */
                    BasicGPUArray *adaptionEligibility;

                    /* hidden derivatives */
                    BasicGPUArray *derivatives;

                    /* input current for hidden and output neurons */
                    BasicGPUArray *I;

                    /* hidden and readout voltage */
                    BasicGPUArray *v;

                    /* hidden spikes */
                    BasicGPUArray *hiddenSpikes;

                    /* time since last spike for hidden neurons */
                    BasicGPUArray *timeStepsSinceLastSpike;

                    /* hidden neuron learn signals for one simulation run */
                    BasicGPUArray *learnSignalsOverTime;

                    /* hidden neuron delta errors (voltage component) */
                    BasicGPUArray *deltaErrorsVoltage;

                    /* hidden neuron delta errors (adaptation component) */
                    BasicGPUArray *deltaErrorsAdaption;

                    /* the input errors for one simmulation run */
                    BasicGPUArray *inputErrorsOverTime;

                    /* the network output errors for one simulation run */
                    BasicGPUArray *outputErrorsOverTime;

                    /* the input errors for one simmulation run (for all batches) */
                    BasicGPUArray *allInputErrorsOverTime;

                    /* the filtered (back propagated) output errors */
                    BasicGPUArray *filteredOutputErrors;

                public:

                    /* constructor */
                    LongShortTermMemorySparseKernelCaller(
                        unsigned batchSize,
                        unsigned numInputs,
                        unsigned numHidden,
                        unsigned numStandartHidden,
                        unsigned numAdaptiveHidden,
                        unsigned numOutputs,
                        unsigned numSimulationTimesteps,
                        FloatType timeStepLength,
                        FloatType spikeThreshold,
                        FloatType refactoryPeriod,
                        FloatType hiddenDecayFactor,
                        FloatType readoutDecayFactor,
                        FloatType adaptationDecayFactor,
                        FloatType thresholdIncreaseConstant,
                        FloatType targetFiringRate,
                        FloatType firingRateScallingFactor,
                        FloatType derivativeDumpingFactor,
                        std::vector<FloatType *> inputSpikesOverTime,
                        std::vector<FloatType *> spikesOverTime,
                        FloatType *firingRates,
                        unsigned numStandartInputSynapses,
                        unsigned numStandartHiddenSynapses,
                        unsigned numAdaptiveInputSynapses,
                        unsigned numAdaptiveHiddenSynapses,
                        unsigned numInputWeights,
                        unsigned numHiddenWeights,
                        unsigned numOutputWeights,
                        unsigned numFeedbackWeights,
                        FloatType *inputWeights,
                        FloatType *hiddenWeights,
                        FloatType *outputWeights,
                        FloatType *feedbackWeights,
                        unsigned *inputWeightsIn,
                        unsigned *hiddenWeightsIn,
                        unsigned *outputWeightsIn,
                        unsigned *feedbackWeightsIn,
                        unsigned *inputWeightsOut,
                        unsigned *hiddenWeightsOut,
                        unsigned *outputWeightsOut,
                        unsigned *feedbackWeightsOut,
                        FloatType *targetWeights,
                        std::vector<FloatType *> targetsOverTime,
                        std::vector<FloatType *> outputsOverTime,
                        std::vector<FloatType *> errorMaskOverTime,
                        std::vector<FloatType *> outputErrorFactorOverTime,
                        FloatType *inputGradients,
                        FloatType *inputFiringRateGradients,
                        FloatType *hiddenGradients,
                        FloatType *hiddenFiringRateGradients,
                        FloatType *leakyReadoutGradients,
                        FloatType *inputErrorsOverTime,
                        std::vector<FloatType *> outputErrorsOverTime,
                        std::vector<FloatType *> allInputErrorsOverTime,
                        std::vector<FloatType *> deltaErrorsOverTime
                    );

                    /* destructor */
                    ~LongShortTermMemorySparseKernelCaller();

                    /* runs the kernel of this (blocks untill finished) */
                    void runAndWait();

                    /* returns the networks squared summed error for the last run and the given output */
                    FloatType getSampleSquaredSummedError(unsigned i);
                    FloatType getOutputSquaredSummedError(unsigned i);
                    FloatType getSquaredSummedError();

                    /* returns the networks summed target for the last run and the given output */
                    FloatType getOutputSummedTarget(unsigned i);
                    FloatType getSummedTarget();

                    /* returns the networks squared summed target for the last run and the given output */
                    FloatType getOutputSquaredSummedTarget(unsigned i);
                    FloatType getSquaredSummedTarget();

                    /* returns the the number of summed values for the last run and the given output */
                    FloatType getOutputNumSummedValues(unsigned i);
                    FloatType getNumSummedValues();

                    /* reload feedback weights and other "not changing" values into device */
                    void reload();

                    /* sets / unsets the given flag */
                    void setFlag(unsigned flag) { *this->executionFlags |= flag; }
                    void unsetsetFlag(unsigned flag) { *this->executionFlags &= ~flag; }
                    void clearFlags() { *this->executionFlags = 0; }

                    /* sets all flags for normal eligibility traiing */
                    void setEligibilityTraining();

                    /* sets all flags for normal backpropagation traiing */
                    void setBackPropagationTraining();

                    /* sets all flags for elibibility backpropagation traiing */
                    void setEligibilityBackPropagationTraining();

                    /* sets all flags for t forward pass evaluation */
                    void setForwardPass();

                    /* returns the classification accuracy of the network */
                    FloatType getAccuracy();
    
                    /* sets the current active device */
                    static void setDevice(int device);
    
            };
        }
    }
}
#endif /* __LONG_SHORT_TERM_MEMORY_SPARSE_KERNEL_CALLER__ */
