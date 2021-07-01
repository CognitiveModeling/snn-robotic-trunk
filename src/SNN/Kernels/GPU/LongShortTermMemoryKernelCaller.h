#ifndef __LONG_SHORT_TERM_MEMORY_KERNEL_CALLER__
#define __LONG_SHORT_TERM_MEMORY_KERNEL_CALLER__
#define BACKPROPAGATION_OFF 0
#define BACKPROPAGATION_FULL 1
#define BACKPROPAGATION_FORWARD 2
#define BACKPROPAGATION_BACKWARD 3
#include "BasicGPUArray.h"
#include <vector>
/**
 * Parallel LONG_SHORT_TERM_MEMORY Kernel for a fully connected Network of neurons
 * SpikePropagation version
 */
namespace SNN {

    namespace Kernels {

        namespace GPU {

            class LongShortTermMemoryKernelCaller {

                private:

                    /* wether to use back propagation or e-prop 1 */
                    int *backPropagation;

                    /* the simmulation start and end time */
                    int *starttime, *endtime;

                    /* the number of GPU blocks */
                    unsigned numBlocks;

                    /* the number of GPU threads */
                    unsigned numThreads;

                    /* the number of hidden neurons */
                    unsigned numHiddenNeurons;
                    
                    /* the number of input neurons */
                    unsigned numInputNeurons;

                    /* the number of output Neurons */
                    unsigned numOutputNeurons;

                    /* the number of simulation timesteps */
                    unsigned numTimeSteps;

                    /* the networks classification accuracy error */
                    FloatType *classificationAccuracyCPU;

                    /* the networks number of classification samples */
                    FloatType *classificationSamplesCPU;

                    /* the network errors (one for each batch */
                    FloatType *batchErrors;

                    /* the network summed target for each batch */
                    FloatType *summedTargets;

                    /* the network squared summed target for each batch */
                    FloatType *squaredSummedTargets;

                    /* the number of values summed for each output */
                    FloatType *numSummedValues;

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

                    /* the processing start and end */
                    BasicGPUArray *startTime;
                    BasicGPUArray *endTime;

                    /* the error mode of this */
                    BasicGPUArray *errorMode;

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

                    /* the synaptic input weights */
                    BasicGPUArray *inputWeights;

                    /* the synaptic input weights */
                    BasicGPUArray *hiddenWeights;

                    /* the synaptic input weights */
                    BasicGPUArray *outputWeights;

                    /* the feedback weights */
                    BasicGPUArray *feedbackWeights;

                    /* the network target weights */
                    BasicGPUArray *targetWeights;

                    /* the network targets for one simulation run */
                    BasicGPUArray *targetsOverTime;

                    /* the network outputs for one simulation run */
                    BasicGPUArray *outputsOverTime;

                    /* the network output errors for one simulation run */
                    BasicGPUArray *outputErrorsOverTime;

                    /* the network derivatives for one simulation run */
                    BasicGPUArray *derivativesOverTime;

                    /* the network derivatives for the last simulation run */
                    BasicGPUArray *oldDerivativesOverTime;

                    /* the network hidden voltage for one simulation run */
                    BasicGPUArray *voltageOverTime;

                    /* time since last spike for hidden neurons over time  */
                    BasicGPUArray *timeStepsSinceLastSpikeOverTime;

                    /* the neurons adaptation values over time */
                    BasicGPUArray *thresholdAdaptationOverTime;

                    /* the network error mask for one simulation run */
                    BasicGPUArray *errorMaskOverTime;

                    /* the network error factors for one simulation run */
                    BasicGPUArray *outputErrorFactorOverTime;

                    /* the gradients for input synapses */
                    BasicGPUArray *inputGradients;

                    /* the firing rate gradients for input synapses  */
                    BasicGPUArray *inputFiringRateGradients;

                    /* the gradients for hidden synapses */
                    BasicGPUArray *hiddenGradients;

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

                    /***** content managed by kernel ******/

                    /* the filtered eligibility traces */
                    BasicGPUArray *filteredEligibilityTraces;

                    /* the filtered hidden spikes */
                    BasicGPUArray *filteredSpikes;

                    /* the filtered hidden spikes (by the readout decay factor) */
                    BasicGPUArray *readoutDecayFilteredSpikes;

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

                    /* hidden neuron learn signals */
                    BasicGPUArray *learnSignals;

                    /* hidden neuron delta errors (voltage component) */
                    BasicGPUArray *deltaErrorsVoltage;

                    /* hidden neuron delta errors (adaptation component) */
                    BasicGPUArray *deltaErrorsAdaption;

                    /* the input errors for one simmulation run */
                    BasicGPUArray *inputErrorsOverTime;

                    /* the input errors for one simmulation run (for all batches) */
                    BasicGPUArray *allInputErrorsOverTime;

                    /* the filtered (back propagated) output errors */
                    BasicGPUArray *filteredOutputErrors;

                    /* summed network output for classification */
                    BasicGPUArray *summedActivation;

                    /* wether to use back propagation or not */
                    BasicGPUArray *useBackPropagation;

                public:

                    /* constructor */
                    LongShortTermMemoryKernelCaller(
                        unsigned batchSize,
                        unsigned numInputs,
                        unsigned numHidden,
                        unsigned numStandartHidden,
                        unsigned numAdaptiveHidden,
                        unsigned numOutputs,
                        unsigned numSimulationTimesteps,
                        unsigned errorMode,
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
                        FloatType *inputWeights,
                        FloatType *hiddenWeights,
                        FloatType *outputWeights,
                        FloatType *feedbackWeights,
                        FloatType *targetWeights,
                        std::vector<FloatType *> targetsOverTime,
                        std::vector<FloatType *> outputsOverTime,
                        std::vector<FloatType *> outputErrorsOverTime,
                        std::vector<FloatType *> errorMaskOverTime,
                        std::vector<FloatType *> outputErrorFactorOverTime,
                        FloatType *inputGradients,
                        FloatType *inputFiringRateGradients,
                        FloatType *hiddenGradients,
                        FloatType *hiddenFiringRateGradients,
                        FloatType *leakyReadoutGradients,
                        FloatType *inputErrorsOverTime, 
                        std::vector<FloatType *> allInputErrorsOverTime
                    );

                    /* destructor */
                    ~LongShortTermMemoryKernelCaller();

                    /* runs the kernel of this (blocks untill finished) */
                    void runAndWait(
                        int backPropagationMode = BACKPROPAGATION_OFF, 
                        bool inputErrors = false,
                        int starttime = -1,
                        int endtime   = -1
                    );

                    /* returns the networks squared summed error for the last run */
                    FloatType getSampleSquaredSummedError(unsigned batch);
                    FloatType getSquaredSummedError();

                    /* returns the networks summed target for the last run */
                    FloatType getSampleSummedTarget(unsigned batch);
                    FloatType getSummedTarget();

                    /* returns the networks squared summed target for the last run */
                    FloatType getSquaredSummedTarget();

                    /* returns the the number of summed values for the last run */
                    FloatType getSampleNumSummedValues(unsigned batch);
                    FloatType getNumSummedValues();

                    /* returns the networks classification accuracy */
                    FloatType getAccuracy();

                    /* reload feedback weights and other "not changing" values into device */
                    void reload();
                    
                    /* sets the current active device */
                    static void setDevice(int device);

            };
        }
    }
}
#endif /* __LONG_SHORT_TERM_MEMORY_KERNEL_CALLER__ */
