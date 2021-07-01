#ifndef __LONG_SHORT_TERM_MEMORY_ELIGIBILITY_NETWORK_H__
#define __LONG_SHORT_TERM_MEMORY_ELIGIBILITY_NETWORK_H__
#include "BasicNetwork.h"
#include "LongShortTermMemoryEligibilityNetworkOptions.h"
/**
 * simple one hidden layer fully connected feed forward classifier
 */ 
namespace SNN {

    /* forward declarations */
    namespace Interfaces { 
        class BasicSynapse; 
        class BasicGradient;
    }
    class BasicSerializer;
    class BasicNeuronSerializer;
    class BasicSynapseSerializer;
    class FeedbackWeightsSerializer;
    class SmartTrainSet;
    namespace Visualizations { class NetworkMonitor; }
    namespace Gradients { class BasicFeedbackWeights; }
    namespace Kernels {
        namespace GPU {
            class LongShortTermMemoryKernelCaller;
            class LongShortTermMemorySparseKernelCaller;
        }
    }
    
    namespace Networks {
        
        class LongShortTermMemoryEligibilityNetwork;
        class LongShortTermMemoryEligibilityNetwork: public Interfaces::BasicNetwork {
            
            protected:

                /* the smart trainset (if used) */
                std::shared_ptr<SmartTrainSet> smartTrainset;

                /* filtered target magnitude */
                std::vector<FloatType> targetMagnitudes;
                FloatType targetMagnitudeSum;

                /* serializers for optimized cpu computation */
                std::vector<std::shared_ptr<BasicSerializer>> serializer; 

                /* the input hidden and output neurons */
                std::vector<std::shared_ptr<Interfaces::BasicNeuron>> inputNeurons;
                std::vector<std::shared_ptr<Interfaces::BasicNeuron>> hiddenNeurons;
                std::vector<std::shared_ptr<Interfaces::BasicNeuron>> outputNeurons;

                /* the input hidden and output synapses */
                std::vector<std::shared_ptr<Interfaces::BasicSynapse>> inputSynapses;
                std::vector<std::shared_ptr<Interfaces::BasicSynapse>> hiddenSynapses; 
                std::vector<std::shared_ptr<Interfaces::BasicSynapse>> outputSynapses;

                /* the different optimizer types for easy access */
                std::vector<std::shared_ptr<Interfaces::BasicOptimizer>> inputOptimizers;
                std::vector<std::shared_ptr<Interfaces::BasicOptimizer>> hiddenOptimizers;
                std::vector<std::shared_ptr<Interfaces::BasicOptimizer>> outputOptimizers;
                std::vector<std::shared_ptr<Interfaces::BasicOptimizer>> inputRegularizers;
                std::vector<std::shared_ptr<Interfaces::BasicOptimizer>> hiddenRegularizers;


                /* the random feedback weights of this */
                std::shared_ptr<Gradients::BasicFeedbackWeights> feedbackMatrix;

                /* gpu optimization kernel for fixed broadcast error calculation */
                std::shared_ptr<Kernels::GPU::LongShortTermMemoryKernelCaller> gpuKernel;

                /* gpu optimization kernel for sparse fixed broadcast error calculation */
                std::shared_ptr<Kernels::GPU::LongShortTermMemorySparseKernelCaller> sparseGpuKernel;

                /* the Networks options of this */
                Options::LongShortTermMemoryEligibilityNetworkOptions &opts;

                /* array of network targets for <batch size> simulation runs (gpu opimization) */
                std::vector<FloatType *> targetsOverTime;

                /* array of network outputs for <batch size> simulation runs (gpu opimization) */
                std::vector<FloatType *> outputsOverTime;

                /* the network output errors for one simulation run */
                std::vector<FloatType *> outputErrorsOverTime;

                /* the network output delta errors for one simulation run */
                std::vector<FloatType *> deltaErrorInputssOverTime;
                std::vector<FloatType *> deltaErrorsOverTime;

                /* array of network error mask for <batch size> simulation runs (gpu opimization) */
                std::vector<FloatType *> errorMaskOverTime;
                std::vector<FloatType *> outputErrorFactorOverTime;

                /* array of network inputs for <batch size> simulation runs (gpu opimization) */
                std::vector<FloatType *> inputsOverTime;

                /* array of network input and hidden spikes for <batch size> simulation runs (gpu opimization) */
                std::vector<FloatType *> spikesOverTime;

                /* array of network hidden derivatives for <batch size> simulation runs */
                std::vector<FloatType *> derivativesOverTime;

                /* array of network inputs errors over time (summed over one batch) */
                FloatType *inputErrorsOverTime;

                /* array of network inputs errors over time (for one batch) */
                std::vector<FloatType *> allInputErrorsOverTime;

                /* array of hidden learnsignals for each batch and timestep 
                 * in order to calculate input errors */
                std::vector<FloatType *> inputErrorLearnSignals;

                /* the weights for each target in the overall error */
                FloatType *targetWeights;

                /*************************************************************************/
                /* for sparse connectivity the input and output indices for each synapse */
                /*************************************************************************/

                /* the synaptic input weight input indices */
                std::vector<unsigned> inputWeightsIn;

                /* the synaptic hidden weight input indices */
                std::vector<unsigned> hiddenWeightsIn;

                /* the synaptic output weight input indices */
                std::vector<unsigned> outputWeightsIn;

                /* the feedback weight input indices */
                std::vector<unsigned> feedbackWeightsIn;

                /* the synaptic input weight output indices */
                std::vector<unsigned> inputWeightsOut;

                /* the synaptic hidden weight output indices */
                std::vector<unsigned> hiddenWeightsOut;

                /* the synaptic output weight output indices */
                std::vector<unsigned> outputWeightsOut;

                /* the feedback weight output indices */
                std::vector<unsigned> feedbackWeightsOut;

                /* the weight serializers oft this */
                std::shared_ptr<BasicSynapseSerializer> inputSynapseSerializer, hiddenSynapseSerializer, outputSynapseSerializer;
                std::shared_ptr<FeedbackWeightsSerializer> feedbackMatrixSerializer;
                
                /* the neuron serializer of this */
                std::shared_ptr<BasicNeuronSerializer> neuronsSerializer; 

                /* array of hidden firing rates */
                FloatType *hiddenFiringRates;

                /* the current sample index */
                unsigned sampleIndex;

                /* the time step counter */
                unsigned timeStepCounter;

                /* the batch counter */
                unsigned batchCounter;

                /* the epoch counter */
                unsigned epochCounter;

                /* time keeping */
                uint64_t runtimeCounter;

                /* random number generator */
                rand128_t *rand;

                /* the initial random state */
                rand128_t initialRandomState;

                /* the squard error of this */
                FloatType summedError, summedErrorLast;

                /* the summed network target and squared summed network target */
                FloatType summedTarget, squaredSummedTarget;
                FloatType summedTargetLast, squaredSummedTargetLast;

                /* the number of values used within the error calculation */
                FloatType numErrorValues, numErrorValuesLast;

                /* the minimum error of this */
                FloatType minError;

                /* the classification accuracy */
                FloatType classificationAccuracy, classificationAccuracyLast;

                /* the number of classification samples */
                unsigned classificationSamples;

                /* the summed activation for each output neuron */
                std::vector<FloatType> summedActivation;

                /* should perform updates bevor updating the network */
                virtual void updateBevor();

                /* should perform the actual update stepp */
                virtual void doUpdate();

                /* should perform updates after updating the network */
                virtual void updateAfter();

                /* should perform updates bevor a back propagation step of the network */
                virtual void backPropagateBevor();

                /* should perform the actual backpropagation step */
                virtual void doBackPropagation();

                /* should perform updates after a back propagation step of the network */
                virtual void backPropagateAfter();

                /* creats a hidden neuron */
                std::shared_ptr<Interfaces::BasicNeuron> createNeuron(bool adaptive);

                /* creats a hidden synapse */
                std::shared_ptr<Interfaces::BasicSynapse> createSynapse(
                    std::shared_ptr<Interfaces::BasicNeuron> input,
                    std::shared_ptr<Interfaces::BasicNeuron> output,
                    FloatType weight,
                    bool adaptive
                );

                /* creats a gradient for the given synapse */
                std::shared_ptr<Interfaces::BasicGradient> createGradient(
                    std::shared_ptr<Interfaces::BasicSynapse> synapse,
                    unsigned neuronIndex, 
                    bool firingRateGradient = false,
                    bool hiddenSynapse = false
                );

                std::shared_ptr<Interfaces::BasicOptimizer> createOptimizer(
                    std::shared_ptr<Interfaces::BasicGradient> gradient,
                    bool firingRateOptimizer = false
                );

                /* create afully connected network */
                void fullyConnect(
                    std::vector<std::shared_ptr<Interfaces::BasicGradient>> &inputFiringRateGradients, 
                    std::vector<std::shared_ptr<Interfaces::BasicGradient>> &hiddenFiringRateGradients,
                    std::vector<std::shared_ptr<Interfaces::BasicGradient>> &inputFixedBroadcastGradients, 
                    std::vector<std::shared_ptr<Interfaces::BasicGradient>> &hiddenFixedBroadcastGradients,
                    std::vector<std::shared_ptr<Interfaces::BasicGradient>> &leakyReadoutGradients
                );

                /* create a sparse connected network */
                void sparseConnect(
                    std::vector<std::shared_ptr<Interfaces::BasicGradient>> &inputFiringRateGradients, 
                    std::vector<std::shared_ptr<Interfaces::BasicGradient>> &hiddenFiringRateGradients,
                    std::vector<std::shared_ptr<Interfaces::BasicGradient>> &inputFixedBroadcastGradients, 
                    std::vector<std::shared_ptr<Interfaces::BasicGradient>> &hiddenFixedBroadcastGradients,
                    std::vector<std::shared_ptr<Interfaces::BasicGradient>> &leakyReadoutGradients,
                    unsigned numAdaptiveInputSynapses,
                    unsigned numAdaptiveHiddenSynapses,
                    std::vector<FloatType> &inputWeights,
                    std::vector<FloatType> &hiddenWeights,
                    std::vector<FloatType> &outputWeights,
                    std::vector<FloatType> &feedbackWeights,
                    std::vector<unsigned> &inputWeightsIn,
                    std::vector<unsigned> &hiddenWeightsIn,
                    std::vector<unsigned> &outputWeightsIn,
                    std::vector<unsigned> &feedbackWeightsIn,
                    std::vector<unsigned> &inputWeightsOut,
                    std::vector<unsigned> &hiddenWeightsOut,
                    std::vector<unsigned> &outputWeightsOut,
                    std::vector<unsigned> &feedbackWeightsOut
                );

                /* initializes this */
                void init(
                    Options::LongShortTermMemoryEligibilityNetworkOptions &opts,
                    unsigned numAdaptiveInputSynapses,
                    unsigned numAdaptiveHiddenSynapses,
                    std::vector<FloatType> inputWeights,
                    std::vector<FloatType> hiddenWeights,
                    std::vector<FloatType> outputWeights,
                    std::vector<FloatType> feedbackWeights,
                    std::vector<unsigned> inputWeightsIn,
                    std::vector<unsigned> hiddenWeightsIn,
                    std::vector<unsigned> outputWeightsIn,
                    std::vector<unsigned> feedbackWeightsIn,
                    std::vector<unsigned> inputWeightsOut,
                    std::vector<unsigned> hiddenWeightsOut,
                    std::vector<unsigned> outputWeightsOut,
                    std::vector<unsigned> feedbackWeightsOut
                );

            protected: 

                /* should perform aditional resets */
                virtual void doReset();

            public:

                /* constructor */
                LongShortTermMemoryEligibilityNetwork(
                    Options::LongShortTermMemoryEligibilityNetworkOptions &opts,
                    unsigned numAdaptiveInputSynapses = -1,
                    unsigned numAdaptiveHiddenSynapses = -1,
                    std::vector<FloatType> inputWeights      = std::vector<FloatType>(),
                    std::vector<FloatType> hiddenWeights     = std::vector<FloatType>(), 
                    std::vector<FloatType> outputWeights     = std::vector<FloatType>(), 
                    std::vector<FloatType> feedbackWeights   = std::vector<FloatType>(), 
                    std::vector<unsigned> inputWeightsIn     = std::vector<unsigned>(),
                    std::vector<unsigned> hiddenWeightsIn    = std::vector<unsigned>(), 
                    std::vector<unsigned> outputWeightsIn    = std::vector<unsigned>(), 
                    std::vector<unsigned> feedbackWeightsIn  = std::vector<unsigned>(), 
                    std::vector<unsigned> inputWeightsOut    = std::vector<unsigned>(), 
                    std::vector<unsigned> hiddenWeightsOut   = std::vector<unsigned>(), 
                    std::vector<unsigned> outputWeightsOut   = std::vector<unsigned>(), 
                    std::vector<unsigned> feedbackWeightsOut = std::vector<unsigned>() 
                );

                /* destructor */
                ~LongShortTermMemoryEligibilityNetwork();

                /* should return the target signal of the network */
                virtual FloatType getTargetSignal(unsigned i) {
                    return targetAt(sampleIndex, timeStepCounter, i);
                }

                /* should return the target weight */
                virtual FloatType getTargetWeight(unsigned i) { return targetWeights[i]; }

                /* should return the network error mask */
                virtual FloatType getErrorMask() {
                    return errorMaskOverTime[sampleIndex][timeStepCounter];
                }

                /* should return the output of the network */
                virtual FloatType getOutput(unsigned i) {
                    assert(i < outputNeurons.size());
                    return outputNeurons[i]->getOutput();
                }

                /* should return the number of network outputs */
                virtual unsigned getNumOutputs() {
                    return opts.numOutputNeurons();
                }

                /* should return the current learn signal of the network (output - target) */
                virtual FloatType getLearnSignal() {
                    FloatType lernSignal = 0;
                    for (unsigned i = 0; i < opts.numOutputNeurons(); i++) {
                        lernSignal += 
                            outputNeurons[i]->getOutput() -
                            targetAt(sampleIndex, timeStepCounter, i);
                    }
                    return lernSignal;
                }

                /* sets the training status of this */
                void setTraining(bool training) {
                    this->opts.training(training);
                }

                /* should return the network error */
                virtual FloatType getError() {
                    if (opts.errorMode() == ERROR_MODE_CLASSIFICATION ||
                        opts.errorMode() == ERROR_MODE_INFINITY)
                        return summedErrorLast / numErrorValuesLast;

                    return getNormalizedMeanSquaredError();
                }

                FloatType getAccuracy() {
                    return classificationAccuracyLast;
                }

                /* returns the networks normalized mean squared error */
                virtual FloatType getNormalizedMeanSquaredError() {
                    FloatType mse     = summedErrorLast         / numErrorValuesLast;
                    FloatType sumT    = summedTargetLast        / numErrorValuesLast;
                    FloatType sqrSumT = squaredSummedTargetLast / numErrorValuesLast;

                    FloatType var = sqrSumT - pow(sumT, 2);
                    var *= numErrorValuesLast / (numErrorValuesLast - 1);
                    
                    return mse / var;
                }

                /* returns the networks root mean squared error */
                virtual FloatType getRootMeanSquaredError() {
                    return sqrt(summedErrorLast / numErrorValuesLast);
                }

                /* returns the networks root mean squared error derivation */
                virtual FloatType getRootMeanSquaredErrorDerivation();

                /* returns the networks squared error */
                virtual FloatType getSquaredError() {
                    return summedErrorLast;
                }

                /* access for the input training data of this */
                FloatType &inputAt(unsigned batchIndex, unsigned timeIndex, unsigned neuronIndex) {
                    assert(inputsOverTime.size() > batchIndex);
                    return inputsOverTime[batchIndex][timeIndex * opts.numInputNeurons() + neuronIndex];
                }

                /* access for the target training data of this */
                FloatType &targetAt(unsigned batchIndex, unsigned timeIndex, unsigned neuronIndex) {
                    assert(targetsOverTime.size() > batchIndex);
                    return targetsOverTime[batchIndex][timeIndex * getNumOutputs() + neuronIndex];
                }

                /* access for the target weights of this */
                FloatType &targetWeightAt(unsigned neuronIndex) {
                    return targetWeights[neuronIndex];
                }

                /* access for the error mask of this */
                FloatType &errorMaskAt(unsigned batchIndex, unsigned timeIndex) {
                    assert(errorMaskOverTime.size() > batchIndex);
                    return errorMaskOverTime[batchIndex][timeIndex];
                }

                /* access for the error mask of this */
                FloatType &outputErrorFactorAt(unsigned batchIndex, unsigned timeIndex, unsigned neuronIndex) {
                    assert(outputErrorFactorOverTime.size() > batchIndex);
                    return outputErrorFactorOverTime[batchIndex][opts.numOutputNeurons() * timeIndex + neuronIndex];
                }

                /* returns the feedback matrix of this */
                std::shared_ptr<Gradients::BasicFeedbackWeights> getFeedBackMatrix() {
                    return feedbackMatrix;
                }

                /* returns the average firing rate of the hidden neurons */
                FloatType getAverageFiringRate() {
                    FloatType averageFiringRate = 0;
                    for (unsigned i = opts.numInputNeurons(); i < neurons.size() - opts.numOutputNeurons(); i++)
                        averageFiringRate += neurons[i]->getFiringRate();

                    return averageFiringRate / opts.numHiddenNeurons();
                }

                /* returns the root mean squard error of the firing rate */
                FloatType getFiringRateError() {
                    assert(opts.numHiddenNeurons() > 1);
                    FloatType mse = 0;
                    for (unsigned i = opts.numInputNeurons(); i < neurons.size() - opts.numOutputNeurons(); i++)
                        mse += pow(opts.targetFiringRate() - neurons[i]->getFiringRate(), 2);

                    return sqrt(mse / (opts.numHiddenNeurons() - 1));
                }

/* debug mode flags */
#define DEBUG_FORWARD_PASS                  1
#define DEBUG_BACKWARD_PASS                 2
#define DEBUG_BPTT_INPUT_ERROR              4
#define DEBUG_CALCULATE_DERIVATIVES         8
#define DEBUG_EPROP_INPUT_ERROR             16
#define DEBUG_LEARNSIGNAL                   32
#define DEBUG_GRADIENTS                     64

                /* run one simulation with GPU optimization */
                void updateGPU(
                    bool batchSetted = false,
                    int backPropagationMode = -1,
                    int debugMode = DEBUG_FORWARD_PASS | DEBUG_BACKWARD_PASS | DEBUG_GRADIENTS,
                    int startTime = -1,
                    int endTime = -1
                );

                /* returns the gpu inputs for the given batch index */
                FloatType *getGPUInputs(unsigned batchIndex) { return inputsOverTime[batchIndex]; }

                /* returns the gpu outputs for the given batch index */
                FloatType *getGPUOutput(unsigned batchIndex) { return outputsOverTime[batchIndex]; }

                /* returns the gpu delta errors for the given batch index */
                FloatType *getGPUDeltaError(unsigned batchIndex) { return deltaErrorsOverTime[batchIndex]; }

                /* returns the gpu output errors for the given batch index */
                FloatType *getGPUOutputError(unsigned batchIndex) { return outputErrorsOverTime[batchIndex]; }

                /* returns the gpu target for the given batch index */
                FloatType *getGPUTarget(unsigned batchIndex) { return targetsOverTime[batchIndex]; }

                /* returns the gpu spikes for the given batch index */
                FloatType *getGPUSpikes(unsigned batchIndex) { return spikesOverTime[batchIndex]; }

                /* returns the gpu input errors for the given batch index */
                FloatType *getGPUInputErrors(unsigned batchIndex) { return allInputErrorsOverTime[batchIndex]; }

                /* returns the gpu input errors learnsignals for the given batch index */
                FloatType *getGPUInputErrorLearnSignals(unsigned batchIndex) { return inputErrorLearnSignals[batchIndex]; }

                /* returns the gpu spikes for the given batch index */
                FloatType *getGPUErrorMask(unsigned batchIndex) { return errorMaskOverTime[batchIndex]; }

                /* returns the squared summed gpu error for the given output */
                FloatType getGPUOutputSquaredSummedError(unsigned index);
                FloatType getGPUSquaredSummedError();

                /* returns the summed gpu target for the given output */
                FloatType getGPUOutputSummedTarget(unsigned index);
                FloatType getGPUSummedTarget();

                /* returns the squared summed gpu target for the given output */
                FloatType getGPUOutputSquaredSummedTarget(unsigned index);
                FloatType getGPUSquaredSummedTarget();

                /* returns the number of summed values for the given gpu output */
                FloatType getGPUOutputNumSummedValues(unsigned index);
                FloatType getGPUNumSummedValues();

                /* returns the number of simulation timesteps */
                unsigned numSimulationTimesteps() { return opts.numSimulationTimesteps(); }

                /* saves this to the given file */
                void save(std::string filename);

                /* saves this to the given file */
                void saveSpikes(int fd, int startTime = -1, int endTime = -1);

                /* loads this from the given file */
                void load(std::string filename, bool loadOptimizer = true, bool loadLearnRate = false);

                /* returns the synapses of this */
                std::vector<std::shared_ptr<Interfaces::BasicSynapse>> getSynapses() { return synapses; };

                /* returns the input synapses of this */
                std::vector<std::shared_ptr<Interfaces::BasicSynapse>> getInputtSynapses() { return inputSynapses; };

                /* returns the hidden synapses of this */
                std::vector<std::shared_ptr<Interfaces::BasicSynapse>> getHiddenSynapses() { return hiddenSynapses; };

                /* returns the output synapses of this */
                std::vector<std::shared_ptr<Interfaces::BasicSynapse>> getOutputSynapses() { return outputSynapses; };

                /* return the feedback weights of this */
                std::shared_ptr<Gradients::BasicFeedbackWeights> getFeedbackWeights() { return feedbackMatrix; }

                /* reload feedback weights and other "not changing" values into device */
                void reloadGPU();

                /* returns the initial random state of this */
                rand128_t getInitialRandomState() { return initialRandomState; }

                /* sets the initial random state of this */
                void setInitialRandomState(rand128_t initialRandomState) {
                    this->initialRandomState = initialRandomState;
                }

                /* returns the weights serializers */
                std::shared_ptr<BasicSynapseSerializer> getInputSynapsesSerializer() {
                    return inputSynapseSerializer;
                }
                std::shared_ptr<BasicSynapseSerializer> getHiddenSynapsesSerializer() {
                    return hiddenSynapseSerializer;
                }
                std::shared_ptr<BasicSynapseSerializer> getOutputSynapsesSerializer() {
                    return outputSynapseSerializer;
                }
                std::shared_ptr<FeedbackWeightsSerializer> getFeedbackSynapsesSerializer() {
                    return feedbackMatrixSerializer;
                }

                /* returns the neuron serializer */
                std::shared_ptr<BasicNeuronSerializer> getNeuronSerializer() {
                    return neuronsSerializer;
                }

                /* checks wether eprop 2 did calculate the input errors correctly */
                void inputErrorEpropCheck();
   
                /* runs the network in cpu mode */
                void runCPU(
                    std::vector<unsigned> batchIndices,
                    bool batchSetted, 
                    bool singleBatch, 
                    int debugMode
                );
   
                /* checks te gradients of this */
                void checkGradients();

                /* returns the optimizers of this */
                std::vector<std::shared_ptr<Interfaces::BasicOptimizer>> getInputOptimizers() { return inputOptimizers; }
                std::vector<std::shared_ptr<Interfaces::BasicOptimizer>> getHiddenOptimizers() { return hiddenOptimizers; }
                std::vector<std::shared_ptr<Interfaces::BasicOptimizer>> getOutputOptimizers() { return outputOptimizers; }
                std::vector<std::shared_ptr<Interfaces::BasicOptimizer>> getInputRegularizers() { return inputRegularizers; }
                std::vector<std::shared_ptr<Interfaces::BasicOptimizer>> getHiddenRegularizers() { return hiddenRegularizers; }

                /* saves the current batch inputs and outputs as csv */
                void saveBatch(std::string prefix, bool debug = false, bool inputErrors = false);

                /* returns the number of active synapses */
                unsigned getNumActiveSynases();

                /* sets the smart trainset of this */
                void setSmartTrainset(std::shared_ptr<SmartTrainSet> smartTrainset) {
                    this->smartTrainset = smartTrainset;
                }

                /* returns a pointer to the input errors */
                FloatType *getInputErrors() { return inputErrorsOverTime; }
   
        };
    }

}
#endif
