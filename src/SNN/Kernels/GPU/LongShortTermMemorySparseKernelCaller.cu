#include "LongShortTermMemorySparseKernelCaller.h"
#include "GPUArray.cu"
#include "LongShortTermMemorySparseKernel.cu"
#include "LongShortTermMemorySparseGradientCollectionKernel.cu"
#include "LongShortTermMemorySparseFiringRateKernel.cu"
#include "utils.h"

using namespace SNN::Kernels::GPU;

using std::vector;

/* constructor */
LongShortTermMemorySparseKernelCaller::LongShortTermMemorySparseKernelCaller(
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
    vector<FloatType *> inputSpikesOverTime,
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
    vector<FloatType *> targetsOverTime,
    vector<FloatType *> outputsOverTime,
    vector<FloatType *> errorMaskOverTime,
    vector<FloatType *> outputErrorFactorOverTime,
    FloatType *inputFixedBroadcastGradients,
    FloatType *inputFiringRateGradients,
    FloatType *hiddenFixedBroadcastGradients,
    FloatType *hiddenFiringRateGradients,
    FloatType *leakyReadoutGradients,
    FloatType *inputErrorsOverTime,
    vector<FloatType *> outputErrorsOverTime,
    vector<FloatType *> allInputErrorsOverTime,
    vector<FloatType *> deltaErrorsOverTime
) :
    executionFlags(new unsigned[1]),
    numBlocks(batchSize),
    numThreads(std::min(1024u, std::max(numInputWeights, std::max(numHiddenWeights, std::max(numOutputWeights, numFeedbackWeights))))),
    numHidden(numHidden),
    numOutputNeurons(numOutputs),
    outputError(new FloatType[numOutputs * (batchSize + 1)]),
    summedTargets(new FloatType[numOutputs * (batchSize + 1)]),
    squaredSummedTargets(new FloatType[numOutputs * (batchSize + 1)]),
    numSummedValues(new FloatType[numOutputs * (batchSize + 1)]),
    classificationAccuracyCPU(new FloatType[numOutputs * (batchSize + 1)]),
    classificationSamplesCPU(new FloatType[numOutputs * (batchSize + 1)]),
    flags(                        new GPUArray<unsigned>(  executionFlags, 1)),
    numInputs(                    new GPUArray<unsigned>( numInputs)),
    numStandartHidden(            new GPUArray<unsigned>( numStandartHidden)),
    numAdaptiveHidden(            new GPUArray<unsigned>( numAdaptiveHidden)),
    numOutputs(                   new GPUArray<unsigned>( numOutputs)),
    batchSize(                    new GPUArray<unsigned>( batchSize)),
    numSimulationTimesteps(       new GPUArray<unsigned>( numSimulationTimesteps)),
    timeStepLength(               new GPUArray<FloatType>(timeStepLength)),
    spikeThreshold(               new GPUArray<FloatType>(spikeThreshold)),
    refactoryPeriod(              new GPUArray<FloatType>(refactoryPeriod)),
    hiddenDecayFactor(            new GPUArray<FloatType>(hiddenDecayFactor)),
    readoutDecayFactor(           new GPUArray<FloatType>(readoutDecayFactor)),
    adaptationDecayFactor(        new GPUArray<FloatType>(adaptationDecayFactor)),
    thresholdIncreaseConstant(    new GPUArray<FloatType>(thresholdIncreaseConstant)),
    targetFiringRate(             new GPUArray<FloatType>(targetFiringRate)),
    firingRateScallingFactor(     new GPUArray<FloatType>(firingRateScallingFactor)),
    derivativeDumpingFactor(      new GPUArray<FloatType>(derivativeDumpingFactor)),
    inputSpikesOverTime(          new GPUArray<FloatType>(inputSpikesOverTime,                                           numInputs * numSimulationTimesteps)),
    spikesOverTime(               new GPUArray<FloatType>(spikesOverTime,                                                (numInputs + numHidden) * numSimulationTimesteps)),
    firingRates(                  new GPUArray<FloatType>(firingRates,                                                   numHidden)),
    numSpikes(                    new GPUArray<FloatType>(vector<FloatType *>(batchSize, NULL),                          numInputs + numHidden)),
    numStandartInputSynapses(     new GPUArray<unsigned>( numStandartInputSynapses)),    
    numStandartHiddenSynapses(    new GPUArray<unsigned>( numStandartHiddenSynapses)),    
    numAdaptiveInputSynapses(     new GPUArray<unsigned>( numAdaptiveInputSynapses)),    
    numAdaptiveHiddenSynapses(    new GPUArray<unsigned>( numAdaptiveHiddenSynapses)),    
    numInputWeights(              new GPUArray<unsigned>( numInputWeights)),    
    numHiddenWeights(             new GPUArray<unsigned>( numHiddenWeights)),    
    numOutputWeights(             new GPUArray<unsigned>( numOutputWeights)),    
    numFeedbackWeights(           new GPUArray<unsigned>( numFeedbackWeights)),    
    inputWeights(                 new GPUArray<FloatType>(inputWeights,                                                  numInputWeights)),
    hiddenWeights(                new GPUArray<FloatType>(hiddenWeights,                                                 numHiddenWeights)),
    outputWeights(                new GPUArray<FloatType>(outputWeights,                                                 numOutputWeights)),
    feedbackWeights(              new GPUArray<FloatType>(feedbackWeights,                                               numFeedbackWeights)),
    inputWeightsIn(               new GPUArray<unsigned>( inputWeightsIn,                                                numInputWeights)),
    hiddenWeightsIn(              new GPUArray<unsigned>( hiddenWeightsIn,                                               numHiddenWeights)),
    outputWeightsIn(              new GPUArray<unsigned>( outputWeightsIn,                                               numOutputWeights)),
    feedbackWeightsIn(            new GPUArray<unsigned>( feedbackWeightsIn,                                             numFeedbackWeights)),
    inputWeightsOut(              new GPUArray<unsigned>( inputWeightsOut,                                               numInputWeights)),
    hiddenWeightsOut(             new GPUArray<unsigned>( hiddenWeightsOut,                                              numHiddenWeights)),
    outputWeightsOut(             new GPUArray<unsigned>( outputWeightsOut,                                              numOutputWeights)),
    feedbackWeightsOut(           new GPUArray<unsigned>( feedbackWeightsOut,                                            numFeedbackWeights)),
    targetWeights(                new GPUArray<FloatType>(targetWeights,                                                 numOutputs)),
    targetsOverTime(              new GPUArray<FloatType>(targetsOverTime,                                               numOutputs * numSimulationTimesteps)),
    outputsOverTime(              new GPUArray<FloatType>(outputsOverTime,                                               numOutputs * numSimulationTimesteps)),
    derivativesOverTime(          new GPUArray<FloatType>(vector<FloatType *>(batchSize, NULL),                          numHidden * numSimulationTimesteps)),
    deltaErrorsOverTime(          new GPUArray<FloatType>(deltaErrorsOverTime,                                           (numHidden + numAdaptiveHidden + numOutputs) * numSimulationTimesteps)),
    errorMaskOverTime(            new GPUArray<FloatType>(errorMaskOverTime,                                             numSimulationTimesteps)),
    outputErrorFactorOverTime(    new GPUArray<FloatType>(outputErrorFactorOverTime,                                     numOutputs * numSimulationTimesteps)),
    inputFixedBroadcastGradients( new GPUArray<FloatType>(vector<FloatType *>(batchSize, inputFixedBroadcastGradients),  numInputWeights)),
    inputFiringRateGradients(     new GPUArray<FloatType>(vector<FloatType *>(batchSize, inputFiringRateGradients),      numInputWeights)),
    hiddenFixedBroadcastGradients(new GPUArray<FloatType>(vector<FloatType *>(batchSize, hiddenFixedBroadcastGradients), numHiddenWeights)),
    hiddenFiringRateGradients(    new GPUArray<FloatType>(vector<FloatType *>(batchSize, hiddenFiringRateGradients),     numHiddenWeights)),
    leakyReadoutGradients(        new GPUArray<FloatType>(vector<FloatType *>(batchSize, leakyReadoutGradients),         numOutputWeights)),
    networkError(                 new GPUArray<FloatType>(outputError,                                                   numOutputs * (batchSize + 1))),
    networkTargets(               new GPUArray<FloatType>(summedTargets,                                                 numOutputs * (batchSize + 1))),
    networkSquaredTargets(        new GPUArray<FloatType>(squaredSummedTargets,                                          numOutputs * (batchSize + 1))),
    summedValues(                 new GPUArray<FloatType>(numSummedValues,                                               numOutputs * (batchSize + 1))),
    classificationAccuracy(       new GPUArray<FloatType>(classificationAccuracyCPU,                                     numOutputs * (batchSize + 1))),
    classificationSamples(        new GPUArray<FloatType>(classificationSamplesCPU,                                      numOutputs * (batchSize + 1))),
    summedActivation(             new GPUArray<FloatType>(vector<FloatType *>(batchSize, NULL),                          numOutputs)),
    filteredEligibilityTraces(    new GPUArray<FloatType>(vector<FloatType *>(batchSize, NULL),                          numInputWeights + numHiddenWeights)),
    filteredSpikes(               new GPUArray<FloatType>(vector<FloatType *>(batchSize, NULL),                          numInputs + numHidden)),
    thresholdAdaptation(          new GPUArray<FloatType>(vector<FloatType *>(batchSize, NULL),                          numHidden)),
    adaptionEligibility(          new GPUArray<FloatType>(vector<FloatType *>(batchSize, NULL),                          numAdaptiveInputSynapses + numAdaptiveHiddenSynapses)),
    derivatives(                  new GPUArray<FloatType>(vector<FloatType *>(batchSize, NULL),                          numHidden)),
    I(                            new GPUArray<FloatType>(vector<FloatType *>(batchSize, NULL),                          numHidden + numOutputs)),
    v(                            new GPUArray<FloatType>(vector<FloatType *>(batchSize, NULL),                          numHidden + numOutputs)),
    hiddenSpikes(                 new GPUArray<FloatType>(vector<FloatType *>(batchSize, NULL),                          numHidden)),
    timeStepsSinceLastSpike(      new GPUArray<FloatType>(vector<FloatType *>(batchSize, NULL),                          numHidden)),
    learnSignalsOverTime(         new GPUArray<FloatType>(vector<FloatType *>(batchSize, NULL),                          numHidden * numSimulationTimesteps)),
    deltaErrorsVoltage(           new GPUArray<FloatType>(vector<FloatType *>(batchSize, NULL),                          numHidden)),
    deltaErrorsAdaption(          new GPUArray<FloatType>(vector<FloatType *>(batchSize, NULL),                          numAdaptiveHidden)),
    inputErrorsOverTime(          new GPUArray<FloatType>(vector<FloatType *>(batchSize, inputErrorsOverTime),           numInputs * numSimulationTimesteps)),
    outputErrorsOverTime(         new GPUArray<FloatType>(outputErrorsOverTime,                                          numOutputs * numSimulationTimesteps)),
    allInputErrorsOverTime(       new GPUArray<FloatType>(allInputErrorsOverTime,                                        numInputs * numSimulationTimesteps)),
    filteredOutputErrors(         new GPUArray<FloatType>(vector<FloatType *>(batchSize, NULL),                          numOutputs)) {

    this->numInputs->copyToDevice();
    this->numStandartHidden->copyToDevice();
    this->numAdaptiveHidden->copyToDevice();
    this->numOutputs->copyToDevice();
    this->batchSize->copyToDevice();
    this->numSimulationTimesteps->copyToDevice();
    this->timeStepLength->copyToDevice();
    this->spikeThreshold->copyToDevice();
    this->refactoryPeriod->copyToDevice();
    this->adaptationDecayFactor->copyToDevice();
    this->thresholdIncreaseConstant->copyToDevice();
    this->hiddenDecayFactor->copyToDevice();
    this->readoutDecayFactor->copyToDevice();
    this->targetFiringRate->copyToDevice();
    this->firingRateScallingFactor->copyToDevice();
    this->derivativeDumpingFactor->copyToDevice();
    this->feedbackWeights->copyToDevice();
    this->firingRates->copyToDevice();
    this->numStandartInputSynapses->copyToDevice();
    this->numStandartHiddenSynapses->copyToDevice();
    this->numAdaptiveInputSynapses->copyToDevice();
    this->numAdaptiveHiddenSynapses->copyToDevice();
    this->numInputWeights->copyToDevice();
    this->numHiddenWeights->copyToDevice();
    this->numOutputWeights->copyToDevice();
    this->numFeedbackWeights->copyToDevice();
    this->inputWeightsIn->copyToDevice();
    this->hiddenWeightsIn->copyToDevice();
    this->outputWeightsIn->copyToDevice();
    this->feedbackWeightsIn->copyToDevice();
    this->inputWeightsOut->copyToDevice();
    this->hiddenWeightsOut->copyToDevice();
    this->outputWeightsOut->copyToDevice();
    this->feedbackWeightsOut->copyToDevice();

    unsigned memory = 0;
    memory += this->flags->globalMemoryConsumption();
    memory += this->numInputs->globalMemoryConsumption();
    memory += this->numStandartHidden->globalMemoryConsumption();
    memory += this->numAdaptiveHidden->globalMemoryConsumption();
    memory += this->numOutputs->globalMemoryConsumption();
    memory += this->batchSize->globalMemoryConsumption();
    memory += this->numSimulationTimesteps->globalMemoryConsumption();
    memory += this->timeStepLength->globalMemoryConsumption();
    memory += this->spikeThreshold->globalMemoryConsumption();
    memory += this->refactoryPeriod->globalMemoryConsumption();
    memory += this->adaptationDecayFactor->globalMemoryConsumption();
    memory += this->thresholdIncreaseConstant->globalMemoryConsumption();
    memory += this->hiddenDecayFactor->globalMemoryConsumption();
    memory += this->readoutDecayFactor->globalMemoryConsumption();
    memory += this->targetFiringRate->globalMemoryConsumption();
    memory += this->firingRateScallingFactor->globalMemoryConsumption();
    memory += this->derivativeDumpingFactor->globalMemoryConsumption();
    memory += this->inputSpikesOverTime->globalMemoryConsumption();
    memory += this->spikesOverTime->globalMemoryConsumption();
    memory += this->firingRates->globalMemoryConsumption();
    memory += this->numStandartInputSynapses->globalMemoryConsumption();
    memory += this->numStandartHiddenSynapses->globalMemoryConsumption();
    memory += this->numAdaptiveInputSynapses->globalMemoryConsumption();
    memory += this->numAdaptiveHiddenSynapses->globalMemoryConsumption();
    memory += this->numInputWeights->globalMemoryConsumption();
    memory += this->numHiddenWeights->globalMemoryConsumption();
    memory += this->numOutputWeights->globalMemoryConsumption();
    memory += this->numFeedbackWeights->globalMemoryConsumption();
    memory += this->numSpikes->globalMemoryConsumption();
    memory += this->inputWeights->globalMemoryConsumption();
    memory += this->hiddenWeights->globalMemoryConsumption();
    memory += this->outputWeights->globalMemoryConsumption();
    memory += this->feedbackWeights->globalMemoryConsumption();
    memory += this->inputWeightsIn->globalMemoryConsumption();
    memory += this->hiddenWeightsIn->globalMemoryConsumption();
    memory += this->outputWeightsIn->globalMemoryConsumption();
    memory += this->feedbackWeightsIn->globalMemoryConsumption();
    memory += this->inputWeightsOut->globalMemoryConsumption();
    memory += this->hiddenWeightsOut->globalMemoryConsumption();
    memory += this->outputWeightsOut->globalMemoryConsumption();
    memory += this->feedbackWeightsOut->globalMemoryConsumption();
    memory += this->targetWeights->globalMemoryConsumption();
    memory += this->targetsOverTime->globalMemoryConsumption();
    memory += this->outputsOverTime->globalMemoryConsumption();
    memory += this->derivativesOverTime->globalMemoryConsumption();
    memory += this->deltaErrorsOverTime->globalMemoryConsumption();
    memory += this->errorMaskOverTime->globalMemoryConsumption();
    memory += this->outputErrorFactorOverTime->globalMemoryConsumption();
    memory += this->inputFixedBroadcastGradients->globalMemoryConsumption();
    memory += this->inputFiringRateGradients->globalMemoryConsumption();
    memory += this->hiddenFixedBroadcastGradients->globalMemoryConsumption();
    memory += this->hiddenFiringRateGradients->globalMemoryConsumption();
    memory += this->leakyReadoutGradients->globalMemoryConsumption();
    memory += this->networkError->globalMemoryConsumption();
    memory += this->networkTargets->globalMemoryConsumption();
    memory += this->networkSquaredTargets->globalMemoryConsumption();
    memory += this->summedValues->globalMemoryConsumption();
    memory += this->classificationAccuracy->globalMemoryConsumption();
    memory += this->classificationSamples->globalMemoryConsumption();
    memory += this->summedActivation->globalMemoryConsumption();
    memory += this->filteredEligibilityTraces->globalMemoryConsumption();
    memory += this->filteredSpikes->globalMemoryConsumption();
    memory += this->thresholdAdaptation->globalMemoryConsumption();
    memory += this->adaptionEligibility->globalMemoryConsumption();
    memory += this->derivatives->globalMemoryConsumption();
    memory += this->I->globalMemoryConsumption();
    memory += this->v->globalMemoryConsumption();
    memory += this->hiddenSpikes->globalMemoryConsumption();
    memory += this->timeStepsSinceLastSpike->globalMemoryConsumption();
    memory += this->learnSignalsOverTime->globalMemoryConsumption();
    memory += this->deltaErrorsVoltage->globalMemoryConsumption();
    memory += this->deltaErrorsAdaption->globalMemoryConsumption();
    memory += this->inputErrorsOverTime->globalMemoryConsumption();
    memory += this->outputErrorsOverTime->globalMemoryConsumption();
    memory += this->allInputErrorsOverTime->globalMemoryConsumption();
    memory += this->filteredOutputErrors->globalMemoryConsumption();
    log_str("globalMemoryConsumption: " + itoa(memory), LOG_DD);
}


/* destructor */
LongShortTermMemorySparseKernelCaller::~LongShortTermMemorySparseKernelCaller() {

    delete[] outputError;
    delete[] executionFlags;
    delete flags;
    delete numInputs;
    delete numStandartHidden;
    delete numAdaptiveHidden;
    delete numOutputs;
    delete batchSize;
    delete numSimulationTimesteps;
    delete timeStepLength;
    delete spikeThreshold;
    delete refactoryPeriod;
    delete hiddenDecayFactor;
    delete readoutDecayFactor;
    delete adaptationDecayFactor;
    delete thresholdIncreaseConstant;
    delete targetFiringRate;
    delete firingRateScallingFactor;
    delete derivativeDumpingFactor;
    delete inputSpikesOverTime;
    delete spikesOverTime;
    delete firingRates;
    delete numSpikes;
    delete numStandartInputSynapses;
    delete numStandartHiddenSynapses;
    delete numAdaptiveInputSynapses;
    delete numAdaptiveHiddenSynapses;
    delete numInputWeights;
    delete numHiddenWeights;
    delete numOutputWeights;
    delete numFeedbackWeights;
    delete inputWeights;
    delete hiddenWeights;
    delete outputWeights;
    delete feedbackWeights;
    delete inputWeightsIn;
    delete hiddenWeightsIn;
    delete outputWeightsIn;
    delete feedbackWeightsIn;
    delete inputWeightsOut;
    delete hiddenWeightsOut;
    delete outputWeightsOut;
    delete feedbackWeightsOut;
    delete targetWeights;
    delete targetsOverTime;
    delete outputsOverTime;
    delete derivativesOverTime;
    delete deltaErrorsOverTime;
    delete errorMaskOverTime;
    delete outputErrorFactorOverTime;
    delete inputFixedBroadcastGradients;
    delete inputFiringRateGradients;
    delete hiddenFixedBroadcastGradients;
    delete hiddenFiringRateGradients;
    delete leakyReadoutGradients;
    delete networkError;
    delete summedValues;
    delete classificationAccuracy;
    delete classificationSamples;
    delete summedActivation;
    delete networkTargets;
    delete networkSquaredTargets;
    delete filteredEligibilityTraces;
    delete filteredSpikes;
    delete thresholdAdaptation;
    delete adaptionEligibility;
    delete derivatives;
    delete I;
    delete v;
    delete hiddenSpikes;
    delete timeStepsSinceLastSpike;
    delete learnSignalsOverTime;
    delete deltaErrorsVoltage;
    delete deltaErrorsAdaption;
    delete inputErrorsOverTime;
    delete outputErrorsOverTime;
    delete allInputErrorsOverTime;
    delete filteredOutputErrors;
}


/* runs the kernel of this (blocks untill finished) */
void LongShortTermMemorySparseKernelCaller::runAndWait() {

    flags->copyToDevice();
    inputSpikesOverTime->copyToDevice();
    outputErrorsOverTime->copyToDevice(); 
    targetWeights->copyToDevice();
    targetsOverTime->copyToDevice();
    errorMaskOverTime->copyToDevice();
    outputErrorFactorOverTime->copyToDevice();
    inputWeights->copyToDevice();
    hiddenWeights->copyToDevice();
    outputWeights->copyToDevice();

    log_str("launching longShortTermMemorySparseKernel<<<" + 
            itoa(numBlocks) + ", " + itoa(numThreads) + ">>>", LOG_DD);

    longShortTermMemorySparseKernel<<<numBlocks, numThreads>>>(
        (unsigned     *) flags->d_ptr(),
        (unsigned     *) numInputs->d_ptr(),
        (unsigned     *) numStandartHidden->d_ptr(),
        (unsigned     *) numAdaptiveHidden->d_ptr(),
        (unsigned     *) numOutputs->d_ptr(),
        (unsigned     *) numSimulationTimesteps->d_ptr(),
        (FloatType    *) timeStepLength->d_ptr(),
        (FloatType    *) spikeThreshold->d_ptr(),
        (FloatType    *) refactoryPeriod->d_ptr(),
        (FloatType    *) hiddenDecayFactor->d_ptr(),
        (FloatType    *) readoutDecayFactor->d_ptr(),
        (FloatType    *) adaptationDecayFactor->d_ptr(),
        (FloatType    *) thresholdIncreaseConstant->d_ptr(),
        (FloatType    *) targetFiringRate->d_ptr(),
        (FloatType    *) firingRateScallingFactor->d_ptr(),
        (FloatType    *) derivativeDumpingFactor->d_ptr(),
        (FloatType    *) inputSpikesOverTime->d_ptr(),
        (FloatType    *) spikesOverTime->d_ptr(),
        (FloatType    *) firingRates->d_ptr(),
        (FloatType    *) numSpikes->d_ptr(),
        (unsigned     *) numStandartInputSynapses->d_ptr(),
        (unsigned     *) numStandartHiddenSynapses->d_ptr(),
        (unsigned     *) numAdaptiveInputSynapses->d_ptr(),
        (unsigned     *) numAdaptiveHiddenSynapses->d_ptr(),
        (unsigned     *) numInputWeights->d_ptr(),
        (unsigned     *) numHiddenWeights->d_ptr(),
        (unsigned     *) numOutputWeights->d_ptr(),
        (unsigned     *) numFeedbackWeights->d_ptr(),
        (FloatType    *) inputWeights->d_ptr(),
        (FloatType    *) hiddenWeights->d_ptr(),
        (FloatType    *) outputWeights->d_ptr(),
        (FloatType    *) feedbackWeights->d_ptr(),
        (unsigned     *) inputWeightsIn->d_ptr(),
        (unsigned     *) hiddenWeightsIn->d_ptr(),
        (unsigned     *) outputWeightsIn->d_ptr(),
        (unsigned     *) feedbackWeightsIn->d_ptr(),
        (unsigned     *) inputWeightsOut->d_ptr(),
        (unsigned     *) hiddenWeightsOut->d_ptr(),
        (unsigned     *) outputWeightsOut->d_ptr(),
        (unsigned     *) feedbackWeightsOut->d_ptr(),
        (FloatType    *) targetWeights->d_ptr(),
        (FloatType    *) targetsOverTime->d_ptr(),
        (FloatType    *) outputsOverTime->d_ptr(),
        (FloatType    *) derivativesOverTime->d_ptr(),
        (FloatType    *) deltaErrorsOverTime->d_ptr(),
        (FloatType    *) errorMaskOverTime->d_ptr(),
        (FloatType    *) outputErrorFactorOverTime->d_ptr(),
        (FloatType    *) inputFixedBroadcastGradients->d_ptr(),
        (FloatType    *) inputFiringRateGradients->d_ptr(),
        (FloatType    *) hiddenFixedBroadcastGradients->d_ptr(),
        (FloatType    *) hiddenFiringRateGradients->d_ptr(),
        (FloatType    *) leakyReadoutGradients->d_ptr(),
        (FloatType    *) networkError->d_ptr(),
        (FloatType    *) networkTargets->d_ptr(),
        (FloatType    *) networkSquaredTargets->d_ptr(),
        (FloatType    *) summedValues->d_ptr(),
        (FloatType    *) classificationAccuracy->d_ptr(),
        (FloatType    *) classificationSamples->d_ptr(),
        (FloatType    *) summedActivation->d_ptr(),
        (FloatType    *) filteredEligibilityTraces->d_ptr(),
        (FloatType    *) filteredSpikes->d_ptr(),
        (FloatType    *) thresholdAdaptation->d_ptr(),
        (FloatType    *) adaptionEligibility->d_ptr(),
        (FloatType    *) derivatives->d_ptr(),
        (FloatType    *) I->d_ptr(),
        (FloatType    *) v->d_ptr(),
        (FloatType    *) hiddenSpikes->d_ptr(),
        (FloatType    *) timeStepsSinceLastSpike->d_ptr(),
        (FloatType    *) learnSignalsOverTime->d_ptr(),
        (FloatType    *) filteredOutputErrors->d_ptr(),
        (FloatType    *) deltaErrorsVoltage->d_ptr(),
        (FloatType    *) deltaErrorsAdaption->d_ptr(),
        (FloatType    *) inputErrorsOverTime->d_ptr(),
        (FloatType    *) outputErrorsOverTime->d_ptr()
    );

    if (!CONTAINS(*executionFlags, LSNN_NO_GRADIENT_COLLECTION)) {
        // TODO parametrisize blocks and threads
        longShortTermMemorySparseGradientCollectionKernel<<<16, 1024>>>(
            (unsigned  *) flags->d_ptr(),
            (unsigned  *) numInputWeights->d_ptr(),
            (unsigned  *) numHiddenWeights->d_ptr(),
            (unsigned  *) numOutputWeights->d_ptr(),
            (unsigned  *) numInputs->d_ptr(),
            (unsigned  *) numOutputs->d_ptr(),
            (unsigned  *) numSimulationTimesteps->d_ptr(),
            (unsigned  *) batchSize->d_ptr(),
            (FloatType *) inputFixedBroadcastGradients->d_ptr(),
            (FloatType *) inputFiringRateGradients->d_ptr(),
            (FloatType *) hiddenFixedBroadcastGradients->d_ptr(),
            (FloatType *) hiddenFiringRateGradients->d_ptr(),
            (FloatType *) leakyReadoutGradients->d_ptr(),
            (FloatType *) inputErrorsOverTime->d_ptr(),
            (FloatType *) allInputErrorsOverTime->d_ptr(),
            (FloatType *) networkError->d_ptr(),
            (FloatType *) networkTargets->d_ptr(),
            (FloatType *) networkSquaredTargets->d_ptr(),
            (FloatType *) summedValues->d_ptr(),
            (FloatType *) classificationAccuracy->d_ptr(),
            (FloatType *) classificationSamples->d_ptr()
        );

        longShortTermMemorySparseFiringRateKernel<<<16, 1024>>>(
            (unsigned  *) numInputs->d_ptr(),
            (unsigned  *) numStandartHidden->d_ptr(),
            (unsigned  *) numAdaptiveHidden->d_ptr(),
            (unsigned  *) batchSize->d_ptr(),
            (unsigned  *) numSimulationTimesteps->d_ptr(),
            (FloatType *) timeStepLength->d_ptr(),
            (FloatType *) firingRates->d_ptr(),
            (FloatType *) numSpikes->d_ptr()
        );

        if (CONTAINS(*executionFlags, LSNN_ELIGIBILTY_GRADIENTS|LSNN_BACKPROPAGATED_GRADIENTS|LSNN_BACKPROPAGATED_ELIGIBILITY_GRADIENTS)) {
            inputFixedBroadcastGradients->copyToHost(0);
            inputFiringRateGradients->copyToHost(0);
            hiddenFixedBroadcastGradients->copyToHost(0);
            hiddenFiringRateGradients->copyToHost(0);
        }
        if (CONTAINS(*executionFlags, LSNN_BACKPROPAGATED_GRADIENTS|LSNN_READOUT_FORWARD_GRAIENTS)) {
            leakyReadoutGradients->copyToHost(0);
        }
    }
    firingRates->copyToHost();
    networkError->copyToHost();
    networkTargets->copyToHost();
    networkSquaredTargets->copyToHost();
    summedValues->copyToHost();
    outputsOverTime->copyToHost();
    spikesOverTime->copyToHost();
    classificationAccuracy->copyToHost();
    classificationSamples->copyToHost();
    if (CONTAINS(*executionFlags, LSNN_DEBUG_DELTA_ERRORS))
        deltaErrorsOverTime->copyToHost();

    if (CONTAINS(*executionFlags, LSNN_INPUT_ERRORS)) {
        inputErrorsOverTime->copyToHost(0);
        allInputErrorsOverTime->copyToHost();
    }

    if (cudaGetLastError())
        log_err("LongShortTermMemoryKernel failed", LOG_EE);
}

/* returns the networks squared summed error for the last run and the given output */
FloatType LongShortTermMemorySparseKernelCaller::getSampleSquaredSummedError(unsigned s) {
    FloatType sum = 0;
    for (unsigned i = 0; i < numOutputNeurons; i++) 
        sum += outputError[s * numOutputNeurons + i];

    return sum;
}
FloatType LongShortTermMemorySparseKernelCaller::getOutputSquaredSummedError(unsigned i) {
    return outputError[numBlocks * numOutputNeurons + i];
}
FloatType LongShortTermMemorySparseKernelCaller::getSquaredSummedError() {
    FloatType sum = 0;
    for (unsigned i = 0; i < numOutputNeurons; i++) 
        sum += outputError[numBlocks * numOutputNeurons + i];

    return sum;
}

/* returns the networks summed target for the last run and the given output */
FloatType LongShortTermMemorySparseKernelCaller::getOutputSummedTarget(unsigned i) {
    return summedTargets[numBlocks * numOutputNeurons + i];
}
FloatType LongShortTermMemorySparseKernelCaller::getSummedTarget() {
    FloatType sum = 0;
    for (unsigned i = 0; i < numOutputNeurons; i++) 
        sum += summedTargets[numBlocks * numOutputNeurons + i];

    return sum;
}

/* returns the networks squared summed target for the last run and the given output */
FloatType LongShortTermMemorySparseKernelCaller::getOutputSquaredSummedTarget(unsigned i) {
    return squaredSummedTargets[numBlocks * numOutputNeurons + i];
}
FloatType LongShortTermMemorySparseKernelCaller::getSquaredSummedTarget() {
    FloatType sum = 0;
    for (unsigned i = 0; i < numOutputNeurons; i++) 
        sum += squaredSummedTargets[numBlocks * numOutputNeurons + i];

    return sum;
}

/* returns the the number of summed values for the last run and the given output */
FloatType LongShortTermMemorySparseKernelCaller::getOutputNumSummedValues(unsigned i) {
    return numSummedValues[numBlocks * numOutputNeurons + i];
}
FloatType LongShortTermMemorySparseKernelCaller::getNumSummedValues() {
    FloatType sum = 0;
    for (unsigned i = 0; i < numOutputNeurons; i++) 
        sum += numSummedValues[numBlocks * numOutputNeurons + i];

    return sum;
}

/* reload feedback weights and other "not changing" values into device */
void LongShortTermMemorySparseKernelCaller::reload() {
    firingRates->copyToDevice();
    feedbackWeights->copyToDevice();
}

/* sets all flags for normal eligibility traiing */
void LongShortTermMemorySparseKernelCaller::setEligibilityTraining() {
    *executionFlags = 
        LSNN_FORWARD_PASS               |
        LSNN_INITIALIZE_FORWARD_PASS    |
        LSNN_INITIALIZE_GRADIENTS       |
        LSNN_FIXED_BROADCAST_LEARNSIGNAL|
        LSNN_ELIGIBILTY_GRADIENTS       |
        LSNN_READOUT_FORWARD_GRAIENTS;
}

/* sets all flags for normal backpropagation traiing */
void LongShortTermMemorySparseKernelCaller::setBackPropagationTraining() {
    *executionFlags = 
        LSNN_FORWARD_PASS               |
        LSNN_BACKWARD_PASS              |
        LSNN_INITIALIZE_FORWARD_PASS    |
        LSNN_INITIALIZE_BACKWARD_PASS   |
        LSNN_INITIALIZE_GRADIENTS       |
        LSNN_REGRESSION_ERROR           |
        LSNN_BACKPROPAGATED_GRADIENTS   |
        LSNN_FAST_FRINIG_RATE;
}

/* sets all flags for elibibility backpropagation traiing */
void LongShortTermMemorySparseKernelCaller::setEligibilityBackPropagationTraining() {
    *executionFlags = 
        LSNN_FORWARD_PASS                         |
        LSNN_BACKWARD_PASS                        |
        LSNN_INITIALIZE_FORWARD_PASS              |
        LSNN_INITIALIZE_BACKWARD_PASS             |
        LSNN_INITIALIZE_GRADIENTS                 |
        LSNN_FIXED_BROADCAST_LEARNSIGNAL          |
        LSNN_SYMETRIC_EPROP                       |
        LSNN_ELIGIBILTY_GRADIENTS                 |
        LSNN_REGRESSION_ERROR                     |
        LSNN_BACKPROPAGATED_ELIGIBILITY_GRADIENTS |
        LSNN_READOUT_FORWARD_GRAIENTS;
}


/* sets all flags for t forward pass evaluation */
void LongShortTermMemorySparseKernelCaller::setForwardPass() {
    *executionFlags = 
        LSNN_FORWARD_PASS               |
        LSNN_INITIALIZE_FORWARD_PASS    |
        LSNN_REGRESSION_ERROR;
}

/* returns the classification accuracy of the network */
FloatType LongShortTermMemorySparseKernelCaller::getAccuracy() {
    FloatType sumRight = 0;
    for (unsigned i = 0; i < numOutputNeurons; i++) 
        sumRight += classificationAccuracyCPU[numBlocks * numOutputNeurons + i];

    FloatType sumSamples = 0;
    for (unsigned i = 0; i < numOutputNeurons; i++) 
        sumSamples += classificationSamplesCPU[numBlocks * numOutputNeurons + i];

    return sumRight / sumSamples;
}

/* sets the current active device */
void LongShortTermMemorySparseKernelCaller::setDevice(int device) {
    cudaSetDevice(device);
}
