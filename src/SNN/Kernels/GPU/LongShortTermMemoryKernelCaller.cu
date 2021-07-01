#include "LongShortTermMemoryKernelCaller.h"
#include "GPUArray.cu"
#include "LongShortTermMemoryKernel.cu"
#include "LongShortTermMemoryGradientCollectionKernel.cu"
#include "LongShortTermMemoryFiringRateKernel.cu"
#include "utils.h"

using namespace SNN;
using namespace Kernels;
using namespace GPU;

using std::vector;

/* constructor */
LongShortTermMemoryKernelCaller::LongShortTermMemoryKernelCaller(
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
    vector<FloatType *> inputSpikesOverTime,
    std::vector<FloatType *> spikesOverTime,
    FloatType *firingRates,
    FloatType *inputWeights,
    FloatType *hiddenWeights,
    FloatType *outputWeights,
    FloatType *feedbackWeights,
    FloatType *targetWeights,
    vector<FloatType *> targetsOverTime,
    vector<FloatType *> outputsOverTime,
    vector<FloatType *> outputErrorsOverTime,
    vector<FloatType *> errorMaskOverTime,
    vector<FloatType *> outputErrorFactorOverTime,
    FloatType *inputGradients,
    FloatType *inputFiringRateGradients,
    FloatType *hiddenGradients,
    FloatType *hiddenFiringRateGradients,
    FloatType *leakyReadoutGradients,
    FloatType *inputErrorsOverTime,
    vector<FloatType *> allInputErrorsOverTime
) :
    backPropagation(                new int[1]),
    starttime(                      new int[1]),
    endtime(                        new int[1]),
    numBlocks(batchSize),
    numThreads(std::max(std::max(numHidden, numInputs), numOutputs)),
    numHiddenNeurons(numHidden),
    numInputNeurons(numInputs),
    numOutputNeurons(numOutputs),
    numTimeSteps(numSimulationTimesteps),
    batchErrors(                    new FloatType[batchSize]),
    summedTargets(                  new FloatType[batchSize]),
    squaredSummedTargets(           new FloatType[batchSize]),
    numSummedValues(                new FloatType[batchSize]),
    classificationAccuracyCPU(      new FloatType[batchSize]),
    classificationSamplesCPU(       new FloatType[batchSize]),
    useBackPropagation(             new GPUArray<int>(      backPropagation, 1)),
    numInputs(                      new GPUArray<unsigned>( numInputs)),
    numStandartHidden(              new GPUArray<unsigned>( numStandartHidden)),
    numAdaptiveHidden(              new GPUArray<unsigned>( numAdaptiveHidden)),
    numOutputs(                     new GPUArray<unsigned>( numOutputs)),
    batchSize(                      new GPUArray<unsigned>( batchSize)),
    numSimulationTimesteps(         new GPUArray<unsigned>( numSimulationTimesteps)),
    startTime(                      new GPUArray<int>(      starttime, 1)),
    endTime(                        new GPUArray<int>(      endtime, 1)),
    errorMode(                      new GPUArray<unsigned>( errorMode)),
    timeStepLength(                 new GPUArray<FloatType>(timeStepLength)),
    spikeThreshold(                 new GPUArray<FloatType>(spikeThreshold)),
    refactoryPeriod(                new GPUArray<FloatType>(refactoryPeriod)),
    hiddenDecayFactor(              new GPUArray<FloatType>(hiddenDecayFactor)),
    readoutDecayFactor(             new GPUArray<FloatType>(readoutDecayFactor)),
    adaptationDecayFactor(          new GPUArray<FloatType>(adaptationDecayFactor)),
    thresholdIncreaseConstant(      new GPUArray<FloatType>(thresholdIncreaseConstant)),
    targetFiringRate(               new GPUArray<FloatType>(targetFiringRate)),
    firingRateScallingFactor(       new GPUArray<FloatType>(firingRateScallingFactor)),
    derivativeDumpingFactor(        new GPUArray<FloatType>(derivativeDumpingFactor)),
    inputSpikesOverTime(            new GPUArray<FloatType>(inputSpikesOverTime,                                           numInputs * numSimulationTimesteps)),
    spikesOverTime(                 new GPUArray<FloatType>(spikesOverTime,                                                (numInputs + numHidden) * numSimulationTimesteps)),
    firingRates(                    new GPUArray<FloatType>(firingRates,                                                   numHidden)),
    numSpikes(                      new GPUArray<FloatType>(vector<FloatType *>(batchSize, NULL),                          numHidden + numInputs)),
    inputWeights(                   new GPUArray<FloatType>(inputWeights,                                                  numInputs * numHidden)),
    hiddenWeights(                  new GPUArray<FloatType>(hiddenWeights,                                                 numHidden * numHidden)),
    outputWeights(                  new GPUArray<FloatType>(outputWeights,                                                 numHidden * numOutputs)),
    feedbackWeights(                new GPUArray<FloatType>(feedbackWeights,                                               numHidden * numOutputs)),
    targetWeights(                  new GPUArray<FloatType>(targetWeights,                                                 numOutputs)),
    targetsOverTime(                new GPUArray<FloatType>(targetsOverTime,                                               numOutputs * numSimulationTimesteps)),
    outputsOverTime(                new GPUArray<FloatType>(outputsOverTime,                                               numOutputs * numSimulationTimesteps)),
    outputErrorsOverTime(           new GPUArray<FloatType>(outputErrorsOverTime,                                          numOutputs * numSimulationTimesteps)),
    derivativesOverTime(            new GPUArray<FloatType>(vector<FloatType *>(batchSize, NULL),                          numHidden * numSimulationTimesteps)),
    oldDerivativesOverTime(         new GPUArray<FloatType>(vector<FloatType *>(batchSize, NULL),                          numHidden * numSimulationTimesteps)),
    voltageOverTime(                new GPUArray<FloatType>(vector<FloatType *>(batchSize, NULL),                          numHidden * numSimulationTimesteps)),
    timeStepsSinceLastSpikeOverTime(new GPUArray<FloatType>(vector<FloatType *>(batchSize, NULL),                          numHidden * numSimulationTimesteps)),
    thresholdAdaptationOverTime(    new GPUArray<FloatType>(vector<FloatType *>(batchSize, NULL),                          numAdaptiveHidden * numSimulationTimesteps)),
    errorMaskOverTime(              new GPUArray<FloatType>(errorMaskOverTime,                                             numSimulationTimesteps)),
    outputErrorFactorOverTime(      new GPUArray<FloatType>(outputErrorFactorOverTime,                                     numOutputs * numSimulationTimesteps)),
    inputGradients(                 new GPUArray<FloatType>(vector<FloatType *>(batchSize, inputGradients),                numInputs * numHidden)),
    inputFiringRateGradients(       new GPUArray<FloatType>(vector<FloatType *>(batchSize, inputFiringRateGradients),      numInputs * numHidden)),
    hiddenGradients(                new GPUArray<FloatType>(vector<FloatType *>(batchSize, hiddenGradients),               numHidden * numHidden)),
    hiddenFiringRateGradients(      new GPUArray<FloatType>(vector<FloatType *>(batchSize, hiddenFiringRateGradients),     numHidden * numHidden)),
    leakyReadoutGradients(          new GPUArray<FloatType>(vector<FloatType *>(batchSize, leakyReadoutGradients),         numHidden * numOutputs)),
    networkError(                   new GPUArray<FloatType>(batchErrors,                                                   batchSize)),
    networkTargets(                 new GPUArray<FloatType>(summedTargets,                                                 batchSize)),
    networkSquaredTargets(          new GPUArray<FloatType>(squaredSummedTargets,                                          batchSize)),
    summedValues(                   new GPUArray<FloatType>(numSummedValues,                                               batchSize)),
    classificationAccuracy(         new GPUArray<FloatType>(classificationAccuracyCPU,                                     batchSize)),
    classificationSamples(          new GPUArray<FloatType>(classificationSamplesCPU,                                      batchSize)),
    filteredEligibilityTraces(      new GPUArray<FloatType>(vector<FloatType *>(batchSize, NULL),                          (numInputs + numHidden) * numHidden)),
    filteredSpikes(                 new GPUArray<FloatType>(vector<FloatType *>(batchSize, NULL),                          numInputs + numHidden)),
    readoutDecayFilteredSpikes(     new GPUArray<FloatType>(vector<FloatType *>(batchSize, NULL),                          numHidden)),
    thresholdAdaptation(            new GPUArray<FloatType>(vector<FloatType *>(batchSize, NULL),                          numHidden)),
    adaptionEligibility(            new GPUArray<FloatType>(vector<FloatType *>(batchSize, NULL),                          (numInputs + numHidden) * numAdaptiveHidden)),
    derivatives(                    new GPUArray<FloatType>(vector<FloatType *>(batchSize, NULL),                          numHidden)),
    I(                              new GPUArray<FloatType>(vector<FloatType *>(batchSize, NULL),                          numHidden + numOutputs)),
    v(                              new GPUArray<FloatType>(vector<FloatType *>(batchSize, NULL),                          numHidden + numOutputs)),
    hiddenSpikes(                   new GPUArray<FloatType>(vector<FloatType *>(batchSize, NULL),                          numHidden)),
    timeStepsSinceLastSpike(        new GPUArray<FloatType>(vector<FloatType *>(batchSize, NULL),                          numHidden)),
    learnSignals(                   new GPUArray<FloatType>(vector<FloatType *>(batchSize, NULL),                          numHidden)),
    deltaErrorsVoltage(             new GPUArray<FloatType>(vector<FloatType *>(batchSize, NULL),                          numHidden)),
    deltaErrorsAdaption(            new GPUArray<FloatType>(vector<FloatType *>(batchSize, NULL),                          numAdaptiveHidden)),
    inputErrorsOverTime(            new GPUArray<FloatType>(inputErrorsOverTime,                                           numInputs * numSimulationTimesteps)),
    allInputErrorsOverTime(         new GPUArray<FloatType>(allInputErrorsOverTime,                                        numInputs * numSimulationTimesteps)),
    filteredOutputErrors(           new GPUArray<FloatType>(vector<FloatType *>(batchSize, NULL),                          numOutputs)),
    summedActivation(               new GPUArray<FloatType>(vector<FloatType *>(batchSize, NULL),                          numOutputs))
    {
        
    this->numInputs->copyToDevice();
    this->numStandartHidden->copyToDevice();
    this->numAdaptiveHidden->copyToDevice();
    this->numOutputs->copyToDevice();
    this->batchSize->copyToDevice();
    this->numSimulationTimesteps->copyToDevice();
    this->errorMode->copyToDevice();
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

    unsigned memory = 0;
    memory += this->useBackPropagation->globalMemoryConsumption();
    memory += this->numInputs->globalMemoryConsumption();
    memory += this->numStandartHidden->globalMemoryConsumption();
    memory += this->numAdaptiveHidden->globalMemoryConsumption();
    memory += this->numOutputs->globalMemoryConsumption();
    memory += this->batchSize->globalMemoryConsumption();
    memory += this->numSimulationTimesteps->globalMemoryConsumption();
    memory += this->startTime->globalMemoryConsumption();
    memory += this->endTime->globalMemoryConsumption();
    memory += this->errorMode->globalMemoryConsumption();
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
    memory += this->numSpikes->globalMemoryConsumption();
    memory += this->inputWeights->globalMemoryConsumption();
    memory += this->hiddenWeights->globalMemoryConsumption();
    memory += this->outputWeights->globalMemoryConsumption();
    memory += this->feedbackWeights->globalMemoryConsumption();
    memory += this->targetWeights->globalMemoryConsumption();
    memory += this->targetsOverTime->globalMemoryConsumption();
    memory += this->outputsOverTime->globalMemoryConsumption();
    memory += this->outputErrorsOverTime->globalMemoryConsumption();
    memory += this->derivativesOverTime->globalMemoryConsumption();
    memory += this->oldDerivativesOverTime->globalMemoryConsumption();
    memory += this->voltageOverTime->globalMemoryConsumption();
    memory += this->timeStepsSinceLastSpikeOverTime->globalMemoryConsumption();
    memory += this->thresholdAdaptationOverTime->globalMemoryConsumption();
    memory += this->errorMaskOverTime->globalMemoryConsumption();
    memory += this->outputErrorFactorOverTime->globalMemoryConsumption();
    memory += this->inputGradients->globalMemoryConsumption();
    memory += this->inputFiringRateGradients->globalMemoryConsumption();
    memory += this->hiddenGradients->globalMemoryConsumption();
    memory += this->hiddenFiringRateGradients->globalMemoryConsumption();
    memory += this->leakyReadoutGradients->globalMemoryConsumption();
    memory += this->networkError->globalMemoryConsumption();
    memory += this->networkTargets->globalMemoryConsumption();
    memory += this->networkSquaredTargets->globalMemoryConsumption();
    memory += this->summedValues->globalMemoryConsumption();
    memory += this->classificationAccuracy->globalMemoryConsumption();
    memory += this->classificationSamples->globalMemoryConsumption();
    memory += this->filteredEligibilityTraces->globalMemoryConsumption();
    memory += this->filteredSpikes->globalMemoryConsumption();
    memory += this->readoutDecayFilteredSpikes->globalMemoryConsumption();
    memory += this->thresholdAdaptation->globalMemoryConsumption();
    memory += this->adaptionEligibility->globalMemoryConsumption();
    memory += this->derivatives->globalMemoryConsumption();
    memory += this->I->globalMemoryConsumption();
    memory += this->v->globalMemoryConsumption();
    memory += this->hiddenSpikes->globalMemoryConsumption();
    memory += this->timeStepsSinceLastSpike->globalMemoryConsumption();
    memory += this->learnSignals->globalMemoryConsumption();
    memory += this->deltaErrorsVoltage->globalMemoryConsumption();
    memory += this->deltaErrorsAdaption->globalMemoryConsumption();
    memory += this->inputErrorsOverTime->globalMemoryConsumption();
    memory += this->allInputErrorsOverTime->globalMemoryConsumption();
    memory += this->filteredOutputErrors->globalMemoryConsumption();
    memory += this->summedActivation->globalMemoryConsumption();
    log_str("globalMemoryConsumption: " + itoa(memory), LOG_I);
}


/* destructor */
LongShortTermMemoryKernelCaller::~LongShortTermMemoryKernelCaller() {

    delete[] backPropagation;
    delete[] starttime;
    delete[] endtime;
    delete[] batchErrors;
    delete[] classificationAccuracyCPU;
    delete[] classificationSamplesCPU;
    delete useBackPropagation;
    delete numInputs;
    delete numStandartHidden;
    delete numAdaptiveHidden;
    delete numOutputs;
    delete batchSize;
    delete numSimulationTimesteps;
    delete startTime;
    delete endTime;
    delete errorMode;
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
    delete inputWeights;
    delete hiddenWeights;
    delete outputWeights;
    delete feedbackWeights;
    delete targetWeights;
    delete targetsOverTime;
    delete outputsOverTime;
    delete outputErrorsOverTime;
    delete derivativesOverTime;
    delete oldDerivativesOverTime;
    delete voltageOverTime;
    delete timeStepsSinceLastSpikeOverTime;
    delete thresholdAdaptationOverTime;
    delete errorMaskOverTime;
    delete outputErrorFactorOverTime;
    delete inputGradients;
    delete inputFiringRateGradients;
    delete hiddenGradients;
    delete hiddenFiringRateGradients;
    delete leakyReadoutGradients;
    delete networkError;
    delete networkTargets;
    delete networkSquaredTargets;
    delete summedValues;
    delete classificationAccuracy;
    delete classificationSamples;
    delete filteredEligibilityTraces;
    delete filteredSpikes;
    delete readoutDecayFilteredSpikes;
    delete thresholdAdaptation;
    delete adaptionEligibility;
    delete derivatives;
    delete I;
    delete v;
    delete hiddenSpikes;
    delete timeStepsSinceLastSpike;
    delete learnSignals;
    delete deltaErrorsVoltage;
    delete deltaErrorsAdaption;
    delete inputErrorsOverTime;
    delete allInputErrorsOverTime;
    delete filteredOutputErrors;
    delete summedActivation;
}


/* runs the kernel of this (blocks untill finished) */
void LongShortTermMemoryKernelCaller::runAndWait(
    int backPropagation, 
    bool inputErrors,
    int starttime,
    int endtime
) {
    static uint64_t cpuTime = 0;
    static uint64_t gpuTime = 0;
    static uint64_t t = gettime_usec();

    this->starttime[0] = starttime;
    this->endtime[0]   = endtime;

    cpuTime += gettime_usec() - t;
    t = gettime_usec();

    this->backPropagation[0] = backPropagation;
    useBackPropagation->copyToDevice();
    targetWeights->copyToDevice();
    inputWeights->copyToDevice();
    hiddenWeights->copyToDevice();
    outputWeights->copyToDevice();
    startTime->copyToDevice();
    endTime->copyToDevice();

    if (starttime == 0 && endtime == int(numTimeSteps)) {
        inputSpikesOverTime->copyToDevice();
        targetsOverTime->copyToDevice();
        errorMaskOverTime->copyToDevice();
        outputErrorFactorOverTime->copyToDevice();
        outputErrorsOverTime->copyToDevice(); 
    } else {
        inputSpikesOverTime->copyToDevice(
           (starttime % numTimeSteps) * numInputNeurons,
           (endtime - starttime) * numInputNeurons
        );
        targetsOverTime->copyToDevice(
           (starttime % numTimeSteps) * numOutputNeurons,
           (endtime - starttime) * numOutputNeurons
        );
        errorMaskOverTime->copyToDevice(
           starttime % numTimeSteps,
           endtime - starttime
        );
        outputErrorFactorOverTime->copyToDevice(
           (starttime % numTimeSteps) * numOutputNeurons,
           (endtime - starttime) * numOutputNeurons
        );
        outputErrorsOverTime->copyToDevice(
           (starttime % numTimeSteps) * numOutputNeurons,
           (endtime - starttime) * numOutputNeurons
        ); 
    }

    log_str("launching longShortTermMemoryKernel<<<" + 
            itoa(numBlocks) + ", " + itoa(numThreads) + ">>>", LOG_DD);

    longShortTermMemoryKernel<<<numBlocks, numThreads>>>(
        (int       *) useBackPropagation->d_ptr(),
        (unsigned  *) numInputs->d_ptr(),
        (unsigned  *) numStandartHidden->d_ptr(),
        (unsigned  *) numAdaptiveHidden->d_ptr(),
        (unsigned  *) numOutputs->d_ptr(),
        (unsigned  *) numSimulationTimesteps->d_ptr(),
        (int       *) startTime->d_ptr(),
        (int       *) endTime->d_ptr(),
        (unsigned  *) errorMode->d_ptr(),
        (FloatType *) timeStepLength->d_ptr(),
        (FloatType *) spikeThreshold->d_ptr(),
        (FloatType *) refactoryPeriod->d_ptr(),
        (FloatType *) hiddenDecayFactor->d_ptr(),
        (FloatType *) readoutDecayFactor->d_ptr(),
        (FloatType *) adaptationDecayFactor->d_ptr(),
        (FloatType *) thresholdIncreaseConstant->d_ptr(),
        (FloatType *) targetFiringRate->d_ptr(),
        (FloatType *) firingRateScallingFactor->d_ptr(),
        (FloatType *) derivativeDumpingFactor->d_ptr(),
        (FloatType *) inputSpikesOverTime->d_ptr(),
        (FloatType *) spikesOverTime->d_ptr(),
        (FloatType *) firingRates->d_ptr(),
        (FloatType *) numSpikes->d_ptr(),
        (FloatType *) inputWeights->d_ptr(),
        (FloatType *) hiddenWeights->d_ptr(),
        (FloatType *) outputWeights->d_ptr(),
        (FloatType *) feedbackWeights->d_ptr(),
        (FloatType *) targetWeights->d_ptr(),
        (FloatType *) targetsOverTime->d_ptr(),
        (FloatType *) outputsOverTime->d_ptr(),
        (FloatType *) outputErrorsOverTime->d_ptr(),
        (FloatType *) derivativesOverTime->d_ptr(),
        (FloatType *) oldDerivativesOverTime->d_ptr(),
        (FloatType *) voltageOverTime->d_ptr(),
        (FloatType *) timeStepsSinceLastSpikeOverTime->d_ptr(),
        (FloatType *) thresholdAdaptationOverTime->d_ptr(),
        (FloatType *) errorMaskOverTime->d_ptr(),
        (FloatType *) outputErrorFactorOverTime->d_ptr(),
        (FloatType *) inputGradients->d_ptr(),
        (FloatType *) inputFiringRateGradients->d_ptr(),
        (FloatType *) hiddenGradients->d_ptr(),
        (FloatType *) hiddenFiringRateGradients->d_ptr(),
        (FloatType *) leakyReadoutGradients->d_ptr(),
        (FloatType *) networkError->d_ptr(),
        (FloatType *) networkTargets->d_ptr(),
        (FloatType *) networkSquaredTargets->d_ptr(),
        (FloatType *) summedValues->d_ptr(),
        (FloatType *) classificationAccuracy->d_ptr(),
        (FloatType *) classificationSamples->d_ptr(),
        (FloatType *) filteredEligibilityTraces->d_ptr(),
        (FloatType *) filteredSpikes->d_ptr(),
        (FloatType *) readoutDecayFilteredSpikes->d_ptr(),
        (FloatType *) thresholdAdaptation->d_ptr(),
        (FloatType *) adaptionEligibility->d_ptr(),
        (FloatType *) derivatives->d_ptr(),
        (FloatType *) I->d_ptr(),
        (FloatType *) v->d_ptr(),
        (FloatType *) hiddenSpikes->d_ptr(),
        (FloatType *) timeStepsSinceLastSpike->d_ptr(),
        (FloatType *) learnSignals->d_ptr(),
        (FloatType *) deltaErrorsVoltage->d_ptr(),
        (FloatType *) deltaErrorsAdaption->d_ptr(),
        (FloatType *) inputErrorsOverTime->d_ptr(),
        (FloatType *) allInputErrorsOverTime->d_ptr(),
        (FloatType *) filteredOutputErrors->d_ptr(),
        (FloatType *) summedActivation->d_ptr()
    );
    
    if (backPropagation == BACKPROPAGATION_OFF  || 
        backPropagation == BACKPROPAGATION_FULL || 
        backPropagation == BACKPROPAGATION_BACKWARD) {

        // TODO parametrisize blocks and threads
        if (numBlocks > 1 || inputErrors) {
            longShortTermMemoryGradientCollectionKernel<<<16, 1024>>>(
                (unsigned  *) numInputs->d_ptr(),
                (unsigned  *) numStandartHidden->d_ptr(),
                (unsigned  *) numAdaptiveHidden->d_ptr(),
                (unsigned  *) numOutputs->d_ptr(),
                (unsigned  *) batchSize->d_ptr(),
                (unsigned  *) numSimulationTimesteps->d_ptr(),
                (FloatType *) inputGradients->d_ptr(),
                (FloatType *) inputFiringRateGradients->d_ptr(),
                (FloatType *) hiddenGradients->d_ptr(),
                (FloatType *) hiddenFiringRateGradients->d_ptr(),
                (FloatType *) leakyReadoutGradients->d_ptr(),
                (FloatType *) inputErrorsOverTime->d_ptr(),
                (FloatType *) allInputErrorsOverTime->d_ptr()
            );
        }

        longShortTermMemoryFiringRateKernel<<<1, numHiddenNeurons>>>(
            (unsigned  *) numInputs->d_ptr(),
            (unsigned  *) numStandartHidden->d_ptr(),
            (unsigned  *) numAdaptiveHidden->d_ptr(),
            (unsigned  *) batchSize->d_ptr(),
            (unsigned  *) numSimulationTimesteps->d_ptr(),
            (FloatType *) timeStepLength->d_ptr(),
            (FloatType *) firingRates->d_ptr(),
            (FloatType *) numSpikes->d_ptr()
        );
    }

    if (backPropagation != BACKPROPAGATION_FORWARD) {
        inputGradients->copyToHost(0);
        inputFiringRateGradients->copyToHost(0);
        hiddenGradients->copyToHost(0);
        hiddenFiringRateGradients->copyToHost(0);
        leakyReadoutGradients->copyToHost(0);
    }
    firingRates->copyToHost();
    networkError->copyToHost();
    networkTargets->copyToHost();
    networkSquaredTargets->copyToHost();
    summedValues->copyToHost();
    classificationAccuracy->copyToHost();
    classificationSamples->copyToHost();
    if (starttime == 0 && endtime == int(numTimeSteps)) {
        outputsOverTime->copyToHost();
        spikesOverTime->copyToHost();
    } else {
        outputsOverTime->copyToHost(
            (starttime % numTimeSteps) * numOutputNeurons,
            (endtime - starttime) * numOutputNeurons
        );
        spikesOverTime->copyToHost(
            (starttime % numTimeSteps) * (numInputNeurons + numHiddenNeurons),
            (endtime - starttime) * (numInputNeurons + numHiddenNeurons)
        );
    }

    if (inputErrors) {
        if (starttime == 0 && endtime == int(numTimeSteps)) {
            inputErrorsOverTime->copyToHost();
            allInputErrorsOverTime->copyToHost();
        } else {
            inputErrorsOverTime->copyToHost(
                (starttime % numTimeSteps) * numInputNeurons,
                (endtime - starttime) * numInputNeurons
            );
            allInputErrorsOverTime->copyToHost(
                (starttime % numTimeSteps) * numInputNeurons,
                (endtime - starttime) * numInputNeurons
            );
        }
    }

    int error = cudaGetLastError();
    if (error)
        log_err("LongShortTermMemoryKernel failed: " + itoa(error), LOG_EE);
    
    gpuTime += gettime_usec() - t;
    t = gettime_usec();
    
    log_str("CPU time: " + ftoa(cpuTime / 1000000.0) + ", GPU time: " + ftoa(gpuTime / 1000000.0), LOG_D);
}

/* returns the networks spuared error for the last run*/
FloatType LongShortTermMemoryKernelCaller::getSampleSquaredSummedError(unsigned batch) {
    return batchErrors[batch];
}
FloatType LongShortTermMemoryKernelCaller::getSquaredSummedError() {
    FloatType error = 0;
    for (unsigned i = 0; i < numBlocks; i++)
        error += batchErrors[i];

    return error;
}

/* returns the networks summed target for the last run */
FloatType LongShortTermMemoryKernelCaller::getSampleSummedTarget(unsigned batch) {
    return summedTargets[batch];
}
FloatType LongShortTermMemoryKernelCaller::getSummedTarget() {
    FloatType error = 0;
    for (unsigned i = 0; i < numBlocks; i++)
        error += summedTargets[i];

    return error;
}

/* returns the networks squared summed target for the last run */
FloatType LongShortTermMemoryKernelCaller::getSquaredSummedTarget() {
    FloatType error = 0;
    for (unsigned i = 0; i < numBlocks; i++)
        error += squaredSummedTargets[i];

    return error;
}

/* returns the the number of summed values for the last run */
FloatType LongShortTermMemoryKernelCaller::getSampleNumSummedValues(unsigned batch) {
    return numSummedValues[batch];
}
FloatType LongShortTermMemoryKernelCaller::getNumSummedValues() {
    FloatType error = 0;
    for (unsigned i = 0; i < numBlocks; i++)
        error += numSummedValues[i];

    return error;
}

/* returns the networks classification accuracy */
FloatType LongShortTermMemoryKernelCaller::getAccuracy() {
    FloatType accuracy = 0;
    FloatType samples = 0;

    for (unsigned i = 0; i < numBlocks; i++) {
        accuracy += classificationAccuracyCPU[i];
        samples  += classificationSamplesCPU[i];
    }
        
    return accuracy / samples;
}

/* reload feedback weights and other "not changing" values into device */
void LongShortTermMemoryKernelCaller::reload() {
    firingRates->copyToDevice();
    feedbackWeights->copyToDevice();
}

/* sets the current active device */
void LongShortTermMemoryKernelCaller::setDevice(int device) {
    cudaSetDevice(device);
}

