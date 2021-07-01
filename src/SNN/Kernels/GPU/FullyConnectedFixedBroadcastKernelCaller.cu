#include "FullyConnectedFixedBroadcastKernelCaller.h"
#include "GPUArray.cu"
#include "FullyConnectedFixedBroadcastKernel.cu"
#include "utils.h"

using namespace SNN;
using namespace Kernels;
using namespace GPU;

using std::vector;

/* constructor */
FullyConnectedFixedBroadcastKernelCaller::FullyConnectedFixedBroadcastKernelCaller(
    unsigned batchSize,
    unsigned numInputs,
    unsigned numHidden,
    unsigned numOutputs,
    unsigned numSimulationTimesteps,
    FloatType timeStepLength,
    FloatType spikeThreshold,
    FloatType refactoryPeriod,
    FloatType hiddenDecayFactor,
    FloatType readoutDecayFactor,
    FloatType targetFiringRate,
    FloatType firingRateScallingFactor,
    FloatType derivativeDumpingFactor,
    vector<FloatType *> inputSpikesOverTime,
    vector<FloatType *> firingRates,
    vector<FloatType *> numSpikes,
    FloatType *inputWeights,
    FloatType *hiddenWeights,
    FloatType *outputWeights,
    FloatType *feedbackWeights,
    vector<FloatType *> targetsOverTime,
    vector<FloatType *> inputFixedBroadcastGradients,
    vector<FloatType *> inputFiringRateGradients,
    vector<FloatType *> hiddenFixedBroadcastGradients,
    vector<FloatType *> hiddenFiringRateGradients,
    vector<FloatType *> leakyReadoutGradients
) :
    numBlocks(batchSize),
    numThreads(numHidden),
    batchErrors(new FloatType[batchSize]),
    numInputs(                    new GPUArray<unsigned>( numInputs)),
    numHidden(                    new GPUArray<unsigned>( numHidden)),
    numOutputs(                   new GPUArray<unsigned>( numOutputs)),
    numSimulationTimesteps(       new GPUArray<unsigned>( numSimulationTimesteps)),
    timeStepLength(               new GPUArray<FloatType>(timeStepLength)),
    spikeThreshold(               new GPUArray<FloatType>(spikeThreshold)),
    refactoryPeriod(              new GPUArray<FloatType>(refactoryPeriod)),
    hiddenDecayFactor(            new GPUArray<FloatType>(hiddenDecayFactor)),
    readoutDecayFactor(           new GPUArray<FloatType>(readoutDecayFactor)),
    targetFiringRate(             new GPUArray<FloatType>(targetFiringRate)),
    firingRateScallingFactor(     new GPUArray<FloatType>(firingRateScallingFactor)),
    derivativeDumpingFactor(      new GPUArray<FloatType>(derivativeDumpingFactor)),
    inputSpikesOverTime(          new GPUArray<FloatType>(inputSpikesOverTime,                  numInputs * numSimulationTimesteps)),
    firingRates(                  new GPUArray<FloatType>(firingRates,                          numHidden)),
    numSpikes(                    new GPUArray<FloatType>(numSpikes,                            numHidden)),
    inputWeights(                 new GPUArray<FloatType>(inputWeights,                         numInputs * numHidden)),
    hiddenWeights(                new GPUArray<FloatType>(hiddenWeights,                        numHidden * numHidden)),
    outputWeights(                new GPUArray<FloatType>(outputWeights,                        numHidden * numOutputs)),
    feedbackWeights(              new GPUArray<FloatType>(feedbackWeights,                      numHidden * numOutputs)),
    targetsOverTime(              new GPUArray<FloatType>(targetsOverTime,                      numOutputs * numSimulationTimesteps)),
    inputFixedBroadcastGradients( new GPUArray<FloatType>(inputFixedBroadcastGradients,         numInputs * numHidden)),
    inputFiringRateGradients(     new GPUArray<FloatType>(inputFiringRateGradients,             numInputs * numHidden)),
    hiddenFixedBroadcastGradients(new GPUArray<FloatType>(hiddenFixedBroadcastGradients,        numHidden * numHidden)),
    hiddenFiringRateGradients(    new GPUArray<FloatType>(hiddenFiringRateGradients,            numHidden * numHidden)),
    leakyReadoutGradients(        new GPUArray<FloatType>(leakyReadoutGradients,                numHidden * numOutputs)),
    networkError(                 new GPUArray<FloatType>(batchErrors,                          batchSize)),
    filteredEligibilityTraces(    new GPUArray<FloatType>(vector<FloatType *>(batchSize, NULL), numInputs * numHidden + numHidden * numHidden)),
    filteredSpikes(               new GPUArray<FloatType>(vector<FloatType *>(batchSize, NULL), numInputs + numHidden)),
    derivatives(                  new GPUArray<FloatType>(vector<FloatType *>(batchSize, NULL), numHidden)),
    I(                            new GPUArray<FloatType>(vector<FloatType *>(batchSize, NULL), numHidden + numOutputs)),
    v(                            new GPUArray<FloatType>(vector<FloatType *>(batchSize, NULL), numHidden + numOutputs)),
    hiddenSpikes(                 new GPUArray<FloatType>(vector<FloatType *>(batchSize, NULL), numHidden)),
    timeStepsSinceLastSpike(      new GPUArray<FloatType>(vector<FloatType *>(batchSize, NULL), numHidden)) {
        
    this->numInputs->copyToDevice();
    this->numHidden->copyToDevice();
    this->numOutputs->copyToDevice();
    this->numSimulationTimesteps->copyToDevice();
    this->timeStepLength->copyToDevice();
    this->spikeThreshold->copyToDevice();
    this->refactoryPeriod->copyToDevice();
    this->hiddenDecayFactor->copyToDevice();
    this->readoutDecayFactor->copyToDevice();
    this->targetFiringRate->copyToDevice();
    this->firingRateScallingFactor->copyToDevice();
    this->derivativeDumpingFactor->copyToDevice();
    this->feedbackWeights->copyToDevice();
    this->firingRates->copyToDevice();

    unsigned memory = 0;
    memory += this->numInputs->globalMemoryConsumption();
    memory += this->numHidden->globalMemoryConsumption();
    memory += this->numOutputs->globalMemoryConsumption();
    memory += this->numSimulationTimesteps->globalMemoryConsumption();
    memory += this->timeStepLength->globalMemoryConsumption();
    memory += this->spikeThreshold->globalMemoryConsumption();
    memory += this->refactoryPeriod->globalMemoryConsumption();
    memory += this->hiddenDecayFactor->globalMemoryConsumption();
    memory += this->readoutDecayFactor->globalMemoryConsumption();
    memory += this->targetFiringRate->globalMemoryConsumption();
    memory += this->firingRateScallingFactor->globalMemoryConsumption();
    memory += this->derivativeDumpingFactor->globalMemoryConsumption();
    memory += this->inputSpikesOverTime->globalMemoryConsumption();
    memory += this->firingRates->globalMemoryConsumption();
    memory += this->numSpikes->globalMemoryConsumption();
    memory += this->inputWeights->globalMemoryConsumption();
    memory += this->hiddenWeights->globalMemoryConsumption();
    memory += this->outputWeights->globalMemoryConsumption();
    memory += this->feedbackWeights->globalMemoryConsumption();
    memory += this->targetsOverTime->globalMemoryConsumption();
    memory += this->inputFixedBroadcastGradients->globalMemoryConsumption();
    memory += this->inputFiringRateGradients->globalMemoryConsumption();
    memory += this->hiddenFixedBroadcastGradients->globalMemoryConsumption();
    memory += this->hiddenFiringRateGradients->globalMemoryConsumption();
    memory += this->leakyReadoutGradients->globalMemoryConsumption();
    memory += this->networkError->globalMemoryConsumption();
    memory += this->filteredEligibilityTraces->globalMemoryConsumption();
    memory += this->filteredSpikes->globalMemoryConsumption();
    memory += this->derivatives->globalMemoryConsumption();
    memory += this->I->globalMemoryConsumption();
    memory += this->v->globalMemoryConsumption();
    memory += this->hiddenSpikes->globalMemoryConsumption();
    memory += this->timeStepsSinceLastSpike->globalMemoryConsumption();
    log_str("globalMemoryConsumption: " + itoa(memory), LOG_D);
}


/* destructor */
FullyConnectedFixedBroadcastKernelCaller::~FullyConnectedFixedBroadcastKernelCaller() {

    delete batchErrors;
    delete numInputs;
    delete numHidden;
    delete numOutputs;
    delete numSimulationTimesteps;
    delete timeStepLength;
    delete spikeThreshold;
    delete refactoryPeriod;
    delete hiddenDecayFactor;
    delete readoutDecayFactor;
    delete targetFiringRate;
    delete firingRateScallingFactor;
    delete derivativeDumpingFactor;
    delete inputSpikesOverTime;
    delete firingRates;
    delete numSpikes;
    delete inputWeights;
    delete hiddenWeights;
    delete outputWeights;
    delete feedbackWeights;
    delete targetsOverTime;
    delete inputFixedBroadcastGradients;
    delete inputFiringRateGradients;
    delete hiddenFixedBroadcastGradients;
    delete hiddenFiringRateGradients;
    delete leakyReadoutGradients;
    delete networkError;
    delete filteredEligibilityTraces;
    delete filteredSpikes;
    delete derivatives;
    delete I;
    delete v;
    delete hiddenSpikes;
    delete timeStepsSinceLastSpike;
}


/* runs the kernel of this (blocks untill finished) */
void FullyConnectedFixedBroadcastKernelCaller::runAndWait() {

    inputSpikesOverTime->copyToDevice();
    targetsOverTime->copyToDevice();
    inputWeights->copyToDevice();
    hiddenWeights->copyToDevice();
    outputWeights->copyToDevice();

    log_str("launching fullyConnectedFixedBroadcastKernel<<<" + 
            itoa(numBlocks) + ", " + itoa(numThreads) + ">>>", LOG_D);

    fullyConnectedFixedBroadcastKernel<<<numBlocks, numThreads>>>(
        (unsigned     *) numInputs->d_ptr(),
        (unsigned     *) numHidden->d_ptr(),
        (unsigned     *) numOutputs->d_ptr(),
        (unsigned     *) numSimulationTimesteps->d_ptr(),
        (FloatType    *) timeStepLength->d_ptr(),
        (FloatType    *) spikeThreshold->d_ptr(),
        (FloatType    *) refactoryPeriod->d_ptr(),
        (FloatType    *) hiddenDecayFactor->d_ptr(),
        (FloatType    *) readoutDecayFactor->d_ptr(),
        (FloatType    *) targetFiringRate->d_ptr(),
        (FloatType    *) firingRateScallingFactor->d_ptr(),
        (FloatType    *) derivativeDumpingFactor->d_ptr(),
        (FloatType    *) inputSpikesOverTime->d_ptr(),
        (FloatType    *) firingRates->d_ptr(),
        (FloatType    *) numSpikes->d_ptr(),
        (FloatType    *) inputWeights->d_ptr(),
        (FloatType    *) hiddenWeights->d_ptr(),
        (FloatType    *) outputWeights->d_ptr(),
        (FloatType    *) feedbackWeights->d_ptr(),
        (FloatType    *) targetsOverTime->d_ptr(),
        (FloatType    *) inputFixedBroadcastGradients->d_ptr(),
        (FloatType    *) inputFiringRateGradients->d_ptr(),
        (FloatType    *) hiddenFixedBroadcastGradients->d_ptr(),
        (FloatType    *) hiddenFiringRateGradients->d_ptr(),
        (FloatType    *) leakyReadoutGradients->d_ptr(),
        (FloatType    *) networkError->d_ptr(),
        (FloatType    *) filteredEligibilityTraces->d_ptr(),
        (FloatType    *) filteredSpikes->d_ptr(),
        (FloatType    *) derivatives->d_ptr(),
        (FloatType    *) I->d_ptr(),
        (FloatType    *) v->d_ptr(),
        (FloatType    *) hiddenSpikes->d_ptr(),
        (FloatType    *) timeStepsSinceLastSpike->d_ptr()
    );

    inputFixedBroadcastGradients->copyToHost();
    inputFiringRateGradients->copyToHost();
    hiddenFixedBroadcastGradients->copyToHost();
    hiddenFiringRateGradients->copyToHost();
    leakyReadoutGradients->copyToHost();
    networkError->copyToHost();
    numSpikes->copyToHost();

    if (cudaGetLastError())
        log_err("FullyConnectedFixedBroadcastKernel failed", LOG_EE);
}

/* returns the networks normalized mean spuared error for the last run*/
FloatType FullyConnectedFixedBroadcastKernelCaller::getError(int batchIndex) {
    return batchErrors[batchIndex];
}
