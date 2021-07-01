#include "LongShortTermMemoryEligibilityNetwork.h"
#include "CurrentInputNeuron.h"
#include "LeakyReadoutSynapse.h"
#include "LeakyIntegrateAndFireSynapse.h"
#include "AdaptiveLeakyIntegrateAndFireSynapse.h"
#include "AdamOptimizer.h"
#include "AMSGradOptimizer.h"
#include "SignDampedMomentumOptimizer.h"
#include "StochasticGradientDescentOptimizer.h"
#include "BackPropagatedGradient.h"
#include "EligibilityBackPropagatedGradient.h"
#include "FixedBroadcastGradient.h"
#include "FiringRateGradient.h"
#include "FastFiringRateGradient.h"
#include "LeakyReadoutGradient.h"
#include "LongShortTermMemoryKernelCaller.h"
#include "LongShortTermMemorySparseKernelCaller.h"
#include "BasicNeuronSerializer.h"
#include "BasicSynapseSerializer.h"
#include "BasicGradientSerializer.h"
#include "FeedbackWeightsSerializer.h"
#include "SmartTrainSet.h"
#include <limits>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

using std::shared_ptr;
using std::static_pointer_cast;
using std::dynamic_pointer_cast;
using std::make_shared;
using std::vector;
using std::numeric_limits;

#define ERROR_MODE_REGRESSION 0
#define ERROR_MODE_CLASSIFICATION 1

using namespace SNN;
using namespace Interfaces;
using namespace Neurons;
using namespace Synapses;
using namespace Gradients;
using namespace Optimizers;
using namespace Networks;
using namespace Options;
using namespace Kernels;
using namespace GPU;
using namespace Visualizations;

/* should perform updates bevor updating the network */
void LongShortTermMemoryEligibilityNetwork::updateBevor() {

    unsigned offset = timeStepCounter * (opts.numInputNeurons() + opts.numHiddenNeurons());
    for (unsigned i = 0; i < opts.numInputNeurons(); i++) {
        inputNeurons[i]->addCurrent(
            inputsOverTime[sampleIndex][timeStepCounter * opts.numInputNeurons() + i]
        );
        spikesOverTime[sampleIndex][offset + i] = 
            inputsOverTime[sampleIndex][timeStepCounter * opts.numInputNeurons() + i];
    }
}

/* should perform the actual update stepp */
void LongShortTermMemoryEligibilityNetwork::doUpdate() { 

    for (auto &n: inputNeurons)   n->update();
    for (auto &s: inputSynapses)  s->propagateSpike();

    for (auto &n: hiddenNeurons)  n->update();
    for (auto &s: inputSynapses)  s->computeEligibilityTrace(true /* input synapse */);

    for (auto &s: hiddenSynapses) s->update();
    for (auto &s: outputSynapses) s->propagateSpike();

    for (auto &n: outputNeurons)  n->update();
    for (auto &s: outputSynapses) s->computeEligibilityTrace(false /* not an input synapse */);

    for (auto &g: gradients)      g->update();
}

/* should perform updates after updating the network */
void LongShortTermMemoryEligibilityNetwork::updateAfter() {

    unsigned offset = timeStepCounter * (opts.numInputNeurons() + opts.numHiddenNeurons()) + opts.numInputNeurons();
    for (unsigned i = 0; i < opts.numHiddenNeurons(); i++)
        spikesOverTime[sampleIndex][offset + i] = hiddenNeurons[i]->getOutput();

    for (unsigned i = 0; i < opts.numOutputNeurons(); i++) {
        if (errorMaskOverTime[sampleIndex][timeStepCounter] != 0) {
            if (opts.errorMode() == ERROR_MODE_CLASSIFICATION ||
                opts.errorMode() == ERROR_MODE_INFINITY) {

                summedError -= targetAt(sampleIndex, timeStepCounter, i) * 
                               log(this->getSoftmaxOutput(i));

                summedActivation[i] += outputNeurons[i]->getOutput();
            } else {
                summedError += pow(
                    targetAt(sampleIndex, timeStepCounter, i) - 
                    outputNeurons[i]->getOutput(), 2
                );
                summedTarget += targetAt(sampleIndex, timeStepCounter, i);
                squaredSummedTarget += pow(targetAt(sampleIndex, timeStepCounter, i), 2);
            }
            numErrorValues += opts.numOutputNeurons();
        }
    }

    if ((opts.errorMode() == ERROR_MODE_CLASSIFICATION ||
        opts.errorMode() == ERROR_MODE_INFINITY) &&
        errorMaskOverTime[sampleIndex][timeStepCounter] && (
            timeStepCounter + 1 == opts.numSimulationTimesteps() || 
            errorMaskOverTime[sampleIndex][timeStepCounter + 1] == 0
        )) {
        
        unsigned maxNeuron = 0;
        for (unsigned i = 1; i < summedActivation.size(); i++)
            if (summedActivation[i] > summedActivation[maxNeuron])
                maxNeuron = i;

        classificationAccuracy += targetAt(sampleIndex, timeStepCounter, maxNeuron);
        classificationSamples++;

        for (unsigned i = 1; i < summedActivation.size(); i++)
            summedActivation[i] = 0;
    }    

    for (unsigned o = 0; o < outputNeurons.size(); o++) {
        outputsOverTime[sampleIndex][timeStepCounter * outputNeurons.size() + o] = outputNeurons[o]->getOutput();
    }

    timeStepCounter++;
    if (timeStepCounter >= opts.numSimulationTimesteps()) {
        if (!opts.useBackPropagation()) {
            if (opts.shuffleBatch())
                sampleIndex = rand128(rand) % opts.batchSize();
            else
                sampleIndex = (sampleIndex + 1) % opts.batchSize();

            batchCounter++;
            timeStepCounter = 0;
            if (batchCounter == opts.batchSize()) {

                summedErrorLast            = summedError;
                summedTargetLast           = summedTarget;
                squaredSummedTargetLast    = squaredSummedTarget;
                numErrorValuesLast         = numErrorValues;
                classificationAccuracyLast = classificationAccuracy / classificationSamples;
                summedError                = 0;
                summedTarget               = 0;
                squaredSummedTarget        = 0;
                batchCounter               = 0;
                numErrorValues             = 0;
                classificationAccuracy     = 0;
                classificationSamples      = 0;
            }
            this->resetNetwork();
        }
    }
}

/* should perform updates bevor a back propagation step of the network */
void LongShortTermMemoryEligibilityNetwork::backPropagateBevor() {
    
    if (batchCounter == 0 && timeStepCounter == opts.numSimulationTimesteps())
        memset(inputErrorsOverTime, 0, sizeof(FloatType) * inputNeurons.size() * opts.numSimulationTimesteps());

    timeStepCounter--;

    for (unsigned i = 0; i < opts.numOutputNeurons(); i++) {
        if (errorMaskOverTime[sampleIndex][timeStepCounter] != 0) {
            outputNeurons[i]->addError(
                (
                    outputsOverTime[sampleIndex][timeStepCounter * outputNeurons.size() + i] -
                    targetsOverTime[sampleIndex][timeStepCounter * outputNeurons.size() + i]
                ) * targetWeights[i] * this->outputErrorFactorAt(sampleIndex, timeStepCounter, i)
            );
        }
    }
}

/* should perform the actual backpropagation step */
void LongShortTermMemoryEligibilityNetwork::doBackPropagation() {
    for (auto &n: outputNeurons)  n->backPropagate();

    for (auto &s: outputSynapses) s->backPropagate();
    for (auto &s: hiddenSynapses) s->backPropagate();

    for (auto &n: hiddenNeurons)  n->backPropagate();

    for (auto &s: inputSynapses)  s->backPropagate();
    for (auto &n: inputNeurons)   n->backPropagate();

    for (auto &g: gradients)      g->backPropagate();
}

/* should perform updates after a back propagation step of the network */
void LongShortTermMemoryEligibilityNetwork::backPropagateAfter() {

    for (unsigned i = 0; i < inputNeurons.size(); i++) {
        inputErrorsOverTime[i + timeStepCounter * inputNeurons.size()] += inputNeurons[i]->getError();
        allInputErrorsOverTime[sampleIndex][i + timeStepCounter * inputNeurons.size()] = inputNeurons[i]->getError();
    }

    if (timeStepCounter == 0) {
        if (opts.shuffleBatch())
            sampleIndex = rand128(rand) % opts.batchSize();
        else
            sampleIndex = (sampleIndex + 1) % opts.batchSize();

        batchCounter++;
        if (batchCounter == opts.batchSize()) {

            summedErrorLast            = summedError;
            summedTargetLast           = summedTarget;
            squaredSummedTargetLast    = squaredSummedTarget;
            numErrorValuesLast         = numErrorValues;
            classificationAccuracyLast = classificationAccuracy / classificationSamples;
            summedError                = 0;
            summedTarget               = 0;
            squaredSummedTarget        = 0;
            batchCounter               = 0;
            numErrorValues             = 0;
            classificationAccuracy     = 0;
            classificationSamples      = 0;
        }
        this->resetNetwork();
    }
}



/* creats a hidden neuron */
shared_ptr<BasicNeuron> LongShortTermMemoryEligibilityNetwork::createNeuron(bool adaptive) {

    if (adaptive) {
        return static_pointer_cast<BasicNeuron>(
            make_shared<AdaptiveLeakyIntegrateAndFireNeuron>(
                opts, 
                opts.spikeThreshold(), 
                opts.thresholdIncreaseConstant(), 
                opts.adaptationTimeConstant(), 
                opts.refactoryPeriod(),
                opts.hiddenMembranTimeConstant()
            )
        );
    } else {
        return static_pointer_cast<BasicNeuron>(
            make_shared<LeakyIntegrateAndFireNeuron>(
                opts,
                opts.spikeThreshold(), 
                opts.refactoryPeriod(),
                opts.hiddenMembranTimeConstant()
            )
        );
    } 
    
    assert(false);
    return nullptr;
}

/* creats a hidden synapse */
shared_ptr<BasicSynapse> LongShortTermMemoryEligibilityNetwork::createSynapse(
    shared_ptr<BasicNeuron> input,
    shared_ptr<BasicNeuron> output,
    FloatType weight,
    bool adaptive
) {
    if (adaptive) {
        assert(dynamic_pointer_cast<AdaptiveLeakyIntegrateAndFireNeuron>(output) != nullptr);
        return static_pointer_cast<BasicSynapse>(
            make_shared<AdaptiveLeakyIntegrateAndFireSynapse>(
                *input,
                *static_pointer_cast<AdaptiveLeakyIntegrateAndFireNeuron>(output),
                weight
            )
        );
    } else {
        assert(dynamic_pointer_cast<LeakyIntegrateAndFireNeuron>(output) != nullptr);
        return static_pointer_cast<BasicSynapse>(
            make_shared<LeakyIntegrateAndFireSynapse>(
                *input,
                *static_pointer_cast<LeakyIntegrateAndFireNeuron>(output),
                weight
            )
        );
    } 
    
    assert(false);
    return nullptr;
}

/* creats a gradient for the given synapse */
std::shared_ptr<Interfaces::BasicGradient> LongShortTermMemoryEligibilityNetwork::createGradient(
    std::shared_ptr<Interfaces::BasicSynapse> synapse,
    unsigned neuronIndex,
    bool firingRateGradient,
    bool hiddenSynapse
) {
    if (opts.useEligibilityBackPropagation()) {
        if (firingRateGradient) {
            return static_pointer_cast<BasicGradient>(
                make_shared<FiringRateGradient>(*synapse, opts) 
            );
        }
        return static_pointer_cast<BasicGradient>(
            make_shared<EligibilityBackPropagatedGradient>(*synapse)
        );
    }

    if (opts.useBackPropagation()) {
        if (firingRateGradient) {
            return static_pointer_cast<BasicGradient>(
                make_shared<FastFiringRateGradient>(
                    *synapse, 
                    opts.targetFiringRate(),
                    opts.timeStepLength() / (opts.numSimulationTimesteps() * opts.batchSize())
                )
            );
        }
        return static_pointer_cast<BasicGradient>(
            make_shared<BackPropagatedGradient>(*synapse, hiddenSynapse)
        );
    }


    if (firingRateGradient) {
        return static_pointer_cast<BasicGradient>(
            make_shared<FiringRateGradient>(*synapse, opts)
        );
    }
    
    return static_pointer_cast<BasicGradient>(
        make_shared<FixedBroadcastGradient>(
            *synapse,
            feedbackMatrix, 
            opts.numOutputNeurons(), 
            neuronIndex,
            opts.readoutMembranTimeConstant(),
            opts.timeStepLength(),
            opts.errorMode() == ERROR_MODE_CLASSIFICATION
        )
    );
}

/* destructor */
LongShortTermMemoryEligibilityNetwork::~LongShortTermMemoryEligibilityNetwork() {
    for (auto &t: targetsOverTime)
        delete[] t;
    for (auto &o: outputsOverTime)
        delete[] o;
    for (auto &e: deltaErrorsOverTime)
        delete[] e;
    for (auto &e: deltaErrorInputssOverTime)
        delete[] e;
    for (auto &o: outputErrorsOverTime)
        delete[] o;
    for (auto &e: errorMaskOverTime)
        delete[] e;
    for (auto &e: outputErrorFactorOverTime)
        delete[] e;
    for (auto &i: inputsOverTime)
        delete[] i;
    for (auto &i: spikesOverTime)
        delete[] i;
    for (auto &i: derivativesOverTime)
        delete[] i;
    for (auto &i: allInputErrorsOverTime)
        delete[] i;
    delete[] inputErrorsOverTime;
    delete[] targetWeights;
}

shared_ptr<BasicOptimizer> LongShortTermMemoryEligibilityNetwork::createOptimizer(
    shared_ptr<BasicGradient> gradient,
    bool firingRateOptimizer
) {
    OptimizerType type               = opts.optimizer();
    FloatType learnRate              = opts.learnRate();
    FloatType momentum               = opts.momentum();
    FloatType learnRateDecay         = opts.learnRateDecay();
    unsigned learnRateDecayIntervall = opts.learnRateDecayIntervall();

    if (firingRateOptimizer) {
        type                    = opts.regularizerOptimizer();
        learnRate               = opts.regularizerLearnRate();
        momentum                = opts.regularizerMomentum();
        learnRateDecay          = opts.regularizerLearnRateDecay();
        learnRateDecayIntervall = opts.regularizerLearnRateDecayIntervall();
    }

    if (type == OptimizerType::StochasticGradientDescent) {
        return static_pointer_cast<BasicOptimizer>(
            make_shared<StochasticGradientDescentOptimizer>(
                *gradient, 
                opts.optimizerUpdateInterval(),
                learnRate,
                learnRateDecay,
                learnRateDecayIntervall
            )
        );
    } else if (type == OptimizerType::Adam) {
        return static_pointer_cast<BasicOptimizer>(
            make_shared<AdamOptimizer>(
                *gradient, 
                opts.optimizerUpdateInterval(),
                learnRate,
                learnRateDecay,
                learnRateDecayIntervall,
                opts.adamBetaOne(),
                opts.adamBetaTwo(),
                opts.adamEpsilon(),
                opts.adamWeightDecay()
            )
        );
    } else if (type == OptimizerType::AMSGrad) {
        return static_pointer_cast<BasicOptimizer>(
            make_shared<AMSGradOptimizer>(
                *gradient, 
                opts.optimizerUpdateInterval(),
                learnRate,
                learnRateDecay,
                learnRateDecayIntervall,
                opts.adamBetaOne(),
                opts.adamBetaTwo(),
                opts.adamEpsilon(),
                opts.adamWeightDecay()
            )
        );
    } else if (type == OptimizerType::SignDampedMomentum) {
        return static_pointer_cast<BasicOptimizer>(
            make_shared<SignDampedMomentumOptimizer>(
                *gradient, 
                opts.optimizerUpdateInterval(),
                learnRate,
                momentum,
                learnRateDecay,
                learnRateDecayIntervall
            )
        );
    }

    assert(false);
    return nullptr;
}

/* create afully connected network */
void LongShortTermMemoryEligibilityNetwork::fullyConnect(
    vector<shared_ptr<BasicGradient>> &inputFiringRateRegularizers, 
    vector<shared_ptr<BasicGradient>> &hiddenFiringRateRegularizers,
    vector<shared_ptr<BasicGradient>> &inputGradients, 
    vector<shared_ptr<BasicGradient>> &hiddenGradients,
    vector<shared_ptr<BasicGradient>> &leakyReadoutGradients
) {

    feedbackMatrix = static_pointer_cast<BasicFeedbackWeights>(
        make_shared<FeedbackWeights>(
            opts.numHiddenNeurons(), 
            opts.numOutputNeurons()
        )
    );

    /* input synapses */ 
    for (unsigned i = 0; i < opts.numInputNeurons(); i++) {
        for (unsigned h = 0; h < opts.numHiddenNeurons(); h++) {
            inputSynapses.push_back(this->createSynapse(
                inputNeurons[i], 
                hiddenNeurons[h],
                opts.inputWeightFactor() * rand128n(rand, 0.0, 1.0) / sqrt(opts.numInputNeurons()),
                !(h < opts.numStandartHiddenNeurons())
            ));
            synapses.push_back(inputSynapses.back());
        }
    }

    /* create gradients in reverse order for better gpu cache performance */
    for (unsigned h = 0; h < opts.numHiddenNeurons(); h++) {
        for (unsigned i = 0; i < opts.numInputNeurons(); i++) {
            shared_ptr<BasicSynapse> synapse = inputSynapses[i * opts.numHiddenNeurons() + h];

            gradients.push_back(createGradient(synapse, h, true));
            inputFiringRateRegularizers.push_back(gradients.back());
            optimizers.push_back(createOptimizer(gradients.back(), true));
            inputRegularizers.push_back(optimizers.back());

            gradients.push_back(createGradient(synapse, h));
            inputGradients.push_back(gradients.back());
            optimizers.push_back(createOptimizer(gradients.back()));
            inputOptimizers.push_back(optimizers.back());
        }
    }

    /* hidden synapses */
    for (unsigned hi = 0; hi < opts.numHiddenNeurons(); hi++) {
        for (unsigned ho = 0; ho < opts.numHiddenNeurons(); ho++) {
            hiddenSynapses.push_back(this->createSynapse(
                hiddenNeurons[hi], 
                hiddenNeurons[ho],
                opts.hiddenWeightFactor() * rand128n(rand, 0.0, 1.0) / sqrt(opts.numHiddenNeurons()),
                !(ho < opts.numStandartHiddenNeurons())
            ));
            synapses.push_back(hiddenSynapses.back());
        }
    }

    /* create gradients in reverse order for better gpu cache performance */
    for (unsigned ho = 0; ho < opts.numHiddenNeurons(); ho++) {
        for (unsigned hi = 0; hi < opts.numHiddenNeurons(); hi++) {
            shared_ptr<BasicSynapse> synapse = hiddenSynapses[hi * opts.numHiddenNeurons() + ho];

            gradients.push_back(createGradient(synapse, ho, true));
            hiddenFiringRateRegularizers.push_back(gradients.back());
            optimizers.push_back(createOptimizer(gradients.back(), true));
            hiddenRegularizers.push_back(optimizers.back());

            gradients.push_back(createGradient(synapse, ho, false, true));
            hiddenGradients.push_back(gradients.back());
            optimizers.push_back(createOptimizer(gradients.back()));
            hiddenOptimizers.push_back(optimizers.back());
        }
    }

    /* readout synapses */
    for (unsigned h = 0; h < opts.numHiddenNeurons(); h++) {
        for (unsigned o = 0; o < opts.numOutputNeurons(); o++) {
            outputSynapses.push_back(static_pointer_cast<BasicSynapse>(
                make_shared<LeakyReadoutSynapse>(
                    *hiddenNeurons[h],
                    *static_pointer_cast<LeakyReadoutNeuron>(outputNeurons[o]), 
                    opts.outputWeightFactor() * rand128n(rand, 0.0, 1.0) / sqrt(opts.numHiddenNeurons())
                )
            ));
            synapses.push_back(outputSynapses.back());
        }
    }

    /* create gradients in reverse order for better gpu cache performance */
    for (unsigned o = 0; o < opts.numOutputNeurons(); o++) {
        for (unsigned h = 0; h < opts.numHiddenNeurons(); h++) {
            shared_ptr<BasicSynapse> synapse = outputSynapses[h * opts.numOutputNeurons() + o];
            if (opts.useBackPropagation() && opts.errorMode() == ERROR_MODE_LEARNSIGNAL) 
                gradients.push_back(createGradient(synapse, h));
            else {
                gradients.push_back(static_pointer_cast<BasicGradient>(
                    make_shared<LeakyReadoutGradient>(
                        *static_pointer_cast<LeakyReadoutSynapse>(synapse), o, 
                        opts.errorMode() == ERROR_MODE_CLASSIFICATION
                    )
                ));
            }
            leakyReadoutGradients.push_back(gradients.back());
            optimizers.push_back(createOptimizer(gradients.back()));
            outputOptimizers.push_back(optimizers.back());
        }
    }
}

/* create a sparse connected network */
void LongShortTermMemoryEligibilityNetwork::sparseConnect(
    vector<shared_ptr<BasicGradient>> &inputFiringRateRegularizers, 
    vector<shared_ptr<BasicGradient>> &hiddenFiringRateRegularizers,
    vector<shared_ptr<BasicGradient>> &inputGradients, 
    vector<shared_ptr<BasicGradient>> &hiddenGradients,
    vector<shared_ptr<BasicGradient>> &leakyReadoutGradients,
    unsigned numAdaptiveInputSynapses,
    unsigned numAdaptiveHiddenSynapses,
    vector<FloatType> &inputWeights,
    vector<FloatType> &hiddenWeights,
    vector<FloatType> &outputWeights,
    vector<FloatType> &feedbackWeights,
    vector<unsigned> &inputWeightsIn,
    vector<unsigned> &hiddenWeightsIn,
    vector<unsigned> &outputWeightsIn,
    vector<unsigned> &feedbackWeightsIn,
    vector<unsigned> &inputWeightsOut,
    vector<unsigned> &hiddenWeightsOut,
    vector<unsigned> &outputWeightsOut,
    vector<unsigned> &feedbackWeightsOut
) {

    feedbackMatrix = static_pointer_cast<BasicFeedbackWeights>(
        make_shared<SparseFeedbackWeights>(
            feedbackWeights,
            feedbackWeightsIn,
            feedbackWeightsOut
        )
    );

    /* input synapses */ 
    for (unsigned i = 0; i < inputWeights.size(); i++) {
        inputSynapses.push_back(this->createSynapse(
            inputNeurons[inputWeightsIn[i]], 
            hiddenNeurons[inputWeightsOut[i]],
            inputWeights[i] * opts.inputWeightFactor(),
            i >= inputWeights.size() - numAdaptiveInputSynapses
        ));
        synapses.push_back(inputSynapses.back());

        gradients.push_back(createGradient(synapses.back(), inputWeightsOut[i], true));
        inputFiringRateRegularizers.push_back(gradients.back());
        optimizers.push_back(createOptimizer(gradients.back(), true));
        inputRegularizers.push_back(optimizers.back());

        gradients.push_back(createGradient(synapses.back(), inputWeightsOut[i]));
        inputGradients.push_back(gradients.back());
        optimizers.push_back(createOptimizer(gradients.back()));
        inputOptimizers.push_back(optimizers.back());
    }

    /* hidden synapses */
    for (unsigned i = 0; i < hiddenWeights.size(); i++) {
        hiddenSynapses.push_back(this->createSynapse(
            hiddenNeurons[hiddenWeightsIn[i]], 
            hiddenNeurons[hiddenWeightsOut[i]],
            hiddenWeights[i] * opts.hiddenWeightFactor(),
            i >= hiddenWeights.size() - numAdaptiveHiddenSynapses
        ));
        synapses.push_back(hiddenSynapses.back());

        gradients.push_back(createGradient(synapses.back(), hiddenWeightsOut[i], true));
        hiddenFiringRateRegularizers.push_back(gradients.back());
        optimizers.push_back(createOptimizer(gradients.back(), true));
        hiddenRegularizers.push_back(optimizers.back());

        gradients.push_back(createGradient(synapses.back(), hiddenWeightsOut[i], false, true));
        hiddenGradients.push_back(gradients.back());
        optimizers.push_back(createOptimizer(gradients.back()));
        hiddenOptimizers.push_back(optimizers.back());
    }

    /* readout synapses */
    for (unsigned i = 0; i < outputWeights.size(); i++) {
        outputSynapses.push_back(static_pointer_cast<BasicSynapse>(
            make_shared<LeakyReadoutSynapse>(
                *hiddenNeurons[outputWeightsIn[i]],
                *static_pointer_cast<LeakyReadoutNeuron>(outputNeurons[outputWeightsOut[i]]), 
                outputWeights[i] * opts.outputWeightFactor()
            )
        ));
        synapses.push_back(outputSynapses.back());

        if (opts.useBackPropagation() && !opts.useEligibilityBackPropagation())
            gradients.push_back(createGradient(synapses.back(), outputWeightsIn[i]));
        else {
            gradients.push_back(static_pointer_cast<BasicGradient>(
                make_shared<LeakyReadoutGradient>(
                    *static_pointer_cast<LeakyReadoutSynapse>(synapses.back()), 
                    outputWeightsOut[i], 
                    opts.errorMode() == ERROR_MODE_CLASSIFICATION
                )
            ));
        }
        leakyReadoutGradients.push_back(gradients.back());
        optimizers.push_back(createOptimizer(gradients.back()));
        outputOptimizers.push_back(optimizers.back());
    }
}

/* should perform aditional resets */
void LongShortTermMemoryEligibilityNetwork::doReset() {

    sampleIndex = 0;
    timeStepCounter = 0; 
    batchCounter = 0;
    epochCounter = 0;
    runtimeCounter = gettime_usec();
    summedError = 0;
    summedErrorLast = 0;
    summedTarget = 0;
    squaredSummedTarget = 0;
    summedTargetLast = 0;
    squaredSummedTargetLast = 0;
    numErrorValues = 0;
    minError = numeric_limits<FloatType>::max();
    classificationAccuracy = 0;
    classificationAccuracyLast = 0;
    classificationSamples = 0;

    for (unsigned i = 0; i < summedActivation.size(); i++)
        summedActivation[i] = 0;
    
    *rand = initialRandomState;

    for (auto &s: serializer)
        s->serialize();
}

/* constructor */
LongShortTermMemoryEligibilityNetwork::LongShortTermMemoryEligibilityNetwork(
    LongShortTermMemoryEligibilityNetworkOptions &opts,
    unsigned numAdaptiveInputSynapses,
    unsigned numAdaptiveHiddenSynapses,
    vector<FloatType> inputWeights,
    vector<FloatType> hiddenWeights,
    vector<FloatType> outputWeights,
    vector<FloatType> feedbackWeights,
    vector<unsigned> inputWeightsIn,
    vector<unsigned> hiddenWeightsIn,
    vector<unsigned> outputWeightsIn,
    vector<unsigned> feedbackWeightsIn,
    vector<unsigned> inputWeightsOut,
    vector<unsigned> hiddenWeightsOut,
    vector<unsigned> outputWeightsOut,
    vector<unsigned> feedbackWeightsOut
) :
    gpuKernel(nullptr),
    sparseGpuKernel(nullptr),
    opts(static_cast<LongShortTermMemoryEligibilityNetworkOptions &>(opts.freeze())),
    inputWeightsIn(inputWeightsIn),
    hiddenWeightsIn(hiddenWeightsIn),
    outputWeightsIn(outputWeightsIn),
    feedbackWeightsIn(feedbackWeightsIn),
    inputWeightsOut(inputWeightsOut),
    hiddenWeightsOut(hiddenWeightsOut),
    outputWeightsOut(outputWeightsOut),
    feedbackWeightsOut(feedbackWeightsOut),
    sampleIndex(0),
    timeStepCounter(0), 
    batchCounter(0),
    epochCounter(0),
    runtimeCounter(gettime_usec()),
    rand(new_rand128_t()),
    summedError(0),
    summedErrorLast(0),
    summedTarget(0),
    squaredSummedTarget(0),
    summedTargetLast(0),
    squaredSummedTargetLast(0),
    numErrorValues(0),
    minError(numeric_limits<FloatType>::max()),
    classificationAccuracy(0),
    classificationAccuracyLast(0),
    classificationSamples(0),
    summedActivation(opts.numOutputNeurons(), 0) {

    init(
        opts,
        numAdaptiveInputSynapses,
        numAdaptiveHiddenSynapses,
        inputWeights,
        hiddenWeights,
        outputWeights,
        feedbackWeights,
        inputWeightsIn,
        hiddenWeightsIn,
        outputWeightsIn,
        feedbackWeightsIn,
        inputWeightsOut,
        hiddenWeightsOut,
        outputWeightsOut,
        feedbackWeightsOut
    );
}

/* initializes this */
void LongShortTermMemoryEligibilityNetwork::init(
    LongShortTermMemoryEligibilityNetworkOptions &opts,
    unsigned numAdaptiveInputSynapses,
    unsigned numAdaptiveHiddenSynapses,
    vector<FloatType> inputWeights,
    vector<FloatType> hiddenWeights,
    vector<FloatType> outputWeights,
    vector<FloatType> feedbackWeights,
    vector<unsigned> inputWeightsIn,
    vector<unsigned> hiddenWeightsIn,
    vector<unsigned> outputWeightsIn,
    vector<unsigned> feedbackWeightsIn,
    vector<unsigned> inputWeightsOut,
    vector<unsigned> hiddenWeightsOut,
    vector<unsigned> outputWeightsOut,
    vector<unsigned> feedbackWeightsOut
) {
    
    targetMagnitudes = vector<FloatType>(opts.numOutputNeurons(), 0); 
    targetMagnitudeSum = 0;

    bool sparseNetwork = (inputWeights.size() > 0) ||
                         (hiddenWeights.size() > 0) ||
                         (outputWeights.size() > 0) ||
                         (feedbackWeights.size() > 0) ||
                         (numAdaptiveInputSynapses != unsigned(-1)) ||
                         (numAdaptiveHiddenSynapses != unsigned(-1));

    std::vector<std::shared_ptr<Interfaces::BasicGradient>> inputFiringRateRegularizers, 
                                      hiddenFiringRateRegularizers, 
                                      inputGradients, 
                                      hiddenGradients,
                                      leakyReadoutGradients;

    shared_ptr<BasicGradientSerializer> inputFiringRateRegularizerSerializer, 
                                         hiddenFiringRateRegularizerSerializer,
                                         inputGradientSerializer, 
                                         hiddenGradientSerializer,
                                         leakyReadoutGradientSerializer;


    for (unsigned i = 0; i < opts.numInputNeurons(); i++) {
        inputNeurons.push_back(static_pointer_cast<BasicNeuron>(
            make_shared<CurrentInputNeuron>(opts)
        ));
        neurons.push_back(inputNeurons.back());
    }
    for (unsigned i = 0; i < opts.numStandartHiddenNeurons(); i++) {
        hiddenNeurons.push_back(this->createNeuron(false));
        neurons.push_back(hiddenNeurons.back());
    }
    for (unsigned i = 0; i < opts.numAdaptiveHiddenNeurons(); i++) {
        hiddenNeurons.push_back(this->createNeuron(true));
        neurons.push_back(hiddenNeurons.back());
    }
    for (unsigned i = 0; i < opts.numOutputNeurons(); i++) {
        outputNeurons.push_back(static_pointer_cast<BasicNeuron>(
            make_shared<LeakyReadoutNeuron>(
                opts,
                0,
                opts.readoutMembranTimeConstant()
            )
        ));
        neurons.push_back(outputNeurons.back());
    }

    if (sparseNetwork) {
        this->sparseConnect(
            inputFiringRateRegularizers, 
            hiddenFiringRateRegularizers,
            inputGradients, 
            hiddenGradients,
            leakyReadoutGradients,
            numAdaptiveInputSynapses,
            numAdaptiveHiddenSynapses,
            inputWeights,
            hiddenWeights,
            outputWeights,
            feedbackWeights,
            inputWeightsIn,
            hiddenWeightsIn,
            outputWeightsIn,
            feedbackWeightsIn,
            inputWeightsOut,
            hiddenWeightsOut,
            outputWeightsOut,
            feedbackWeightsOut
        );
    } else {
        this->fullyConnect(
            inputFiringRateRegularizers, 
            hiddenFiringRateRegularizers,
            inputGradients, 
            hiddenGradients,
            leakyReadoutGradients
        );
    }

    for (auto &gradient: gradients)
        gradient->registerNetwork(this);

    neuronsSerializer = make_shared<BasicNeuronSerializer>(neurons);

    inputSynapseSerializer  = make_shared<BasicSynapseSerializer>(inputSynapses);
    hiddenSynapseSerializer = make_shared<BasicSynapseSerializer>(hiddenSynapses);
    outputSynapseSerializer = make_shared<BasicSynapseSerializer>(outputSynapses, false);

    inputFiringRateRegularizerSerializer  = make_shared<BasicGradientSerializer>(inputFiringRateRegularizers, "input-firing-rate");
    hiddenFiringRateRegularizerSerializer = make_shared<BasicGradientSerializer>(hiddenFiringRateRegularizers, "hidden-firing-rate" );
    inputGradientSerializer               = make_shared<BasicGradientSerializer>(inputGradients, "input");
    hiddenGradientSerializer              = make_shared<BasicGradientSerializer>(hiddenGradients, "hidden");
    leakyReadoutGradientSerializer        = make_shared<BasicGradientSerializer>(leakyReadoutGradients, "readout");
    feedbackMatrixSerializer              = make_shared<FeedbackWeightsSerializer>(feedbackMatrix);

    serializer.push_back(static_pointer_cast<BasicSerializer>(neuronsSerializer));
    serializer.push_back(static_pointer_cast<BasicSerializer>(inputSynapseSerializer));
    serializer.push_back(static_pointer_cast<BasicSerializer>(hiddenSynapseSerializer));
    serializer.push_back(static_pointer_cast<BasicSerializer>(outputSynapseSerializer));
    serializer.push_back(static_pointer_cast<BasicSerializer>(inputFiringRateRegularizerSerializer));
    serializer.push_back(static_pointer_cast<BasicSerializer>(hiddenFiringRateRegularizerSerializer));
    serializer.push_back(static_pointer_cast<BasicSerializer>(inputGradientSerializer));
    serializer.push_back(static_pointer_cast<BasicSerializer>(hiddenGradientSerializer));
    serializer.push_back(static_pointer_cast<BasicSerializer>(leakyReadoutGradientSerializer));
    serializer.push_back(static_pointer_cast<BasicSerializer>(feedbackMatrixSerializer));

    hiddenFiringRates = neuronsSerializer->getFiringRates() + inputNeurons.size();

    for (unsigned i = 0; i < opts.batchSize(); i++) {
        inputsOverTime.push_back(
            new FloatType[inputNeurons.size() * opts.numSimulationTimesteps()]
        );
        allInputErrorsOverTime.push_back(
            new FloatType[inputNeurons.size() * opts.numSimulationTimesteps()]
        );
        spikesOverTime.push_back(
            new FloatType[(inputNeurons.size() + hiddenNeurons.size()) * opts.numSimulationTimesteps()]
        );
        derivativesOverTime.push_back(
            new FloatType[hiddenNeurons.size() * opts.numSimulationTimesteps()]
        );
        targetsOverTime.push_back(
            new FloatType[outputNeurons.size() * opts.numSimulationTimesteps()]
        );
        outputsOverTime.push_back(
            new FloatType[outputNeurons.size() * opts.numSimulationTimesteps()]
        );
        deltaErrorsOverTime.push_back(
            new FloatType[(opts.numHiddenNeurons() + opts.numAdaptiveHiddenNeurons() + opts.numOutputNeurons()) * opts.numSimulationTimesteps()]
        );
        deltaErrorInputssOverTime.push_back(
            new FloatType[(opts.numHiddenNeurons() + opts.numAdaptiveHiddenNeurons() + opts.numOutputNeurons()) * opts.numSimulationTimesteps()]
        );
        outputErrorsOverTime.push_back(
            new FloatType[outputNeurons.size() * opts.numSimulationTimesteps()]
        );
        errorMaskOverTime.push_back(
            new FloatType[opts.numSimulationTimesteps()]
        );
        outputErrorFactorOverTime.push_back(
            new FloatType[opts.numOutputNeurons() * opts.numSimulationTimesteps()]
        );
        for (unsigned n = 0; n < inputNeurons.size() * opts.numSimulationTimesteps(); n++) {
            inputsOverTime.back()[n] = 0;
            allInputErrorsOverTime.back()[n] = 0;
        }
        for (unsigned n = 0; n < (inputNeurons.size() + hiddenNeurons.size()) * opts.numSimulationTimesteps(); n++)
            spikesOverTime.back()[n] = 0;
        for (unsigned n = 0; n < outputNeurons.size() * opts.numSimulationTimesteps(); n++) {
            targetsOverTime.back()[n] = 0;
            outputsOverTime.back()[n] = 0;
            outputErrorsOverTime.back()[n] = 0;
        }
        for (unsigned n = 0; n < (opts.numHiddenNeurons() + opts.numAdaptiveHiddenNeurons() + opts.numOutputNeurons()) * opts.numSimulationTimesteps(); n++) {
            deltaErrorsOverTime.back()[n] = 0;
            deltaErrorInputssOverTime.back()[n] = 0;
        }
        for (unsigned n = 0; n < opts.numSimulationTimesteps(); n++) 
            errorMaskOverTime.back()[n] = 0;
        for (unsigned n = 0; n < opts.numOutputNeurons() * opts.numSimulationTimesteps(); n++) 
            outputErrorFactorOverTime.back()[n] = 3.141;
    }
    inputErrorsOverTime = new FloatType[opts.numInputNeurons() * opts.numSimulationTimesteps()];
    targetWeights = new FloatType[opts.numOutputNeurons()];
    for (unsigned i = 0; i < opts.numOutputNeurons(); i++)
        targetWeights[i] = 1;

    for (auto &s: serializer)
        s->serialize();

    if (sparseNetwork) {
        sparseGpuKernel = make_shared<LongShortTermMemorySparseKernelCaller>(
            opts.batchSize(),
            inputNeurons.size(),
            hiddenNeurons.size(),
            opts.numStandartHiddenNeurons(),
            opts.numAdaptiveHiddenNeurons(),
            outputNeurons.size(),
            opts.numSimulationTimesteps(),
            opts.timeStepLength(),
            opts.spikeThreshold(),
            int(round(opts.refactoryPeriod() / opts.timeStepLength())),
            exp(-1.0 * opts.timeStepLength() / opts.hiddenMembranTimeConstant()),
            exp(-1.0 * opts.timeStepLength() / opts.readoutMembranTimeConstant()),
            exp(-1.0 * opts.timeStepLength() / opts.adaptationTimeConstant()),
            opts.thresholdIncreaseConstant(),
            opts.targetFiringRate(),
            opts.timeStepLength() / (opts.numSimulationTimesteps() * opts.batchSize()),
            opts.derivativeDumpingFactor(),
            inputsOverTime,
            spikesOverTime,
            neuronsSerializer->getFiringRates() + inputNeurons.size(),
            inputWeights.size() - numAdaptiveInputSynapses,
            hiddenWeights.size() - numAdaptiveHiddenSynapses,
            numAdaptiveInputSynapses,
            numAdaptiveHiddenSynapses,
            inputWeights.size(),
            hiddenWeights.size(), 
            outputSynapses.size(),
            feedbackWeights.size(),
            inputSynapseSerializer->getWeights(),
            hiddenSynapseSerializer->getWeights(),
            outputSynapseSerializer->getWeights(),
            feedbackMatrixSerializer->getWeights(),
            this->inputWeightsIn.data(),
            this->hiddenWeightsIn.data(),
            this->outputWeightsIn.data(),
            this->feedbackWeightsIn.data(),
            this->inputWeightsOut.data(),
            this->hiddenWeightsOut.data(),
            this->outputWeightsOut.data(),
            this->feedbackWeightsOut.data(),
            targetWeights,
            targetsOverTime,
            outputsOverTime,
            errorMaskOverTime,
            outputErrorFactorOverTime,
            inputGradientSerializer->getGradients(),
            inputFiringRateRegularizerSerializer->getGradients(),
            hiddenGradientSerializer->getGradients(),
            hiddenFiringRateRegularizerSerializer->getGradients(),
            leakyReadoutGradientSerializer->getGradients(),
            inputErrorsOverTime,
            outputErrorsOverTime,
            allInputErrorsOverTime,
            deltaErrorsOverTime
        );
        if (!opts.training()) {
            sparseGpuKernel->setForwardPass();
            log_str("Sparse Kernel Mode: Forward Pass", LOG_I);
        } else {
            if (opts.useEligibilityBackPropagation()) {
                sparseGpuKernel->setEligibilityBackPropagationTraining();
                log_str("Sparse Kernel Mode: e-prop 3", LOG_I);
            } else if (opts.useBackPropagation()) {
                sparseGpuKernel->setBackPropagationTraining();
                log_str("Sparse Kernel Mode: Back-Propagation", LOG_I);
            } else {
                sparseGpuKernel->setEligibilityTraining();
                if (opts.symetricEprop1()) {
                    sparseGpuKernel->setFlag(LSNN_SYMETRIC_EPROP);
                    log_str("Sparse Kernel Mode: symetric e-prop 1", LOG_I);
                } else
                    log_str("Sparse Kernel Mode: e-prop 1", LOG_I);
            }
            if (opts.useInputErrors()) {
                sparseGpuKernel->setFlag(LSNN_INPUT_ERRORS);
                log_str("Sparse Kernel Mode: Input Errors", LOG_I);
            }
        }
        if (opts.errorMode() == ERROR_MODE_CLASSIFICATION) {
            sparseGpuKernel->unsetsetFlag(LSNN_REGRESSION_ERROR);
            sparseGpuKernel->setFlag(LSNN_CLASSIFICATION_ERROR);
            log_str("Sparse Kernel Mode: Classification", LOG_I);
        }
    } else {
        gpuKernel = make_shared<LongShortTermMemoryKernelCaller>(
            opts.batchSize(),
            inputNeurons.size(),
            hiddenNeurons.size(),
            opts.numStandartHiddenNeurons(),
            opts.numAdaptiveHiddenNeurons(),
            outputNeurons.size(),
            opts.numSimulationTimesteps(),
            opts.errorMode(),
            opts.timeStepLength(),
            opts.spikeThreshold(),
            int(round(opts.refactoryPeriod() / opts.timeStepLength())),
            exp(-1.0 * opts.timeStepLength() / opts.hiddenMembranTimeConstant()),
            exp(-1.0 * opts.timeStepLength() / opts.readoutMembranTimeConstant()),
            exp(-1.0 * opts.timeStepLength() / opts.adaptationTimeConstant()),
            opts.thresholdIncreaseConstant(),
            opts.targetFiringRate(),
            opts.timeStepLength() / (opts.numSimulationTimesteps() * opts.batchSize()),
            opts.derivativeDumpingFactor(),
            inputsOverTime,
            spikesOverTime,
            neuronsSerializer->getFiringRates() + inputNeurons.size(),
            inputSynapseSerializer->getWeights(),
            hiddenSynapseSerializer->getWeights(),
            outputSynapseSerializer->getWeights(),
            feedbackMatrixSerializer->getWeights(),
            targetWeights,
            targetsOverTime,
            outputsOverTime,
            outputErrorsOverTime,
            errorMaskOverTime,
            outputErrorFactorOverTime,
            inputGradientSerializer->getGradients(),
            inputFiringRateRegularizerSerializer->getGradients(),
            hiddenGradientSerializer->getGradients(),
            hiddenFiringRateRegularizerSerializer->getGradients(),
            leakyReadoutGradientSerializer->getGradients(),
            inputErrorsOverTime,
            allInputErrorsOverTime
        );
    }

    initialRandomState = *rand;
}

/* run one simulation with GPU optimization */
void LongShortTermMemoryEligibilityNetwork::updateGPU(
    bool batchSetted,
    int backPropagationMode,
    int errorMode,
    int startTime,
    int endTime
) {

    if (startTime < 0)
        startTime = 0;
    if (endTime < 0)
        endTime = opts.numSimulationTimesteps();

    if (backPropagationMode < 0)
        backPropagationMode = opts.useBackPropagation();

    if (opts.symetricEprop1() && gpuKernel != nullptr) {
        for (unsigned h = 0; h < opts.numHiddenNeurons(); h++) {
            for (unsigned o = 0; o < opts.numOutputNeurons(); o++) {
                feedbackMatrix->setWeight(
                    outputSynapseSerializer->getWeights()[h * opts.numOutputNeurons() + o],
                    h, 
                    o
                );
            }
        }
        feedbackMatrixSerializer->serialize();
        gpuKernel->reload();
    }

    /* reset firing rates */
    for (unsigned n = 0; n < hiddenNeurons.size(); n++) 
        hiddenNeurons[n]->setFiringRate(hiddenFiringRates[n]);

    for (auto &s: serializer)
        s->serialize(true);

    vector<unsigned> batchIndices(opts.trainSetSize(), 0);   
    std::iota(batchIndices.begin(), batchIndices.end(), 0);

    vector<shared_ptr<SmartSample>> smartSamples;
    if (smartTrainset != nullptr && (errorMode & DEBUG_FORWARD_PASS)) {
        for (unsigned b = 0; b < opts.batchSize(); b++) {
            smartSamples.push_back(smartTrainset->getNextSample());
            smartSamples.back()->copy(targetsOverTime[b], errorMaskOverTime[b], inputsOverTime[b]);
        }
    } 

    if (sparseGpuKernel != nullptr) {
        sparseGpuKernel->runAndWait();

        summedErrorLast            = sparseGpuKernel->getSquaredSummedError();
        summedTargetLast           = sparseGpuKernel->getSummedTarget();
        squaredSummedTargetLast    = sparseGpuKernel->getSquaredSummedTarget();
        numErrorValuesLast         = sparseGpuKernel->getNumSummedValues();
        classificationAccuracyLast = sparseGpuKernel->getAccuracy();

        for (unsigned i = 0; i < smartSamples.size(); i++) {
            smartSamples[i]->updateError(sparseGpuKernel->getSampleSquaredSummedError(i));
            smartTrainset->readd(smartSamples[i]);
        }
    } else {
        gpuKernel->runAndWait(backPropagationMode, opts.useInputErrors(), startTime, endTime);

        summedErrorLast            = gpuKernel->getSquaredSummedError();
        summedTargetLast           = gpuKernel->getSummedTarget();
        squaredSummedTargetLast    = gpuKernel->getSquaredSummedTarget();
        numErrorValuesLast         = gpuKernel->getNumSummedValues();
        classificationAccuracyLast = gpuKernel->getAccuracy();

        for (unsigned i = 0; i < smartSamples.size(); i++) {
            smartSamples[i]->updateError(gpuKernel->getSampleSquaredSummedError(i));
            smartTrainset->readd(smartSamples[i]);
        }
    }

    summedError                = 0;
    summedTarget               = 0;
    squaredSummedTarget        = 0;
    numErrorValues             = 0;
    classificationAccuracy     = 0;
    classificationSamples      = 0;

    bool runTest = (rand_range_d128(rand, 0.0, 1.0) < opts.debugPropability()) ||
                   ((errorMode & DEBUG_CALCULATE_DERIVATIVES) && opts.debugPropability() > 0);
    bool saveModdel = (gettime_usec() - runtimeCounter > opts.saveInterval() * 1000000) ||
                      (epochCounter > opts.saveStartEpoch() && minError > this->getError());

    if (minError > this->getError())
        minError = this->getError();

    if (runTest) {
        if ((errorMode & DEBUG_FORWARD_PASS) && (errorMode & DEBUG_BACKWARD_PASS) && opts.useInputErrors())
            errorMode |= DEBUG_BPTT_INPUT_ERROR;

        if ((errorMode & (DEBUG_CALCULATE_DERIVATIVES | 
                          DEBUG_FORWARD_PASS          | 
                          DEBUG_BACKWARD_PASS         | 
                          DEBUG_BPTT_INPUT_ERROR      | 
                          DEBUG_LEARNSIGNAL))) {
         
            runCPU(batchIndices, batchSetted, !runTest, errorMode);
        }
            
        if (errorMode & DEBUG_GRADIENTS)
            checkGradients();
    }

    for (auto &s: serializer)
        s->deserialize(true);

    if (opts.training())
        for (auto &o: optimizers) o->updateGPU();

    for (auto &g: gradients) g->reset();

    /* save the moddel to disk */
    if (saveModdel && opts.training()) {
        std::string filename = "LSNN-timestamp" + itoa(gettime_usec() / 1000000) + 
                               "-epoch" + itoa(epochCounter) + 
                               "-error" + ftoa(this->getError(), 5) + ".snn";

        this->save(filename);
    }
        
    unsigned numActiveSynapses = this->getNumActiveSynases();
    if (opts.pruneStart() > 0 && unsigned(opts.pruneStart()) <= epochCounter && epochCounter % opts.pruneIntervall() == 0) {
        inputSynapseSerializer->prune(
            opts.pruneStrength(), opts.numInputNeurons(), opts.numHiddenNeurons()
        );
        hiddenSynapseSerializer->prune(
            opts.pruneStrength(), opts.numHiddenNeurons(), opts.numHiddenNeurons()
        );
        outputSynapseSerializer->prune(
            opts.pruneStrength(), opts.numHiddenNeurons(), opts.numOutputNeurons()
        );
        this->makeSparse();
        if (numActiveSynapses == this->getNumActiveSynases())
            exit(EXIT_SUCCESS);
    }

    epochCounter++;
}

/* saves this to the given file */
void LongShortTermMemoryEligibilityNetwork::saveSpikes(int fd, int startTime, int endTime) {
    if (startTime < 0)
        startTime = 0;
    if (endTime < 0)
        endTime = opts.numSimulationTimesteps();

    for (int t = startTime; t < endTime; t++) {
        std::vector<int> spikes;
        for (unsigned n = 0; n < opts.numHiddenNeurons(); n++)
            if (spikesOverTime[0][t * (opts.numHiddenNeurons() + opts.numInputNeurons()) + opts.numInputNeurons() + n] > 0.5)
                spikes.push_back(n);

        int tmp = spikes.size();
        assert(write(fd, &tmp, sizeof(int)) == sizeof(int));
        for (int i: spikes) 
            assert(write(fd, &i, sizeof(int)) == sizeof(int));
    }
}

/* saves this to the given file */
void LongShortTermMemoryEligibilityNetwork::save(std::string filename) {
    int fd = open(filename.c_str(), O_CREAT|O_TRUNC|O_WRONLY, 00777);
    if (fd > 0) {
        for (auto &s: serializer)
            s->save(fd);

        for (auto &o: optimizers)
            o->save(fd);

        close(fd);
        log_str("Moddel saved to: " + filename);
        runtimeCounter = gettime_usec();
    } else 
        log_err("failed to save moddel", LOG_W);
}

/* loads this from the given file */
void LongShortTermMemoryEligibilityNetwork::load(
    std::string filename, 
    bool loadOptimizer,
    bool loadLearnRate
) {
    
    int fd = open(filename.c_str(), O_RDONLY);
    if (fd > 0) {
        for (auto &s: serializer)
            s->load(fd);

        gpuKernel->reload();

        if (loadOptimizer) 
            for (auto &o: optimizers)
                o->load(fd, loadLearnRate);


        close(fd);
        log_str("loaded: " + filename);
    } else
        log_err("failed to load moddel", LOG_W);
}

/* reload feedback weights and other "not changing" values into device */
void LongShortTermMemoryEligibilityNetwork::reloadGPU() {
    if (gpuKernel != nullptr)
        gpuKernel->reload();
    if (sparseGpuKernel != nullptr)
        sparseGpuKernel->reload();
}

/* returns the squared summed gpu error for the given output */
FloatType LongShortTermMemoryEligibilityNetwork::getGPUOutputSquaredSummedError(unsigned index) {
    return sparseGpuKernel->getOutputSquaredSummedError(index);
}

/* returns the summed gpu target for the given output */
FloatType LongShortTermMemoryEligibilityNetwork::getGPUOutputSummedTarget(unsigned index) {
    return sparseGpuKernel->getOutputSummedTarget(index);
}

/* returns the squared summed gpu target for the given output */
FloatType LongShortTermMemoryEligibilityNetwork::getGPUOutputSquaredSummedTarget(unsigned index) {
    return sparseGpuKernel->getOutputSquaredSummedTarget(index);
}

/* returns the number of summed values for the given gpu output */
FloatType LongShortTermMemoryEligibilityNetwork::getGPUOutputNumSummedValues(unsigned index) {
    return sparseGpuKernel->getOutputNumSummedValues(index);
}
/* returns the squared summed gpu error for the given output */
FloatType LongShortTermMemoryEligibilityNetwork::getGPUSquaredSummedError() {
    return gpuKernel->getSquaredSummedError();
}

/* returns the summed gpu target for the given output */
FloatType LongShortTermMemoryEligibilityNetwork::getGPUSummedTarget() {
    return gpuKernel->getSummedTarget();
}

/* returns the squared summed gpu target for the given output */
FloatType LongShortTermMemoryEligibilityNetwork::getGPUSquaredSummedTarget() {
    return gpuKernel->getSquaredSummedTarget();
}

/* returns the number of summed values for the given gpu output */
FloatType LongShortTermMemoryEligibilityNetwork::getGPUNumSummedValues() {
    return gpuKernel->getNumSummedValues();
}

/* returns the networks root mean squared error derivation */
FloatType LongShortTermMemoryEligibilityNetwork::getRootMeanSquaredErrorDerivation() {
    FloatType rmse = sqrt(summedErrorLast / numErrorValuesLast);
    FloatType std  = 0;
    for (unsigned i = 0; i < opts.batchSize(); i++) {
        std += pow(rmse - sqrt(gpuKernel->getSampleSquaredSummedError(i) / gpuKernel->getSampleNumSummedValues(i)), 2);
    }
    return sqrt(std / FloatType(opts.batchSize() - 1));
}

/* runs the network in cpu mode */
void LongShortTermMemoryEligibilityNetwork::runCPU(
    vector<unsigned> batchIndices,
    bool , 
    bool singleBatch, 
    int errorMode
) {

    unsigned numUpdates = opts.numSimulationTimesteps() * opts.batchSize();
    if (opts.useBackPropagation()) numUpdates *= 2;

    FloatType *inputErrorsOverTimeCPU = new FloatType[opts.numInputNeurons() * opts.numSimulationTimesteps()];
    memset(inputErrorsOverTimeCPU, 0, sizeof(FloatType) * inputNeurons.size() * opts.numSimulationTimesteps());
    
    vector<FloatType *> allInputErrorsOverTimeCPU;
    for (unsigned b = 0; b < opts.batchSize(); b++) {
        allInputErrorsOverTimeCPU.push_back(
            new FloatType[inputNeurons.size() * opts.numSimulationTimesteps()]
        );
        memset(allInputErrorsOverTimeCPU.back(), 0, sizeof(FloatType) * inputNeurons.size() * opts.numSimulationTimesteps());
    }

    unsigned updateCounter = 0;
    for (unsigned b = 0; b < opts.batchSize(); b++) {

        sampleIndex = batchIndices[b];

        if (errorMode & DEBUG_FORWARD_PASS) {
            assert(opts.shuffleBatch() == false || opts.batchSize() == 1);
            for (unsigned t = 0; t < opts.numSimulationTimesteps(); t++, updateCounter++) {
                timeStepCounter = t;

                for (unsigned i = 0; i < opts.numInputNeurons(); i++)
                    inputNeurons[i]->addCurrent(inputAt(sampleIndex, t, i));
                
                this->doUpdate();

                if (errorMode & DEBUG_CALCULATE_DERIVATIVES)
                    for (unsigned h = 0; h < hiddenNeurons.size(); h++) 
                        derivativesOverTime[b][t * hiddenNeurons.size() + h] = hiddenNeurons[h]->getDerivative();

                for (unsigned o = 0; o < outputNeurons.size(); o++) {
                    if (fabs(outputsOverTime[sampleIndex][t * outputNeurons.size() + o] - outputNeurons[o]->getOutput()) > 0.0001 || std::isnan(outputNeurons[o]->getOutput())) {
                        log_err("outputs differ: " + 
                                ftoa(outputsOverTime[sampleIndex][t * outputNeurons.size() + o], 18) + 
                                " != " + ftoa(outputNeurons[o]->getOutput(), 18) + " => " + 
                                ftoa(fabs(outputsOverTime[sampleIndex][t * outputNeurons.size() + o] - outputNeurons[o]->getOutput()), 18), LOG_W);
                    }

                    outputsOverTime[sampleIndex][timeStepCounter * outputNeurons.size() + o] = outputNeurons[o]->getOutput();
                }

                fprintf(stderr, "testing / logging: %.3f %%   \r", 100.0 * updateCounter / numUpdates);
            }
        }

        if (errorMode & DEBUG_BACKWARD_PASS && opts.useBackPropagation()) {
            timeStepCounter = opts.numSimulationTimesteps();

            for (int t = opts.numSimulationTimesteps() - 1; t >= 0; t--, updateCounter++) {
                timeStepCounter--;


                if (opts.useEligibilityBackPropagation()) {
                    unsigned nDeltaErrors = opts.numHiddenNeurons() + 
                                            opts.numAdaptiveHiddenNeurons() + 
                                            opts.numOutputNeurons();

                    for (unsigned n = 0; n < opts.numStandartHiddenNeurons(); n++) {
                        assert(dynamic_pointer_cast<LeakyIntegrateAndFireNeuron>(hiddenNeurons[n]) != nullptr);
                        
                        if (deltaErrorInputssOverTime[sampleIndex][t * nDeltaErrors + n] != 0) {
                            hiddenNeurons[n]->setDeltaError(
                                {deltaErrorInputssOverTime[sampleIndex][t * nDeltaErrors + n]}
                            );
                        }
                    }
                    for (unsigned n = opts.numStandartHiddenNeurons(); n < opts.numHiddenNeurons(); n++) {
                        assert(dynamic_pointer_cast<AdaptiveLeakyIntegrateAndFireNeuron>(hiddenNeurons[n]) != nullptr);
                        
                        int index1 = t * nDeltaErrors + n;
                        int index2 = index1 + opts.numAdaptiveHiddenNeurons();

                        if (deltaErrorInputssOverTime[sampleIndex][index1] != 0 &&
                            deltaErrorInputssOverTime[sampleIndex][index2] != 0) {

                            hiddenNeurons[n]->setDeltaError({
                                deltaErrorInputssOverTime[sampleIndex][index1],
                                deltaErrorInputssOverTime[sampleIndex][index2]
                            });
                        }
                    }
                    for (unsigned n = 0; n < opts.numOutputNeurons(); n++) {
                        assert(dynamic_pointer_cast<LeakyReadoutNeuron>(outputNeurons[n]) != nullptr);
                        
                        int index = t * nDeltaErrors + opts.numHiddenNeurons() + 
                                    opts.numAdaptiveHiddenNeurons() + n;

                        if (deltaErrorInputssOverTime[sampleIndex][index] != 0) {
                            outputNeurons[n]->setDeltaError(
                                {deltaErrorInputssOverTime[sampleIndex][index]}
                            );
                        }
                    }
                }
                for (unsigned i = 0; i < opts.numOutputNeurons(); i++) {
                    if (opts.errorMode() == ERROR_MODE_LEARNSIGNAL) {
                        outputNeurons[i]->addError(
                            outputErrorsOverTime[sampleIndex][timeStepCounter * outputNeurons.size() + i]
                        );
                    } else if (errorMaskOverTime[sampleIndex][timeStepCounter] != 0) {
                        outputNeurons[i]->addError(
                            (
                                outputsOverTime[sampleIndex][timeStepCounter * outputNeurons.size() + i] -
                                targetsOverTime[sampleIndex][timeStepCounter * outputNeurons.size() + i]
                            ) * targetWeights[i] * this->outputErrorFactorAt(sampleIndex, timeStepCounter, i)
                        );
                    }
                }

                this->doBackPropagation();

                for (unsigned i = 0; i < inputNeurons.size(); i++) {
                    inputErrorsOverTimeCPU[i + t * inputNeurons.size()] += inputNeurons[i]->getError();
                    allInputErrorsOverTimeCPU[b][i + t * inputNeurons.size()] += inputNeurons[i]->getError();
                }

                fprintf(stderr, "testing / logging: %.3f %%   \r", 100.0 * updateCounter / numUpdates);
            }
        }
        
        this->resetNetwork();
        if (singleBatch) break;
    }

    if (errorMode & DEBUG_BPTT_INPUT_ERROR) {
        for (unsigned i = 0; i < inputNeurons.size() * opts.numSimulationTimesteps(); i++) {
            if (fabs(this->inputErrorsOverTime[i] - inputErrorsOverTimeCPU[i]) > 0.0001) {
                log_err("Input Error check failed: " + 
                        ftoa(this->inputErrorsOverTime[i], 18) + " != " + 
                        ftoa(inputErrorsOverTimeCPU[i], 18) + " => " + 
                        ftoa(fabs(this->inputErrorsOverTime[i] - inputErrorsOverTimeCPU[i]), 18), LOG_W);
            }
        }
    }

    delete[] inputErrorsOverTimeCPU;
    for (unsigned b = 0; b < opts.batchSize(); b++) 
        delete[] allInputErrorsOverTimeCPU[b];
}

/* checks te gradients of this */
void LongShortTermMemoryEligibilityNetwork::checkGradients() {

    for (auto &s: serializer)
        s->check(true);
}

/* saves the current batch inputs and outputs as csv */
void LongShortTermMemoryEligibilityNetwork::saveBatch(std::string prefix, bool debug, bool inputErrors) {
    
    for (unsigned b = 0; b < opts.batchSize(); b++) {
        FILE *file = fopen(std::string(prefix + "-batch" + itoa(b) + ".csv").c_str(), "w");
        if (file != NULL) {
            if (debug) {
                for (unsigned i = 0; i < opts.numHiddenNeurons(); i++) 
                    fprintf(file, "hidden %u,", i);

                for (unsigned i = 0; i < opts.numHiddenNeurons(); i++) 
                    fprintf(file, "hidden error %u,", i);

                for (unsigned i = 0; i < opts.numAdaptiveHiddenNeurons(); i++) 
                    fprintf(file, "hidden adaptive error %u,", i);

                for (unsigned o = 0; o < opts.numOutputNeurons(); o++) 
                    fprintf(file, "output error %u,", o);
            }

            if (inputErrors) {
                for (unsigned i = 0; i < opts.numInputNeurons(); i++) 
                    fprintf(file, "input error %u,", i);
            }

            for (unsigned i = 0; i < opts.numInputNeurons(); i++) 
                fprintf(file, "input %u,", i);

            for (unsigned o = 0; o < opts.numOutputNeurons(); o++) 
                fprintf(file, "output %u,", o);

            for (unsigned o = 0; o < opts.numOutputNeurons(); o++) {
                if (o == opts.numOutputNeurons() - 1)
                    fprintf(file, "target %u\n", o);
                else
                    fprintf(file, "target %u,", o);
            }
            FloatType *outputsOverTime = this->getGPUOutput(b);
            FloatType *targetsOverTime = this->getGPUTarget(b);
            FloatType *inputsOverTime  = this->getGPUInputs(b);
            FloatType *inputErrorsOverTime = this->getGPUInputErrors(b);

            for (unsigned t = 0; t < opts.numSimulationTimesteps(); t++) {
                if (debug) {
                    for (unsigned i = 0; i < opts.numHiddenNeurons(); i++) 
                        fprintf(file, "%.18f,", spikesOverTime[b][t * (opts.numInputNeurons() + opts.numHiddenNeurons()) + opts.numInputNeurons() + i]);

                    for (unsigned i = 0; i < opts.numHiddenNeurons(); i++) 
                        fprintf(file, "%.18f,", deltaErrorsOverTime[b][t * (opts.numHiddenNeurons() + opts.numAdaptiveHiddenNeurons() + opts.numOutputNeurons()) + i]);

                    for (unsigned i = 0; i < opts.numAdaptiveHiddenNeurons(); i++) 
                        fprintf(file, "%.18f,", deltaErrorsOverTime[b][t * (opts.numHiddenNeurons() + opts.numAdaptiveHiddenNeurons() + opts.numOutputNeurons()) + opts.numHiddenNeurons() + i]);

                    for (unsigned i = 0; i < opts.numOutputNeurons(); i++) 
                        fprintf(file, "%.18f,", deltaErrorsOverTime[b][t * (opts.numHiddenNeurons() + opts.numAdaptiveHiddenNeurons() + opts.numOutputNeurons()) + opts.numHiddenNeurons() + opts.numAdaptiveHiddenNeurons() + i]);
                }
                if (inputErrors) {
                    for (unsigned i = 0; i < opts.numInputNeurons(); i++) 
                        fprintf(file, "%.18f,", inputErrorsOverTime[t * opts.numInputNeurons() + i]);
                }
                for (unsigned i = 0; i < opts.numInputNeurons(); i++) 
                    fprintf(file, "%.18f,", inputsOverTime[t * opts.numInputNeurons() + i]);

                for (unsigned o = 0; o < opts.numOutputNeurons(); o++) 
                    fprintf(file, "%.18f,", outputsOverTime[t * opts.numOutputNeurons() + o]);

                for (unsigned o = 0; o < opts.numOutputNeurons(); o++) {
                    if (o == opts.numOutputNeurons() - 1)
                        fprintf(file, "%.18f\n", targetsOverTime[t * opts.numOutputNeurons() + o]); 
                    else
                        fprintf(file, "%.18f,",  targetsOverTime[t * opts.numOutputNeurons() + o]); 
                }
            }
            fclose(file);
        }
    }
}

/* returns the number of active synapses */
unsigned LongShortTermMemoryEligibilityNetwork::getNumActiveSynases() {
    return inputSynapseSerializer->getNumActiveSynases() + 
           hiddenSynapseSerializer->getNumActiveSynases() +
           outputSynapseSerializer->getNumActiveSynases();
}

