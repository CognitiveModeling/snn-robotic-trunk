#ifndef __FIXED_BROADCAST_GRADIENT_H__
#define __FIXED_BROADCAST_GRADIENT_H__
#include "BasicGradient.h"
#include "BasicNetwork.h"
#include "utils.h"
#include <unordered_map>
#include <assert.h>
/**
 * Gradient  with fixed weigths for a global error brodcast of the network
 */
namespace SNN {
    
    namespace Gradients {

        /* a feedback weight matrix */
        class BasicFeedbackWeights {

            public:

                /* destructor */
                virtual ~BasicFeedbackWeights() { }

                /* returns the weight for the given input and 
                 * the given output and the given timestep */
                virtual FloatType getWeight(unsigned input, unsigned output) = 0;
                virtual FloatType getWeight(unsigned index) = 0;

                /* sets the given weight for the given input and 
                 * the given output and the given timestep */
                virtual void setWeight(FloatType weight, unsigned input, unsigned output) = 0;
                virtual void setWeight(FloatType weight, unsigned index) = 0;

                /* should return the number of weights */
                virtual size_t size() = 0;

                /* sets all weights to zero */
                virtual void clear() = 0;
        };

        /* a feedback weight matrix */
        class FeedbackWeights: public BasicFeedbackWeights {

            private:

                /* the weights of this */
                std::vector<std::vector<FloatType>> weights;

            public:

                /* creats a random feed back weight matrix */
                FeedbackWeights(unsigned numInputs, unsigned numOutputs) {
                    
                    rand128_t *rand = new_rand128_t();
                    for (unsigned i = 0; i < numInputs; i++) {
                        weights.push_back(std::vector<FloatType>());

                        for (unsigned o = 0;o < numOutputs; o++) {
                            weights.back().push_back(
                                rand128n(rand, 0.0, 1.0 / numInputs)
                            );
                        }
                    }
                    free(rand);
                }

                /* returns the weight for the given input and 
                 * the given output and the given timestep */
                FloatType getWeight(unsigned input, unsigned output) {
                    assert(weights.size() > input);
                    assert(weights[input].size() > output);
                    return weights[input][output];
                }
                virtual FloatType getWeight(unsigned index) {
                    return getWeight(index / numOutputs(), index % numOutputs());
                }

                /* returns the number of inputs */
                unsigned numInputs() { return weights.size(); }

                /* returns the number of outputs */
                unsigned numOutputs() { return weights[0].size(); }

                /* should return the number of weights */
                virtual size_t size() { return weights.size() * weights[0].size(); }

                /* sets the given weight for the given input and 
                 * the given output and the given timestep */
                void setWeight(FloatType weight, unsigned input, unsigned output) {
                    assert(weights.size() > input);
                    assert(weights[input].size() > output);
                    weights[input][output] = weight;
                }
                virtual void setWeight(FloatType weight, unsigned index) {
                    return setWeight(weight, index / numOutputs(), index % numOutputs());
                }

                /* sets all weights to zero */
                void clear() {
                    for (auto &vv: weights)
                        for (auto &v: vv)
                            v = 0;
                }
        };

        /* a feedback weight matrix */
        class SparseFeedbackWeights: public BasicFeedbackWeights {

            private:

                /* fast access to the weights of this */
                std::unordered_map<uint64_t, FloatType> weights;

                /* input and output indices for the weights of this */
                std::vector<unsigned> inputIndices;
                std::vector<unsigned> outputIndices;

            public:

                /* creats a random feed back weight matrix */
                SparseFeedbackWeights(
                    std::vector<FloatType> weights,
                    std::vector<unsigned> inputIndices,
                    std::vector<unsigned> outputIndices
                ): inputIndices(inputIndices), outputIndices(outputIndices) {
                    
                    assert(weights.size() == inputIndices.size());
                    assert(weights.size() == outputIndices.size());

                    for (unsigned i = 0; i < weights.size(); i++) {
                        this->weights.insert({
                            (uint64_t(inputIndices[i]) << 32) | uint64_t(outputIndices[i]),
                            weights[i]
                        });
                    }
                }

                /* returns the weight for the given input and 
                 * the given output and the given timestep */
                virtual FloatType getWeight(unsigned input, unsigned output) {
                    uint64_t id = (uint64_t(input) << 32) | uint64_t(output);
                    if (!weights.count(id))
                        return 0;

                    return weights[id];
                }
                virtual FloatType getWeight(unsigned index) {
                    assert(inputIndices.size() > index);
                    assert(outputIndices.size() > index);
                    return getWeight(inputIndices[index], outputIndices[index]);
                }

                /* sets the given weight for the given input and 
                 * the given output and the given timestep */
                virtual void setWeight(FloatType weight, unsigned input, unsigned output) {
                    uint64_t id = (uint64_t(input) << 32) | uint64_t(output);
                    assert(weights.count(id) != 0);
                    weights[id] = weight;
                }
                virtual void setWeight(FloatType weight, unsigned index) {
                    assert(inputIndices.size() > index);
                    assert(outputIndices.size() > index);
                    setWeight(weight, inputIndices[index], outputIndices[index]);
                }

                /* should return the number of weights */
                virtual size_t size() { return inputIndices.size(); }

                /* sets all weights to zero */
                virtual void clear() {
                    for (auto &e: weights) {
                        weights[e.first] = 0;
                    }
                }
        };

        class FixedBroadcastGradient: public Interfaces::BasicGradient {

            private: 

                /* the filtered eligibility trace of this */
                FloatType filteredEligibilityTrace;

                /* the leaky readout neuron decay factor for filtering the eligibility trace */
                FloatType decayFactor;

                /* the gradient of this */
                FloatType feedbackGradient;

                /* the random weights for calculating the error signal of this */
                std::shared_ptr<BasicFeedbackWeights> feedbackMatrix;

                /* the neuron index of this */
                unsigned neuronIndex;

                /* the number of output neurons of the network */
                unsigned numOutputNeurons;

                /* wether we are a classification or regression network */
                bool classification;

                /* should calculate the gradient for the current time step (forward pass) */
                virtual FloatType calcGradientForward() {
                    assert(network != NULL);

                    filteredEligibilityTrace *= decayFactor;
                    filteredEligibilityTrace += synapse->getEligibilityTrace();

                    /* randomly weighted network error */
                    FloatType lernSignal = 0;
                    for (unsigned i = 0; i < numOutputNeurons; i++) {
                        FloatType weight = feedbackMatrix->getWeight(neuronIndex,  i);
                        weight *= network->getTargetWeight(i);
                        weight *= network->getErrorMask();

                        if (classification)
                            lernSignal += weight * (network->getSoftmaxOutput(i) - network->getTargetSignal(i));
                        else
                            lernSignal += weight * (network->getOutput(i) - network->getTargetSignal(i));
                    }

                    feedbackGradient += lernSignal * filteredEligibilityTrace;
                    return feedbackGradient;
                }

                /* should calculate the gradient for the current time step (backward pass) */
                virtual FloatType calcGradientBackward() {
                    return feedbackGradient;
                }

                /* should reset child values */
                virtual void doReset() {
                    filteredEligibilityTrace = 0;
                    feedbackGradient = 0;
                }

            public:

                /* constructor */
                FixedBroadcastGradient(
                    Interfaces::BasicSynapse &synapse,
                    std::shared_ptr<BasicFeedbackWeights> feedbackMatrix,
                    unsigned numOutputNeurons,
                    unsigned neuronIndex,
                    FloatType readoutMembraneTimeConstant = 20,
                    FloatType timeStepLength = 1,
                    bool classification = false
                ): 
                    BasicGradient(synapse),
                    filteredEligibilityTrace(0),
                    decayFactor(exp(-1.0 * timeStepLength / readoutMembraneTimeConstant)),
                    feedbackGradient(0),
                    feedbackMatrix(feedbackMatrix),
                    neuronIndex(neuronIndex),
                    numOutputNeurons(numOutputNeurons),
                    classification(classification) { }

        };

    }
}

#endif 
