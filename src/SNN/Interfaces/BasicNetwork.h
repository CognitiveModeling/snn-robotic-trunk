#ifndef __BASIC_NETWORK_H__
#define __BASIC_NETWORK_H__
#include <vector>
#include <memory>
#include "BasicNeuron.h"
#include "BasicSynapse.h"
#include "BasicGradient.h"
#include "BasicOptimizer.h"
/**
 * Interface for a basic network 
 */
namespace SNN {

    namespace Interfaces {

        class BasicNetwork {

            protected:

                /* the neurons of this */
                std::vector<std::shared_ptr<BasicNeuron>> neurons;

                /* the synapses of this */
                std::vector<std::shared_ptr<BasicSynapse>> synapses;

                /* the gradient estimators of this */
                std::vector<std::shared_ptr<BasicGradient>> gradients;

                /* the optimizers of this */
                std::vector<std::shared_ptr<BasicOptimizer>> optimizers;


                /* should perform updates bevor updating the network */
                virtual void updateBevor() { }

                /* should perform the actual update stepp */
                virtual void doUpdate() { 
                    for (auto &n: neurons)    n->update();
                    for (auto &s: synapses)   s->update();
                    for (auto &g: gradients)  g->update();
                }

                /* should perform updates after updating the network */
                virtual void updateAfter() { }

                /* should perform updates bevor a back propagation step of the network */
                virtual void backPropagateBevor() { }

                /* should perform the actual backpropagation step */
                virtual void doBackPropagation() {
                    for (auto &n: neurons)    n->backPropagate();
                    for (auto &s: synapses)   s->backPropagate();
                    for (auto &g: gradients)  g->backPropagate();
                }
                
                /* should perform updates after a back propagation step of the network */
                virtual void backPropagateAfter() { }

                /* should perform aditional resets */
                virtual void doReset() { }
            
            
            public:

                /* constructor */
                BasicNetwork() { }

                /* constructor */
                BasicNetwork(
                    std::vector<std::shared_ptr<BasicNeuron>> neurons,
                    std::vector<std::shared_ptr<BasicSynapse>> synapses,
                    std::vector<std::shared_ptr<BasicGradient>> gradients,
                    std::vector<std::shared_ptr<BasicOptimizer>> optimizers
                ) : neurons(neurons), 
                    synapses(synapses), 
                    gradients(gradients), 
                    optimizers(optimizers) { 
                    
                    for (auto &g: gradients)
                        g->registerNetwork(this);
                }

                /* updates this */
                void update(bool training = true) {
                    this->updateBevor();
                    this->doUpdate();

                    if (training)
                        for (auto &o: optimizers) o->update();

                    this->updateAfter();
                }

                /* updates this (back propagation) */
                void backPropagate(bool training = true) {
                    this->backPropagateBevor();
                    this->doBackPropagation();
                    
                    if (training)
                        for (auto &o: optimizers) o->backPropagate();

                    this->backPropagateAfter();
                }

                /* resets the network and the gradients of this */
                void resetNetwork() {
                    for (auto &s: synapses)   s->reset();
                    for (auto &n: neurons)    n->networkReset();
                    for (auto &g: gradients)  g->networkReset();
                    for (auto &o: optimizers) o->networkReset();
                }

                /* resets the network */
                void reset() {
                    for (auto &s: synapses)   s->reset();
                    for (auto &n: neurons)    n->reset();
                    for (auto &g: gradients)  g->reset();
                    for (auto &o: optimizers) o->reset();
                    this->doReset();
                }

                /* should return the target signal of the network */
                virtual FloatType getTargetSignal(unsigned) = 0;

                /* should return the target weight */
                virtual FloatType getTargetWeight(unsigned) { return 1; }

                /* should return the output of the network */
                virtual FloatType getOutput(unsigned) = 0;

                /* should return the number of network outputs */
                virtual unsigned getNumOutputs() = 0;

                /* should return the network error */
                virtual FloatType getError() = 0;

                /* should return the network accuracy */
                virtual FloatType getAccuracy() { return std::nan(""); };

                /* returns the average firing rate of the hidden neurons */
                virtual FloatType getAverageFiringRate() { return 0; }

                /* should return the network error mask */
                virtual FloatType getErrorMask() { return 1; }

                /* returns the soft max output with the given index */
                FloatType getSoftmaxOutput(unsigned index) {
                    FloatType expSum = 1e-9;
                    for (unsigned i = 0; i < this->getNumOutputs(); i++)
                        expSum += exp(this->getOutput(i));

                    return exp(this->getOutput(index)) / expSum;
                }

                /* access operators for neurons, synapses,  gradients and optimizers */
                unsigned long getNumNeurons()    { return neurons.size();    }
                unsigned long getNumSynapses()   { return synapses.size();   }
                unsigned long getNumGradients()  { return gradients.size();  }
                unsigned long getNumOptimizers() { return optimizers.size(); }
                Interfaces::BasicNeuron    &getNeuron(unsigned i)    { return *neurons[i];    }
                Interfaces::BasicSynapse   &getSynapse(unsigned i)   { return *synapses[i];   }
                Interfaces::BasicGradient  &getGradient(unsigned i)  { return *gradients[i];  }
                Interfaces::BasicOptimizer &getOptimizer(unsigned i) { return *optimizers[i]; }

                /* should return the gpu outputs for the given batch index */
                virtual FloatType *getGPUOutput(unsigned) { assert(false); return 0; }

                /* should return the gpu target for the given batch index */
                virtual FloatType *getGPUTarget(unsigned) { assert(false); return 0; }

                /* should return the number of simulation timesteps */
                virtual unsigned numSimulationTimesteps() { assert(false); return 0; }

                /* removes all optimizers with zero connections */
                void makeSparse() {
                    std::vector<std::shared_ptr<BasicOptimizer>> sparseOptimizers;
                    for (auto &o: optimizers)
                        if (fabs(o->getGradient().getSynapse().getWeight()) > 0.000001)
                            sparseOptimizers.push_back(o);

                    this->optimizers = sparseOptimizers;
                }
        };

    }

}

#endif /* __BASIC_NETWORK_H__ */
