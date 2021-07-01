#ifndef __BASIC_GRADIENT__
#define __BASIC_GRADIENT__
#include "BasicSynapse.h"
/**
 * Class that combines a leraning signal with an eligibility trace
 * in order to estimate a lerning gradient
 */
namespace SNN {

    namespace Interfaces {

        /* forward declaration */
        class BasicNetwork;
        
        class BasicGradient {

            protected:

                /* the synapses of this */
                BasicSynapse *synapse;

                /*  back reference to the network of this */
                BasicNetwork *network;

            private:

                /* the current gradient of this */
                FloatType gradient;

                /* the accumulated gradient (over the different simulations per batch) */
                FloatType accumulatedGradient;

                /* the simulation time step counter */
                int timeStepCounter;

                /* should calculate the gradient for non synapse dependent gradients */
                virtual FloatType calcGradient() { assert(false); return 0; }

                /* should calculate the gradient for the current time step (forward pass) */
                virtual FloatType calcGradientForward() { assert(false); return 0; }

                /* should calculate the gradient for the current time step (backward pass) */
                virtual FloatType calcGradientBackward() { assert(false); return 0; }

                /* should reset child values */
                virtual void doReset() { }
            
            public:

                /* constructor */
                BasicGradient(BasicSynapse &synapse): 
                    synapse(&synapse), 
                    network(NULL),
                    gradient(0), 
                    accumulatedGradient(0),
                    timeStepCounter(0) { }

                /* constructor for non synapse dependent gradients */
                BasicGradient(): 
                    synapse(NULL), 
                    network(NULL),
                    gradient(0), 
                    accumulatedGradient(0),
                    timeStepCounter(0) { }

                /* resets this */
                void reset() {
                    this->doReset();
                    accumulatedGradient = 0;
                    gradient            = 0;
                    timeStepCounter     = 0;
                }

                /* resets the timeStepCounter and accumulates the gradientfor the current batch */
                virtual void networkReset() {
                    timeStepCounter      = 0;
                    accumulatedGradient += gradient;
                    this->doReset();
                }

                /* virtual destructor for correct polymophism */
                virtual ~BasicGradient() { }

                /* returns the gradient for the current timestep */
                FloatType getGradient() {
                    return gradient;
                }

                /* sets the gradient for the current timestep */
                void setGradient(FloatType gradient) {
                    this->gradient = gradient;
                }

                /* sets the accumulated gradient for the current timestep */
                void setAccumulatedGradient(FloatType gradient) {
                    this->accumulatedGradient = gradient;
                }

                /* returns the accumulated gradient */
                FloatType getAccumulatedGradient() {
                    if (fabs(gradient) > 1e10 || fabs(accumulatedGradient) > 1e10)
                        log_err("gradient to high: " + ftoa(std::max(fabs(accumulatedGradient), fabs(gradient))), LOG_W);
                    return (accumulatedGradient != 0) ? accumulatedGradient : gradient; 
                }

                /* updates this */
                void update() {

                    /* synapses have to be updated bevore this */
                    if (synapse != NULL) {
                        timeStepCounter++;
                        //assert(synapse->getSimulationTimeStep() == timeStepCounter);

                        gradient = this->calcGradientForward();        
                    } else 
                        gradient = this->calcGradient();
                }

                /* updates this (back propagation) */
                void backPropagate() {
                    assert(synapse != NULL);

                    /* synapses have to be updated bevore this */
                    timeStepCounter--;
                    //assert(synapse->getSimulationTimeStep() == timeStepCounter);
                    //assert(timeStepCounter >= 0);

                    gradient = this->calcGradientBackward();        
                }

                /* returns the synapse of this */
                BasicSynapse &getSynapse() {
                    return *synapse;
                }

                /* set the synapse of this */
                void setSynapse(BasicSynapse &synapse) {
                    this->synapse = &synapse;
                }

                /* returns the current simulation time step of this */
                int getSimulationTimeStep() {
                    return timeStepCounter;
                }
                
                /* sets the current simulation time step of this */
                void setSimulationTimeStep(int timeStepCounter) {
                    this->timeStepCounter = timeStepCounter;
                }

                /* registers the network by this */
                virtual void registerNetwork(BasicNetwork *network) {
                    this->network = network;
                }

                /* returns the weight of the synapse of this */
                FloatType getWeight() {
                    return (synapse == NULL) ? 0 : synapse->getWeight();
                }
        };
    }
}

#endif /* __BASIC_GRADIENT__ */
