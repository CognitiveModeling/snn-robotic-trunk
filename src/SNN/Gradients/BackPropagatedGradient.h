#ifndef __BACK_PROPAGATED_GRADIENT_H__
#define __BACK_PROPAGATED_GRADIENT_H__
#include "BasicGradient.h"
#include "BasicNetwork.h"
#include "utils.h"
/**
 * Simple Gradient for Back Propagation Through Time
 */
namespace SNN {
    
    namespace Gradients {

        class BackPropagatedGradient: public Interfaces::BasicGradient {

            private: 

                /* the gradient of this */
                FloatType backPropagatedGradient;

                /* wether this is a gradient for a hidden synapse or not */
                bool hiddenSynapse;

                /* should calculate the gradient for the current time step (forward pass) */
                virtual FloatType calcGradientForward() {
                    return 0;
                }

                /* should calculate the gradient for the current time step (backward pass) */
                virtual FloatType calcGradientBackward() {

                    const int timeStep = this->getSimulationTimeStep();
                    assert(timeStep >= 0);

                    if (synapse->getDelay() > 0) {

                        if (timeStep - int(synapse->getDelay()) >= 0) {

                            /* multiplies delta error from current timestep with 
                             * input spike form pervious timestep */
                            backPropagatedGradient += synapse->getOutput().getError() * 
                                                      synapse->getInput().getOutput(timeStep - synapse->getDelay());
                        }
                    } else {

                        if (hiddenSynapse) {
                            if (timeStep > 0) {

                                /* multiplies delta error from current timestep with 
                                 * input spike form pervious timestep */
                                backPropagatedGradient += synapse->getOutput().getError() * 
                                                          synapse->getInput().getOutput(timeStep - 1);
                            }
                        } else {

                            /* multiplies delta error from current timestep with 
                             * input spike form pervious layer */
                            backPropagatedGradient += synapse->getOutput().getError() * 
                                                      synapse->getInput().getOutput(timeStep);
                        }
                    }


                    return backPropagatedGradient;
                }

                /* should reset child values */
                virtual void doReset() {
                    backPropagatedGradient = 0;
                }

            public:

                /* constructor */
                BackPropagatedGradient(
                    Interfaces::BasicSynapse &synapse,
                    bool hiddenSynapse
                ): 
                    BasicGradient(synapse), hiddenSynapse(hiddenSynapse) {
                    doReset();
                }

        };

    }
}

#endif 
