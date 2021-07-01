#ifndef __ELIGIBILITY_BACK_PROPAGATED_GRADIENT_H__
#define __ELIGIBILITY_BACK_PROPAGATED_GRADIENT_H__
#include "BasicGradient.h"
#include "BasicNetwork.h"
#include "utils.h"
/**
 * Simple Gradient for Back Propagation Through Time
 */
namespace SNN {
    
    namespace Gradients {

        class EligibilityBackPropagatedGradient: public Interfaces::BasicGradient {

            private: 

                /* the gradient of this */
                FloatType backPropagatedGradient;

                /* the eligibility trace ofer time */
                std::vector<FloatType> eligibilityTraces;

                /* should calculate the gradient for the current time step (forward pass) */
                virtual FloatType calcGradientForward() {
                    eligibilityTraces.push_back(synapse->getEligibilityTrace());
                    return 0;
                }

                /* should calculate the gradient for the current time step (backward pass) */
                virtual FloatType calcGradientBackward() {

                    const int timeStep = this->getSimulationTimeStep();
                    assert(timeStep >= 0 && unsigned(timeStep + 1) == eligibilityTraces.size());

                    backPropagatedGradient += eligibilityTraces.back() * 
                                              synapse->getOutput().getLearnSignal();

                    eligibilityTraces.pop_back();
                    return backPropagatedGradient;
                }

                /* should reset child values */
                virtual void doReset() {
                    backPropagatedGradient = 0;
                }

            public:

                /* constructor */
                EligibilityBackPropagatedGradient(
                    Interfaces::BasicSynapse &synapse
                ): 
                    BasicGradient(synapse) {
                    doReset();
                }

        };

    }
}

#endif 
