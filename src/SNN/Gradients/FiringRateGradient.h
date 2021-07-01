#ifndef __FIRING_RATE_GRADIENT_H__
#define __FIRING_RATE_GRADIENT_H__
#include "BasicGradient.h"
#include "BasicNetwork.h"
#include "BasicNetworkOptions.h"
#include "utils.h"
/**
 * Gradient for firing rates
 */
namespace SNN {
    
    namespace Gradients {

        class FiringRateGradient: public Interfaces::BasicGradient {

            private: 

                /* the firing rate gradient of this */
                FloatType firingRateGradient;

                /* the network options of this */
                Interfaces::BasicNetworkOptions &opts;

                /* should calculate the gradient for the current time step (forward pass) */
                virtual FloatType calcGradientForward() {

                    firingRateGradient += 
                        opts.timeStepLength() / (opts.numSimulationTimesteps() * opts.batchSize()) *
                        (synapse->getOutput().getFiringRate() - opts.targetFiringRate()) *
                        synapse->getEligibilityTrace();

                    return firingRateGradient;
                }

                /* should calculate the gradient for the current time step (backward pass) */
                virtual FloatType calcGradientBackward() {
                    return firingRateGradient;
                }

                /* should reset child values */
                virtual void doReset() { 
                    firingRateGradient = 0;    
                }

            public:

                /* constructor */
                FiringRateGradient(
                    Interfaces::BasicSynapse &synapse,
                    Interfaces::BasicNetworkOptions &opts
                ): 
                    BasicGradient(synapse),
                    firingRateGradient(0),
                    opts(opts.freeze()) { }

        };

    }
}

#endif 
