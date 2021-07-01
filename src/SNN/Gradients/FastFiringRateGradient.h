#ifndef __FAST_FIRING_RATE_GRADIENT_H__
#define __FAST_FIRING_RATE_GRADIENT_H__
#include "BasicGradient.h"
#include "BasicNetwork.h"
#include "BasicNetworkOptions.h"
#include "utils.h"
/**
 * Gradient for firing rates
 */
namespace SNN {
    
    namespace Gradients {

        class FastFiringRateGradient: public Interfaces::BasicGradient {

            private: 

                /* the firing rate gradient of this */
                FloatType firingRateGradient;

                /* the target firing rate */
                FloatType targetFiringRate;

                /* the firing rate scalling factor */
                FloatType firingRateScallingFactor;

                /* should calculate the gradient for the current time step (forward pass) */
                virtual FloatType calcGradientForward() {

                    firingRateGradient += 
                        firingRateScallingFactor *
                        (synapse->getOutput().getFiringRate() - targetFiringRate) *
                        (synapse->getInput().getOutput() + synapse->getOutput().getOutput());

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
                FastFiringRateGradient(
                    Interfaces::BasicSynapse &synapse,
                    FloatType targetFiringRate,
                    FloatType firingRateScallingFactor
                ): 
                    BasicGradient(synapse),
                    firingRateGradient(0),
                    targetFiringRate(targetFiringRate),
                    firingRateScallingFactor(firingRateScallingFactor) { }

        };

    }
}

#endif 
