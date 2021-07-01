#ifndef __LEAKY_READOUT_GRADIENT_H__
#define __LEAKY_READOUT_GRADIENT_H__
#include "BasicGradient.h"
#include "BasicNetwork.h"
#include "LeakyReadoutSynapse.h"
#include "utils.h"
/**
 * Gradient for the leaky readout neurons of the network
 */
namespace SNN {
    
    namespace Gradients {

        class LeakyReadoutGradient: public Interfaces::BasicGradient {

            private: 

                /* the index of the readout neuron of this */
                unsigned outputIndex;

                /* readout gradient */
                FloatType readoutGradient;

                /* wether we are a classification or regression network */
                bool classification;


                /* should calculate the gradient for the current time step (forward pass) */
                virtual FloatType calcGradientForward() {
                    assert(network != NULL);
                    FloatType lernSignal = network->getErrorMask();

                    if (classification) {
                        lernSignal *= network->getSoftmaxOutput(outputIndex) - 
                                      network->getTargetSignal(outputIndex);
                    } else {
                        lernSignal *= synapse->getOutput().getOutput() - 
                                      network->getTargetSignal(outputIndex);
                    }

                    readoutGradient     += lernSignal * synapse->getEligibilityTrace();
                    return readoutGradient;
                }

                /* should calculate the gradient for the current time step (backward pass) */
                virtual FloatType calcGradientBackward() {
                    return readoutGradient;
                }

                /* should reset child values */
                virtual void doReset() {
                    readoutGradient = 0;
                }

            public:

                /* constructor */
                LeakyReadoutGradient(
                    Synapses::LeakyReadoutSynapse &synapse,
                    unsigned outputIndex,
                    bool classification = false
                ): 
                    BasicGradient(synapse),
                    outputIndex(outputIndex),
                    readoutGradient(0),
                    classification(classification) { }
        };

    }
}

#endif 
