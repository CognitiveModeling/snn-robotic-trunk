#ifndef __CURRENT_INPUT_GRADIENT_H__
#define __CURRENT_INPUT_GRADIENT_H__
#include "BasicGradient.h"
#include "BasicNetwork.h"
#include "BasicNetworkOptions.h"
#include "utils.h"
/**
 * Gradient for firing rates
 */
namespace SNN {
    
    namespace Gradients {

        class CurrentInputGradient: public Interfaces::BasicGradient {

            private: 

                /* the time period for the gradient */
                int startTime, endTime;

                /* neuron interval associated with the input */
                unsigned firstNeuron, lastNeuron;

                /* the input errors over time */
                FloatType *inputErrorsOverTime;

                /* the total number of input neurons */
                unsigned numInputNeurons;

                /* salling factor */
                FloatType scallingFactor;

                /* should calculate the gradient for non synapse dependent gradients */
                virtual FloatType calcGradient() { 

                    FloatType inputGradient = 0;

                    for (int t = startTime; t <= endTime; t++) {
                        for (unsigned n = firstNeuron; n <= lastNeuron; n++) {
                            inputGradient += inputErrorsOverTime[t * numInputNeurons + n];
                        }
                    }
                    return scallingFactor * inputGradient / ((endTime - startTime + 1) * (lastNeuron - firstNeuron + 1));
                }

            public:

                /* constructor */
                CurrentInputGradient(
                    int startTime,
                    int endTime,
                    unsigned firstNeuron,
                    unsigned lastNeuron,
                    unsigned numInputNeurons,
                    FloatType *inputErrorsOverTime
                ): 
                    startTime(startTime),
                    endTime(endTime),
                    firstNeuron(firstNeuron),
                    lastNeuron(lastNeuron),
                    inputErrorsOverTime(inputErrorsOverTime),
                    numInputNeurons(numInputNeurons),
                    scallingFactor(1){ }

                /* sets the start and end time of this */
                void setTimeWindow(int startTime, int endTime) {
                    this->startTime = startTime;
                    this->endTime   = endTime;
                }

                /* sets the scallingFactor */
                void setScallingFactor(FloatType scallingFactor) {
                    this->scallingFactor = scallingFactor;
                }

        };

    }
}

#endif 
