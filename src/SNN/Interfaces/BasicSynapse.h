#ifndef __BASIC_SYNAPSE__
#define __BASIC_SYNAPSE__
#include "BasicNeuron.h"
#include "LeakyReadoutNeuron.h"
#include <assert.h>
/**
 * A Base class for Synapses between neurons
 */
namespace SNN {

    namespace Interfaces {

        class BasicSynapse {

            protected:

                /* the input Neuron of this */
                Interfaces::BasicNeuron &input;

                /* the output neuron of this */
                Interfaces::BasicNeuron &output;

                /* the weight of this */
                FloatType weight;

                /* synaptic delay of this */
                unsigned delay;

                /* value range for the weight of this */
                FloatType minWeight, maxWeight;

                /* the Eligibility trace of this */
                FloatType eligibilityTrace;

                /* the simulation time step counter */
                int timeStepCounter;

                /* should calculate the eligibility trace for the current timestep */
                virtual FloatType getEligibilityTrace(bool inputSynapse) = 0;

                /* should calculate the input eligibility trace for the current timestep */
                virtual FloatType getInputEligibilityTrace(bool inputSynapse) {
                    (void) inputSynapse;
                    return 0;
                }

                /* should reset child values */
                virtual void doReset() = 0;

            public:

                /* constructor */
                BasicSynapse(
                    Interfaces::BasicNeuron &input,
                    Interfaces::BasicNeuron &output,
                    FloatType weight,
                    FloatType delay = 0
                ) : 
                    input(input), 
                    output(output), 
                    weight(weight), 
                    delay(delay),
                    minWeight(-1e10),
                    maxWeight(1e10),
                    eligibilityTrace(0),
                    timeStepCounter(0) { }

                /* resets this */
                void reset() {
                    this->doReset();
                    eligibilityTrace = 0;
                    timeStepCounter  = 0;
                }

                /* virtual destructor for correct polymophism */
                virtual ~BasicSynapse() { }

                /* propagates the spike from the input neuron to the output neuron */
                virtual void propagateSpike() {
                    timeStepCounter++;
                    assert(input.getSimulationTimeStep() == timeStepCounter);
                    assert(output.getSimulationTimeStep() == timeStepCounter - 1 || 
                           output.getSimulationTimeStep() == timeStepCounter);

                    if (delay == 0)
                        output.addCurrent(weight * input.getOutput());
                    else if (output.getSimulationTimeStep() >= int(delay))
                        output.addCurrent(weight * input.getOutput(output.getSimulationTimeStep() - delay));
                }

                /* computes the eligibility trace of this */
                void computeEligibilityTrace(bool inputSynapse = false) {
                    //assert(input.getSimulationTimeStep() == timeStepCounter);
                    //assert(output.getSimulationTimeStep() == timeStepCounter);

                    eligibilityTrace = this->getEligibilityTrace(inputSynapse);
                    input.addEligibilityError(this->getInputEligibilityTrace(inputSynapse));
                }

                /* updates the synaptic conections of this (after neurons are updated) */
                void update(bool inputSynapse = false) {
                    this->propagateSpike();
                    this->computeEligibilityTrace(inputSynapse);
                }

                /* error back propagation (after neurons updated their error) */
                virtual void backPropagate() {
                    timeStepCounter--;
                    
                    assert(input.getSimulationTimeStep() - 1 == timeStepCounter);

                    assert(output.getSimulationTimeStep() == timeStepCounter ||
                           output.getSimulationTimeStep() - 1 == timeStepCounter);

                    assert(timeStepCounter >= 0);
                    
                    if (delay == 0)
                        input.addError(weight * output.getError());
                    else 
                        input.addError(weight * output.getError(output.getSimulationTimeStep() + delay - 1));
                }

                /* returns the Eligibility trace of this */
                FloatType getEligibilityTrace() {
                    return eligibilityTrace;
                }

                /* returns the input of this */
                Interfaces::BasicNeuron &getInput() {
                    return input;
                }

                /* returns the output of this */
                Interfaces::BasicNeuron &getOutput() {
                    return output;
                }

                /* returns the delay of this */
                unsigned getDelay() {
                    return delay;
                }


                /* should return the eligibility vector */
                virtual std::vector<FloatType> getEligibilityVector() = 0;

                /* should return a list containing the name for each eligibility vector component */
                virtual std::vector<std::string> getEligibilityVectorNames() = 0;

                /* returns the current simulation time step of this */
                int getSimulationTimeStep() {
                    return timeStepCounter;
                }

                /* sets the current simulation time step of this */
                void setSimulationTimeStep(int timeStepCounter) {
                    this->timeStepCounter = timeStepCounter;
                }

                /* trains this by applying the given weight change */
                void addWeightChange(FloatType weightChange) {
                    this->setWeight(weight + weightChange);
                }

                /* returns the weight of this */
                FloatType getWeight() {
                    return weight;
                }

                /* sets the weight of this */
                void setWeight(FloatType weight) {
                    this->weight = std::min(std::max(weight, minWeight), maxWeight);
                }

                /* sets the min and max of this */
                void constrain(FloatType minWeight, FloatType maxWeight) {
                    this->minWeight = minWeight;
                    this->maxWeight = maxWeight;
                }

                /* sets the eligibility trace of this */
                void setEligibilityTrace(FloatType eligibilityTrace) {
                    this->eligibilityTrace = eligibilityTrace;
                }

        };
    }

}
#endif /* __BASIC_SYNAPSE__ */
