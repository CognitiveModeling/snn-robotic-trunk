#ifndef __LEAKY_INTEGRATE_AND_FIRE_NEURON__
#define __LEAKY_INTEGRATE_AND_FIRE_NEURON__
#include "BasicNeuron.h"
#include <math.h>
/**
 * A Leaky Integrate And Fire Neuron
 */
namespace SNN {

    namespace Neurons {

        class LeakyIntegrateAndFireNeuron: public Interfaces::BasicNeuron {

            private:

                /* the membran potential of this */
                FloatType v;

                /* the input current of this */
                FloatType I;

                /* the threshold of this */
                FloatType threshold;

                /* the decay factor fo this */
                FloatType decayFactor;

                /* the refactory period in timesteps */
                int refractoryPeriod;

                /* counter for the refactory period */
                int timeStepsSinceLastSpike;

                /* pseudo-derivative dumping factor */
                FloatType derivativeDumpingFactor;

                /* the pseudo-derivative of this */
                FloatType pseudoDerivative;

                /* delta error for backpropagation */
                FloatType deltaError;

            public:

                /* constructor */
                LeakyIntegrateAndFireNeuron(
                    Interfaces::BasicNetworkOptions &opts,
                    FloatType threshold,
                    FloatType refractoryPeriod,
                    FloatType membranTimeConstant,
                    FloatType derivativeDumpingFactor = 0.3
                ) : BasicNeuron(opts),
                    threshold(threshold),
                    decayFactor(exp(-1.0 * opts.timeStepLength() / membranTimeConstant)),
                    refractoryPeriod(int(round(refractoryPeriod / opts.timeStepLength()))),
                    derivativeDumpingFactor(derivativeDumpingFactor) {
                    doReset();
                }

                /* resets this */
                void doReset() {
                    v = 0;
                    I = 0;
                    pseudoDerivative = 0;
                    timeStepsSinceLastSpike = 2 * refractoryPeriod;
                    deltaError = 0;
                }

                /* adds a input current to this */
                void addCurrent(FloatType current) {
                    I += current;
                }

                /* computes the next timestep */
                void doUpdate() {

                    bool fired = this->fired();

                    /* update voltage */
                    v = v * decayFactor + I - int(fired) * threshold;
                    I = 0;

                    /* update number of timesteps that have pased since the last spike */
                    timeStepsSinceLastSpike = fired ? 1 : timeStepsSinceLastSpike + 1;

                    /* pseudo-derivative is fixed to zero within the refractory period */
                    if (timeStepsSinceLastSpike > refractoryPeriod) {
                        pseudoDerivative = derivativeDumpingFactor * std::max(
                            0.0, 
                            1.0 - fabs((v - threshold) / threshold)
                        );
                    } else 
                        pseudoDerivative = 0;
                }

                /* should compute the voltage component of next delta error 
                 * (we need voltage component only since other components 
                 *  will get zero weighted during learn signal computations)
                 */
                FloatType computeDeltaErrorVoltage(FloatType learnSignal, FloatType derivative) {
                    deltaError = learnSignal * derivative + deltaError * decayFactor;
                    return deltaError;
                }

                /* should compute the reset value for the learn signal */
                virtual FloatType resetLearnSignal() {
                    return -1 * deltaError * threshold;
                }

                /* returns whether this neuron has fired during the current time step */
                bool fired() {
                    return timeStepsSinceLastSpike > refractoryPeriod && v > threshold;
                }

                /* returns the output of this Neuron */
                virtual FloatType getOutput() {
                    return fired() ? 1.0 : 0.0;
                }

                /* returns the pseudo-derivative of this */
                FloatType getDerivative() {
                    return pseudoDerivative;
                }

                /* returns the voltage decay factor */
                FloatType getVoltageDecayFactor() {
                    return decayFactor;
                }

                /* returns the voltage of this */
                FloatType getVoltage() {
                    return v;
                }

                /* should return the input current of this */
                virtual FloatType getInputCurrent() {
                    return I;
                }

                /* should set the voltage of this */
                virtual void setVoltage(FloatType v) {
                    this->v = v;
                }

                /* should set the input current of this */
                virtual void setInputCurrent(FloatType I) {
                    this->I = I;
                }

                /* should set the (pseudo) derivative of this */
                void setDerivative(FloatType pseudoDerivative) {
                    this->pseudoDerivative = pseudoDerivative;
                }

                /* should return the threshold of this */
                virtual FloatType getThreshold() {
                    return threshold;
                }

                /* sets the delta error of this */
                virtual void doSetDeltaError(std::vector<FloatType> errors) {
                    deltaError = errors[0];
                }

#ifdef __TESTING__

                /* returns the threshold of this */
                FloatType getThreshold() { 
                    return threshold;
                }

                /* sets the threshold of this */
                void setThreshold(FloatType threshold) { 
                    this->threshold = threshold;
                }

#endif
        };

        /* type shortening */
        typedef LeakyIntegrateAndFireNeuron LIFNeuron;

    }

}
#endif /* __LEAKY_INTEGRATE_AND_FIRE_NEURON__ */
