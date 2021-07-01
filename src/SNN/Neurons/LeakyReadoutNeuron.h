#ifndef __LEAKY_READOUT_NEURON__
#define __LEAKY_READOUT_NEURON__
#include "BasicNeuron.h"
#include <math.h>
/**
 * A Leaky Readout Neuron
 */
namespace SNN {

    namespace Neurons {

        class LeakyReadoutNeuron: public Interfaces::BasicNeuron {

            private:

                /* the membran potential of this */
                FloatType v;

                /* the input current of this */
                FloatType I;

                /* the bias of this */
                FloatType bias;

                /* the voltage decay factor fo this */
                FloatType decayFactor;

                /* the voltage delta error of this */
                FloatType deltaError;

                /* wether this is a spike timing readout or not */
                bool spikeTimingReadout;

            public:

                /* constructor */
                LeakyReadoutNeuron(
                    Interfaces::BasicNetworkOptions &opts,
                    FloatType bias,
                    FloatType membranTimeConstant = 20,
                    bool spikeTimingReadout = false
                ) : BasicNeuron(opts), 
                    bias(bias),
                    decayFactor(exp(-1.0 * opts.timeStepLength() / membranTimeConstant)),
                    spikeTimingReadout(spikeTimingReadout) {
                    doReset();
                }

                /* resets this */
                void doReset() {
                    v = 0;
                    I = 0;
                    deltaError = 0;
                }

                /* adds a input current to this */
                void addCurrent(FloatType current) {
                    I += current;
                }

                /* computes the next timestep */
                void doUpdate() {

                    /* update voltage */
                    v = v * decayFactor + I + bias;
                    I = 0;
                }

                /* should compute the voltage component of next delta error 
                 * (we need voltage component only since other components 
                 *  will get zero weighted during learn signal computations)
                 */
                FloatType computeDeltaErrorVoltage(FloatType learnSignal, FloatType) {
                    if (spikeTimingReadout) 
                        deltaError = learnSignal;
                    else
                        deltaError = learnSignal + decayFactor * deltaError;

                    return deltaError;
                }

                /* since this is a leaky artificial neuron ist output is defined by ist voltage */
                bool fired() {
                    return v > 0.9;
                }

                /* returns the output of this Neuron */
                virtual FloatType getOutput() {
                    return v;
                }

                /* since for the output of this f(v) = v
                 * the derivative f'(v) = 1 */
                FloatType getDerivative() {
                    return 1;
                }

                /* should set the (pseudo) derivative of this */
                virtual void setDerivative(FloatType) { }

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


                /* returns the voltage decay factor */
                FloatType getVoltageDecayFactor() {
                    return decayFactor;
                }

                /* sets the delta error of this */
                virtual void doSetDeltaError(std::vector<FloatType> errors) {
                    deltaError = errors[0];
                }
        };

        /* type shortening */
        typedef LeakyReadoutNeuron LRNeuron;

    }

}
#endif /* __LEAKY_READOUT_NEURON__ */
