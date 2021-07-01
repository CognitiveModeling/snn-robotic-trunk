#ifndef __CURRENT_INPUT_NEURON__
#define __CURRENT_INPUT_NEURON__
#include "BasicNeuron.h"
#include <math.h>
/**
 * A Leaky Integrate And Fire Neuron
 */
namespace SNN {

    namespace Neurons {

        class CurrentInputNeuron: public Interfaces::BasicNeuron {

            private:

                /* the membran potential of this */
                FloatType v;

                /* the input current of this */
                FloatType I;

            public:

                /* constructor */
                CurrentInputNeuron(Interfaces::BasicNetworkOptions &opts): 
                    BasicNeuron(opts), v(0), I(0) { }

                /* resets this */
                void doReset() {
                    v = 0;
                    I = 0;
                }

                /* adds a input current to this */
                void addCurrent(FloatType current) {
                    I += current;
                }

                /* computes the next timestep */
                void doUpdate() {
                    v = I;
                    I = 0;
                }

                /* should compute the voltage component of next delta error 
                 * (we need voltage component only since other components 
                 *  will get zero weighted during learn signal computations)
                 */
                FloatType computeDeltaErrorVoltage(FloatType learnSignal, FloatType derivative) {
                    return learnSignal * derivative;
                }

                /* returns whether this neuron has fired during the current time step */
                bool fired() {
                    return v != 0;
                }

                /* returns the output of this Neuron */
                virtual FloatType getOutput() {
                    return v;
                }

                /* returns the pseudo-derivative of this */
                FloatType getDerivative() {
                    return 1;
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

                /* should set the (pseudo) derivative of this */
                virtual void setDerivative(FloatType) { }

                /* should set the input current of this */
                virtual void setInputCurrent(FloatType I) {
                    this->I = I;
                }

        };

        /* type shortening */
        typedef CurrentInputNeuron CINeuron;

    }

}
#endif /* __CURRENT_INPUT_NEURON__ */
