#ifndef __LEAKY_READOUT_SYNAPSE__
#define __LEAKY_READOUT_SYNAPSE__
#include "BasicSynapse.h"
#include "LeakyReadoutNeuron.h"
/**
 * A Synapse between a BasicNeuron and a Leaky Readout Neuron
 */
namespace SNN {

    namespace Synapses {

        class LeakyReadoutSynapse: public Interfaces::BasicSynapse {

            private:

				/* the eligibility voctor of this */
				FloatType eligibilityVector;

                /* For Leaky Readout Neurons the eligibility trace 
                 * is just the pesynaptice spike */
                virtual FloatType getEligibilityTrace(bool inputSynapse = false) {
                    assert(inputSynapse == false);

                    Neurons::LeakyReadoutNeuron &readout = 
                        static_cast<Neurons::LeakyReadoutNeuron &>(output);

					eligibilityVector = eligibilityVector * readout.getVoltageDecayFactor() +
                                        input.getOutput();

                    return eligibilityVector;
                }

                /* should reset child values */
                virtual void doReset() {
                    eligibilityVector = 0;
                }

            public:

                /* constructor */
                LeakyReadoutSynapse(
                    Interfaces::BasicNeuron &input,
                    Neurons::LeakyReadoutNeuron &output,
                    FloatType weight
                ) : BasicSynapse(input, output, weight), eligibilityVector(0) { }

                /* returns the eligibility vector */
                virtual std::vector<FloatType> getEligibilityVector() {
                    return { eligibilityVector };
                }

                /* returns a list containing the name for each eligibility vector component */
                virtual std::vector<std::string> getEligibilityVectorNames() {
                    return { "input eligibility" };
                }

        };
    }

}
#endif /* __LEAKY_READOUT_SYNAPSE__ */
