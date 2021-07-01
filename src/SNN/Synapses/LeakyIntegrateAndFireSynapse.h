#ifndef __LEAKY_INTEGRATE_AND_FIRE_SYNAPSE__
#define __LEAKY_INTEGRATE_AND_FIRE_SYNAPSE__
#include "BasicSynapse.h"
#include "LeakyIntegrateAndFireNeuron.h"
/**
 * A Synapse between a BasicNeuron and a LeakyIntegrateAndFireNeuron
 */
namespace SNN {

    namespace Synapses {

        class LeakyIntegrateAndFireSynapse: public Interfaces::BasicSynapse {

            private:

                /* the Eligibility trace of this */
                FloatType eligibilityVector;

                /* the input eligibility vector */
                FloatType inputEligibilityVector;

                /* should calculate the eligibility value for the current timestep */
                virtual FloatType getEligibilityTrace(bool inputSynapse = false) {
                    Neurons::LIFNeuron &lifOutput = static_cast<Neurons::LIFNeuron &>(output);
                                                                           
                    /* only spikes from the last timestep can have a influenc 
                     * on the current eligibility trace */
                    FloatType eligibilityVectorLast = eligibilityVector;

                    eligibilityVector =                        // from paper
                        lifOutput.getVoltageDecayFactor() *    // \alpha
                        eligibilityVector +                    // \epsilon_{ji}^t
                        input.getOutput();                     // z_i^t

                    /* spikes from input neurons had an influenc on the current timestep */
                    if (inputSynapse)
                        eligibilityVectorLast = eligibilityVector;
                    
                    // from paper h_j^t * \epsilon_{ji}^t
                    return output.getDerivative() * eligibilityVectorLast;
                }

                /* should calculate the input eligibility value for the current timestep */
                virtual FloatType getInputEligibilityTrace(bool inputSynapse = false) {
                    Neurons::LIFNeuron &lifOutput = static_cast<Neurons::LIFNeuron &>(output);
                                                                           
                    /* only spikes from the last timestep can have a influenc 
                     * on the current eligibility trace */
                    FloatType inputEligibilityVectorLast = inputEligibilityVector;

                    inputEligibilityVector =                   // from paper
                        lifOutput.getVoltageDecayFactor() *    // \alpha
                        inputEligibilityVector +               // \epsilon_{ji}^t
                        this->getWeight();                     // w_i

                    /* spikes from input neurons had an influenc on the current timestep */
                    if (inputSynapse)
                        inputEligibilityVectorLast = inputEligibilityVector;
                    
                    // from paper h_j^t * \epsilon_{ji}^t
                    return output.getDerivative() * inputEligibilityVectorLast;
                }

                /* should reset child values */
                virtual void doReset() {
                    eligibilityVector = 0;
                    inputEligibilityVector = 0;
                }

            public:

                /* constructor */
                LeakyIntegrateAndFireSynapse(
                    Interfaces::BasicNeuron &input,
                    Neurons::LeakyIntegrateAndFireNeuron &output,
                    FloatType weight
                ) : BasicSynapse(input, output, weight),
                    eligibilityVector(0) { }

                /* returns the eligibility vector */
                virtual std::vector<FloatType> getEligibilityVector() {
                    return { eligibilityVector };
                }

                /* returns a list containing the name for each eligibility vector component */
                virtual std::vector<std::string> getEligibilityVectorNames() {
                    return { "voltage eligibility" };
                }

        };
    }

}
#endif /* __LEAKY_INTEGRATE_AND_FIRE_SYNAPSE__ */
