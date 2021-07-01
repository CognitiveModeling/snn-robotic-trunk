#ifndef __ADAPTIVE_LEAKY_INTEGRATE_AND_FIRE_SYNAPSE__
#define __ADAPTIVE_LEAKY_INTEGRATE_AND_FIRE_SYNAPSE__
#include "BasicSynapse.h"
#include "AdaptiveLeakyIntegrateAndFireNeuron.h"
/**
 * A Synapse between a BasicNeuron and a AdaptivLeakyIntegrateAndFireNeuron
 */
namespace SNN {

    namespace Synapses {

        class AdaptiveLeakyIntegrateAndFireSynapse: public Interfaces::BasicSynapse {

            private:

                /* the Eligibility trace of this */
                FloatType voltageEligibilityVector;
                FloatType thresholdAdaptionEligibilityVector;

                /* the input Eligibility trace of this */
                FloatType inputVoltageEligibilityVector;
                FloatType inputThresholdAdaptionEligibilityVector;


                /* should calculate the eligibility value for the current timestep */
                virtual FloatType getEligibilityTrace(bool inputSynapse = false) {
                    Neurons::ALIFNeuron &alifOutput  = static_cast<Neurons::ALIFNeuron &>(output);
                    
                    /* spikes from input neurons had an influenc on the current timestep */
                    if (inputSynapse) {

                        FloatType thresholdAdaptionEligibilityVectorLast = 
                            thresholdAdaptionEligibilityVector;

                        voltageEligibilityVector =                  // from paper
                            alifOutput.getVoltageDecayFactor() *    // \alpha
                            voltageEligibilityVector +              // \epsilon_{ji,v}^t
                            input.getOutput();                      // z_i^t

                        /* calculating adaption eligibility for the next time step */
                        thresholdAdaptionEligibilityVector =                    // from paper
                            alifOutput.getDerivative() *                        // h_j^t
                            voltageEligibilityVector +                          // \epsilon_{ji,v}^t
                            (
                                alifOutput.getThresholdAdaptionDecayFactor() -  // p
                                alifOutput.getDerivative() *                    // h_j^t
                                alifOutput.getThresholdIncreaseConstant()       // \beta
                            ) * thresholdAdaptionEligibilityVector;             // \epsilon_{ji,a}^t

                                                                         // from paper
                        return output.getDerivative() * (                // h_j^t
                            voltageEligibilityVector -                   // \epsilon_{ji,v}^{t + 1}
                            alifOutput.getThresholdIncreaseConstant() *  // \beta
                            thresholdAdaptionEligibilityVectorLast       // \epsilon_{ji,a}^t
                        );

                    } 

                    /* since neurons are allready updated, we actually claulating
                     * the eligibility vector for the next timestep */
                    FloatType thresholdAdaptionEligibilityVectorLast = 
                        thresholdAdaptionEligibilityVector;
                    FloatType voltageEligibilityVectorLast =
                        voltageEligibilityVector;

                    thresholdAdaptionEligibilityVector =                    // from paper
                        alifOutput.getDerivative() *                        // h_j^t
                        voltageEligibilityVector +                          // \epsilon_{ji,v}^t
                        (
                            alifOutput.getThresholdAdaptionDecayFactor() -  // p
                            alifOutput.getDerivative() *                    // h_j^t
                            alifOutput.getThresholdIncreaseConstant()       // \beta
                        ) * thresholdAdaptionEligibilityVector;             // \epsilon_{ji,a}^t

                    voltageEligibilityVector =                  // from paper
                        alifOutput.getVoltageDecayFactor() *    // \alpha
                        voltageEligibilityVector +              // \epsilon_{ji,v}^t
                        input.getOutput();                      // z_i^t

                                                                     // from paper
                    return output.getDerivative() * (                // h_j^t
                        voltageEligibilityVectorLast -               // \epsilon_{ji,v}^{t + 1}
                        alifOutput.getThresholdIncreaseConstant() *  // \beta
                        thresholdAdaptionEligibilityVectorLast       // \epsilon_{ji,a}^t
                    );
                }

                /* should calculate the eligibility value for the current timestep */
                virtual FloatType getInputEligibilityTrace(bool inputSynapse = false) {
                    Neurons::ALIFNeuron &alifOutput  = static_cast<Neurons::ALIFNeuron &>(output);

                    /* spikes from input neurons had an influenc on the current timestep */
                    if (inputSynapse) {

                        FloatType inputThresholdAdaptionEligibilityVectorLast = 
                            inputThresholdAdaptionEligibilityVector;

                        inputVoltageEligibilityVector =             // from paper
                            alifOutput.getVoltageDecayFactor() *    // \alpha
                            inputVoltageEligibilityVector +         // \epsilon_{ji,v}^t
                            this->getWeight();                      // w_i

                        inputThresholdAdaptionEligibilityVector =               // from paper
                            alifOutput.getDerivative() *                        // h_j^t
                            inputVoltageEligibilityVector +                     // \epsilon_{ji,v}^t
                            (
                                alifOutput.getThresholdAdaptionDecayFactor() -  // p
                                alifOutput.getDerivative() *                    // h_j^t
                                alifOutput.getThresholdIncreaseConstant()       // \beta
                            ) * inputThresholdAdaptionEligibilityVector;        // \epsilon_{ji,a}^t

                                                                         // from paper
                        return output.getDerivative() * (                // h_j^t
                            inputVoltageEligibilityVector -              // \epsilon_{ji,v}^{t + 1}
                            alifOutput.getThresholdIncreaseConstant() *  // \beta
                            inputThresholdAdaptionEligibilityVectorLast  // \epsilon_{ji,a}^t
                        );
                    }
                    
                    /* since neurons are allready updated, we actually claulating
                     * the eligibility vector for the next timestep */
                    FloatType inputThresholdAdaptionEligibilityVectorLast = 
                        inputThresholdAdaptionEligibilityVector;
                    FloatType inputVoltageEligibilityVectorLast =
                        inputVoltageEligibilityVector;

                    inputThresholdAdaptionEligibilityVector =               // from paper
                        alifOutput.getDerivative() *                        // h_j^t
                        inputVoltageEligibilityVector +                     // \epsilon_{ji,v}^t
                        (
                            alifOutput.getThresholdAdaptionDecayFactor() -  // p
                            alifOutput.getDerivative() *                    // h_j^t
                            alifOutput.getThresholdIncreaseConstant()       // \beta
                        ) * inputThresholdAdaptionEligibilityVector;        // \epsilon_{ji,a}^t

                    inputVoltageEligibilityVector =             // from paper
                        alifOutput.getVoltageDecayFactor() *    // \alpha
                        inputVoltageEligibilityVector +         // \epsilon_{ji,v}^t
                        this->getWeight();                      // w_i

                                                                     // from paper
                    return output.getDerivative() * (                // h_j^t
                        inputVoltageEligibilityVectorLast -          // \epsilon_{ji,v}^{t + 1}
                        alifOutput.getThresholdIncreaseConstant() *  // \beta
                        inputThresholdAdaptionEligibilityVectorLast  // \epsilon_{ji,a}^t
                    );
                }

                /* should reset child values */
                virtual void doReset() {
                    voltageEligibilityVector = 0;
                    thresholdAdaptionEligibilityVector = 0;
                    inputVoltageEligibilityVector = 0;
                    thresholdAdaptionEligibilityVector = 0;
                }

            public:

                /* constructor */
                AdaptiveLeakyIntegrateAndFireSynapse(
                    Interfaces::BasicNeuron &input,
                    Neurons::AdaptiveLeakyIntegrateAndFireNeuron &output,
                    FloatType weight
                ) : BasicSynapse(input, output, weight),
                    voltageEligibilityVector(0),
                    thresholdAdaptionEligibilityVector(0) { }

                /* returns the eligibility vector */
                virtual std::vector<FloatType> getEligibilityVector() {
                    return { voltageEligibilityVector, thresholdAdaptionEligibilityVector };
                }

                /* returns a list containing the name for each eligibility vector component */
                virtual std::vector<std::string> getEligibilityVectorNames() {
                    return { "voltage eligibility", "threshold adaption eligibility" };
                }

        };
    }

}
#endif /* __ADAPTIVE_LEAKY_INTEGRATE_AND_FIRE_SYNAPSE__ */
