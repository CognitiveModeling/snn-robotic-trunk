#ifndef __ADAPTIVE_LEAKY_INTEGRATE_AND_FIRE_NEURON__
#define __ADAPTIVE_LEAKY_INTEGRATE_AND_FIRE_NEURON__
#include "BasicNeuron.h"
/**
 * A Adaptiv Integrate And Fire Neuron with adaptive threshold
 */
namespace SNN {

    namespace Neurons {

        class AdaptiveLeakyIntegrateAndFireNeuron: public Interfaces::BasicNeuron {

            private:

                /* the membran potential of this */
                FloatType v;

                /* the input current of this */
                FloatType I;

                /* the base threshold of this */
                FloatType baseThreshold;

                /* the adapted threshold */
                FloatType adaptiveThreshold;

                /* the threshold adaptation value */
                FloatType thresholdAdaptation;

                /* the threshold adaptation weight */
                FloatType thresholdIncreaseConstant;

                /* adaptation deacy factor */
                FloatType adaptationDecay;

                /* the voltage decay factor fo this */
                FloatType voltageDecay;

                /* the refactory period in timesteps */
                int refractoryPeriod;

                /* counter for the refactory period */
                int timeStepsSinceLastSpike;

                /* pseudo-derivative dumping factor */
                FloatType derivativeDumpingFactor;

                /* the pseudo-derivative of this */
                FloatType pseudoDerivative;

                /* the voltage component of the delta error of this */
                FloatType deltaErrorVoltage;

                /* the adaption component of the delta error of this */
                FloatType deltaErrorAdaption;

            public:

                /* constructor */
                AdaptiveLeakyIntegrateAndFireNeuron(
                    Interfaces::BasicNetworkOptions &opts,
                    FloatType baseThreshold,
                    FloatType thresholdIncreaseConstant,
                    FloatType adaptationTimeConstant,
                    FloatType refractoryPeriod,
                    FloatType membranTimeConstant,
                    FloatType derivativeDumpingFactor = 0.3
                ) : BasicNeuron(opts),
                    baseThreshold(baseThreshold),
                    thresholdIncreaseConstant(thresholdIncreaseConstant),
                    adaptationDecay(exp(-1.0 * opts.timeStepLength() / adaptationTimeConstant)),
                    voltageDecay(exp(-1.0 * opts.timeStepLength() / membranTimeConstant)),
                    refractoryPeriod(int(round(refractoryPeriod / opts.timeStepLength()))),
                    derivativeDumpingFactor(derivativeDumpingFactor) {
                    doReset();
                }

                /* resets this */
                void doReset() {
                    v = 0;
                    I = 0;
                    pseudoDerivative        = 0;
                    thresholdAdaptation     = 0;
                    timeStepsSinceLastSpike = 2 * refractoryPeriod;
                    adaptiveThreshold       = baseThreshold;
                    deltaErrorVoltage       = 0;
                    deltaErrorAdaption      = 0;
                }

                /* adds a input current to this */
                void addCurrent(FloatType current) {
                    I += current;
                }

                /* computes the next timestep */
                void doUpdate() {

                    bool fired = this->fired();

                    /* update voltage */
                    v = v * voltageDecay + I - int(fired) * baseThreshold;
                    I = 0;

                    /* update threshold adaptation value */
                    thresholdAdaptation = thresholdAdaptation * adaptationDecay + int(fired);
                    adaptiveThreshold = baseThreshold + thresholdIncreaseConstant * thresholdAdaptation;

                    /* update number of timesteps that have pased since the last spike */
                    timeStepsSinceLastSpike = fired ? 1 : timeStepsSinceLastSpike + 1;

                    /* pseudo-derivative is fixed to zero within the refractory period */
                    if (timeStepsSinceLastSpike > refractoryPeriod) {
                        pseudoDerivative = derivativeDumpingFactor * std::max(
                            0.0, 
                            1.0 - fabs((v - adaptiveThreshold) / baseThreshold)
                        );
                    } else 
                        pseudoDerivative = 0;
                    
                }

                /* should compute the voltage component of next delta error 
                 * (we need voltage component only since other components 
                 *  will get zero weighted during learn signal computations)
                 */
                FloatType computeDeltaErrorVoltage(FloatType learnSignal, FloatType derivative) {
                    deltaErrorAdaption = 
                        -1.0 * learnSignal * derivative * thresholdIncreaseConstant +
                        adaptationDecay * deltaErrorAdaption;

                    deltaErrorVoltage = learnSignal * derivative + voltageDecay * deltaErrorVoltage;
                    return deltaErrorVoltage;
                }

                virtual FloatType resetLearnSignal() {
                    return -1 * deltaErrorVoltage * baseThreshold + deltaErrorAdaption;
                }

                /* returns whether this neuron has fired during the current time step */
                bool fired() {
                    return timeStepsSinceLastSpike > refractoryPeriod && v > adaptiveThreshold;
                }

                /* returns the output of this Neuron */
                virtual FloatType getOutput() {
                    return fired() ? 1.0 : 0.0;
                }

                /* returns the pseudo-derivative of this */
                FloatType getDerivative() {
                    return pseudoDerivative;
                }

                /* should set the (pseudo) derivative of this */
                void setDerivative(FloatType pseudoDerivative) {
                    this->pseudoDerivative = pseudoDerivative;
                }

                /* returns the voltage decay factor */
                FloatType getVoltageDecayFactor() {
                    return voltageDecay;
                }

                /* returns the threshold adaptation decay factor */
                FloatType getThresholdAdaptionDecayFactor() {
                    return adaptationDecay;
                }

                /* returns the threshold increas constant (threshold adaption weight) */
                FloatType getThresholdIncreaseConstant() {
                    return thresholdIncreaseConstant;
                }

                /* returns the base threshold of this */
                FloatType getBaseThreshold() {
                    return baseThreshold;
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

                /* sets the delta error of this */
                virtual void doSetDeltaError(std::vector<FloatType> errors) {
                    deltaErrorVoltage  = errors[0];
                    deltaErrorAdaption = errors[1];
                }

                /* returns the threshold of this */
                FloatType getThreshold() { 
                    return adaptiveThreshold;
                }


        };

        /* type shortening */
        typedef AdaptiveLeakyIntegrateAndFireNeuron ALIFNeuron;

    }

}
#endif /* __ADAPTIVE_LEAKY_INTEGRATE_AND_FIRE_NEURON__ */
