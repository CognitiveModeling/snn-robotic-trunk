#ifndef __SIGN_DAMPED_MOMENTUM_OPTIMIZER_H__
#define __SIGN_DAMPED_MOMENTUM_OPTIMIZER_H__
#include "BasicOptimizer.h"
/**
 * simple stochastic gradient descent optimizer
 */
namespace SNN {

    namespace Optimizers {

        class SignDampedMomentumOptimizer: public Interfaces::BasicOptimizer {

            private:

                /* the learn rate of this */
                FloatType learnRate;

                /* the momentum of this */
                FloatType momentum;

                /* the initial leran rate of this */
                FloatType initialLearnRate;

                /* the leran rate decay factor */
                FloatType decayFactor;

                /* the number of updes after which the learnate decays */
                unsigned decayIntervall;

                /* the number of updates performed */
                unsigned updateCounter;

                /* the sign oscilation magnitude */
                FloatType signOscillationMagnitude;

                /* the decay rate for the oscilation magnitude */
                FloatType oscilationDecay;

                /* the last weigh change */
                FloatType lastWeightChange;
                
                /* runing average of gradient variance for normalization */
                FloatType variance;

                /* weight decay factor */
                FloatType weightDecay;

                /* should completly reset this */
                virtual void doReset() {

                    learnRate = initialLearnRate;
                    signOscillationMagnitude = (1.0 - oscilationDecay) / 2.0;
                    updateCounter = 0;
                    lastWeightChange = 0;
                    variance = 1;
                }

            public:

                /* should return the learnrate */
                virtual FloatType getLearnRate() { return learnRate; }

                /* should decay the leran rate of this */
                virtual void decayLearnRate(FloatType factor) {
                    learnRate *= factor;
                }

                /* should set the learnrate */
                virtual void setLearnRate(FloatType learnrate) {
                    this->learnRate = learnrate;
                }

                /* calculates the weight change */
                virtual FloatType calcWeightChange() {
                    updateCounter++;
                    if (updateCounter % decayIntervall == 0)
                        learnRate *= decayFactor;
                    
                    FloatType gradient = this->gradient.getAccumulatedGradient();
                    signOscillationMagnitude = oscilationDecay * signOscillationMagnitude + 
                                               (1.0 - oscilationDecay) * sgn(gradient);

                    variance = variance * 0.9 + 0.1 * pow(gradient, 2);
                    gradient /= sqrt(variance) + 1e-16;
                    FloatType weightChange = -1.0 * gradient * pow(signOscillationMagnitude, 2) * learnRate 
                                             + momentum * lastWeightChange;

                    lastWeightChange = weightChange;
                    return weightChange - learnRate * weightDecay * this->gradient.getWeight();
                }

                /* constructor */
                SignDampedMomentumOptimizer(
                    Interfaces::BasicGradient &gradient,
                    unsigned updateIntervall,
                    FloatType learnRate = 0.0001,
                    FloatType momentum  = 0.3,
                    FloatType decayFactor = 1.0,
                    unsigned decayIntervall = 1000000,
                    FloatType oscilationDecay = 0.9,
                    FloatType weightDecay = 0
                ) : BasicOptimizer(gradient, updateIntervall), 
                    learnRate(learnRate),
                    momentum(momentum),
                    initialLearnRate(learnRate), 
                    decayFactor(decayFactor),
                    decayIntervall(decayIntervall),
                    updateCounter(0),
                    signOscillationMagnitude((1.0 - oscilationDecay) / 2.0),
                    oscilationDecay(oscilationDecay), 
                    lastWeightChange(0),
                    variance(1), 
                    weightDecay(weightDecay) { this->doReset(); }




        };

    }

}

#endif 
