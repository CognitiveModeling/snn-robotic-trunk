#ifndef __STOCHASTIC_GRADIENT_DESCENT_OPTIMIZER_H__
#define __STOCHASTIC_GRADIENT_DESCENT_OPTIMIZER_H__
#include "BasicOptimizer.h"
/**
 * simple stochastic gradient descent optimizer
 */
namespace SNN {

    namespace Optimizers {

        class StochasticGradientDescentOptimizer: public Interfaces::BasicOptimizer {

            private:

                /* the learn rate of this */
                FloatType learnRate;

                /* the initial leran rate of this */
                FloatType initialLearnRate;

                /* the leran rate decay factor */
                FloatType decayFactor;

                /* the number of updes after which the learnate decays */
                unsigned decayIntervall;

                /* the number of updates performed */
                unsigned updateCounter;

                /* should completly reset this */
                virtual void doReset() {

                    learnRate = initialLearnRate;
                    updateCounter = 0;
                }

            public:

                /* should return the learnrate */
                virtual FloatType getLearnRate() { return learnRate; }

                /* should set the learnrate */
                virtual void setLearnRate(FloatType learnrate) {
                    this->learnRate = learnrate;
                }

                /* should decay the leran rate of this */
                virtual void decayLearnRate(FloatType factor) {
                    learnRate *= factor;
                }

                /* calculates the weight change */
                virtual FloatType calcWeightChange() {
                    updateCounter++;
                    if (updateCounter % decayIntervall == 0)
                        learnRate *= decayFactor;

                    return -1.0 * this->gradient.getAccumulatedGradient() * learnRate;
                }

                /* constructor */
                StochasticGradientDescentOptimizer(
                    Interfaces::BasicGradient &gradient,
                    unsigned updateIntervall,
                    FloatType learnRate = 0.0001,
                    FloatType decayFactor = 1.0,
                    unsigned decayIntervall = 1000000
                ) : BasicOptimizer(gradient, updateIntervall), 
                    learnRate(learnRate),
                    initialLearnRate(learnRate), 
                    decayFactor(decayFactor),
                    decayIntervall(decayIntervall),
                    updateCounter(0) { }




        };

    }

}

#endif 
