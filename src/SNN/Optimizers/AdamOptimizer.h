#ifndef __ADAM_OPTIMIZER_H__
#define __ADAM_OPTIMIZER_H__
#include "BasicOptimizer.h"
/**
 * adam optimizer
 */
namespace SNN {

    namespace Optimizers {

        class AdamOptimizer: public Interfaces::BasicOptimizer {

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

                /* the beta 1 and 2 parameters */
                FloatType beta1, beta2;

                /* epsilon parameter */
                FloatType epsilon;
                
                /* computational values */
                FloatType m, v;

                /* runing product of beta parameters */
                FloatType beta1Product, beta2Product;

                /* weight decay factor */
                FloatType weightDecay;


                /* should completly reset this */
                virtual void doReset() {

                    learnRate = initialLearnRate;
                    updateCounter = 0;
                    m = 0;
                    v = 0;
                    beta1Product = 1;
                    beta2Product = 1;
                }

            public:

                /* should return the learnrate */
                virtual FloatType getLearnRate() { return learnRate; }

                /* should set the learnrate */
                virtual void setLearnRate(FloatType learnrate) {
                    this->learnRate = learnrate;
                }

                /* calculates the weight change */
                virtual FloatType calcWeightChange() {
                    FloatType gradient = this->gradient.getAccumulatedGradient();
                    m = beta1 * m + (1 - beta1) * gradient;
                    v = beta2 * v + (1 - beta2) * pow(gradient, 2);

                    beta1Product *= beta1;
                    beta2Product *= beta2;

                    FloatType unbiased_m = m / (1 - beta1Product);
                    FloatType unbiased_v = v / (1 - beta2Product);

                    updateCounter++;
                    if (updateCounter % decayIntervall == 0)
                        learnRate *= decayFactor;

                    return -1.0 * learnRate * unbiased_m / (sqrt(unbiased_v) + epsilon) - 
                           learnRate * weightDecay * this->gradient.getWeight();
                }

                /* should decay the leran rate of this */
                virtual void decayLearnRate(FloatType factor) {
                    learnRate *= factor;
                }

                /* constructor */
                AdamOptimizer(
                    Interfaces::BasicGradient &gradient,
                    unsigned updateIntervall,
                    FloatType learnRate = 0.003,
                    FloatType decayFactor = 0.7,
                    unsigned decayIntervall = 100, 
                    FloatType beta1 = 0.9,
                    FloatType beta2 = 0.999,
                    FloatType epsilon = 1e-8,
                    FloatType weightDecay = 0
                ) : BasicOptimizer(gradient, updateIntervall), 
                    learnRate(learnRate), 
                    initialLearnRate(learnRate), 
                    decayFactor(decayFactor),
                    decayIntervall(decayIntervall),
                    updateCounter(0),
                    beta1(beta1), 
                    beta2(beta2),
                    epsilon(epsilon),
                    m(0),
                    v(0),
                    beta1Product(1),
                    beta2Product(1),
                    weightDecay(weightDecay) { }

                /* should save neccesarry values for model restoring to the given file */
                virtual void save(int fd) { 
                    writeValue(fd, &learnRate, sizeof(FloatType));
                    writeValue(fd, &decayFactor, sizeof(FloatType));
                    writeValue(fd, &decayIntervall, sizeof(unsigned));
                    writeValue(fd, &updateCounter, sizeof(unsigned));
                    writeValue(fd, &beta1, sizeof(FloatType));
                    writeValue(fd, &beta2, sizeof(FloatType));
                    writeValue(fd, &epsilon, sizeof(FloatType));
                    writeValue(fd, &m, sizeof(FloatType));
                    writeValue(fd, &v, sizeof(FloatType));
                    writeValue(fd, &beta1Product, sizeof(FloatType));
                    writeValue(fd, &beta2Product, sizeof(FloatType));
                }

                /* should load neccesarry values for model restoring from the given file */
                virtual void load(int fd, bool loadLearnRated) { 
                    FloatType oldLearnRate = this->learnRate;

                    readValue(fd, &learnRate, sizeof(FloatType));
                    readValue(fd, &decayFactor, sizeof(FloatType));
                    readValue(fd, &decayIntervall, sizeof(unsigned));
                    readValue(fd, &updateCounter, sizeof(unsigned));
                    readValue(fd, &beta1, sizeof(FloatType));
                    readValue(fd, &beta2, sizeof(FloatType));
                    readValue(fd, &epsilon, sizeof(FloatType));
                    readValue(fd, &m, sizeof(FloatType));
                    readValue(fd, &v, sizeof(FloatType));
                    readValue(fd, &beta1Product, sizeof(FloatType));
                    readValue(fd, &beta2Product, sizeof(FloatType));

                    if (!loadLearnRated)
                        this->learnRate = oldLearnRate;
                }

        };

    }

}

#endif 
