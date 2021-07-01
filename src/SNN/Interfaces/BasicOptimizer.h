#ifndef __BASIC_OPTIMIZER__
#define __BASIC_OPTIMIZER__
#include "BasicGradient.h"
/**
 * Interface for a gradient based network optimizer
 */
namespace SNN {

     namespace Interfaces {

         class BasicOptimizer {
            
            protected:

                /* the gradient estimator of this */
                BasicGradient &gradient;

                /* the update intervall of this */
                unsigned updateIntervall;

                /* the simulation time step counter */
                int timeStepCounter;

                /* learning time counter */
                int learningTimeCounter;

                /* should completly reset this */
                virtual void doReset() = 0;

                /* the initial sign of the synapse of this */
                FloatType sign;

            public:

                /* should calculate the weight change */
                virtual FloatType calcWeightChange() = 0;

                /* should decay the leran rate of this */
                virtual void decayLearnRate(FloatType factor) = 0;

                /* should return the learnrate */
                virtual FloatType getLearnRate() = 0;

                /* should set the learnrate */
                virtual void setLearnRate(FloatType) = 0;

                /* constructor */
                BasicOptimizer(BasicGradient &gradient, unsigned updateIntervall): 
                    gradient(gradient), 
                    updateIntervall(updateIntervall), 
                    timeStepCounter(0),
                    learningTimeCounter(0),
                    sign(sgn(gradient.getWeight())) { }


                /* virtual destructor for correct polymophism */
                virtual ~BasicOptimizer() { }

                /* updates this */
                void update() {

                    /* gradients have to be updated bevore this */
                    timeStepCounter++;
                    learningTimeCounter++;
                    assert(gradient.getSimulationTimeStep() == timeStepCounter);

                    if (learningTimeCounter % updateIntervall == 0) {
                        FloatType weightChange = this->calcWeightChange();
                        if (std::isnan(weightChange))
                            log_err("NaN encounterd in gradient update", LOG_W);
                        else if (this->getLearnRate() != 0)
                            gradient.getSynapse().addWeightChange(weightChange);
                        gradient.reset();
                    }
                }

                /* updates this (back propagation) */
                void backPropagate() {

                    /* gradients have to be updated bevore this */
                    timeStepCounter--;
                    learningTimeCounter++;
                    assert(gradient.getSimulationTimeStep() == timeStepCounter);
                    assert(timeStepCounter >= 0);

                    if (learningTimeCounter % updateIntervall == 0) {
                        assert(timeStepCounter == 0);
                        FloatType weightChange = this->calcWeightChange();
                        if (std::isnan(weightChange))
                            log_err("NaN encounterd in gradient update", LOG_W);
                        else if (this->getLearnRate() != 0)
                            gradient.getSynapse().addWeightChange(weightChange);
                        gradient.reset();
                    }
                }

                /* performs an optimizer calculation and returns the weigh change */
                FloatType computeChange() {
                    FloatType weightChange = this->calcWeightChange();
                    if (std::isnan(weightChange))
                        log_err("NaN encounterd in gradient update", LOG_W);
                    else if (this->getLearnRate() != 0)
                        return weightChange;

                    return 0;
                }

                /* resets the timeStepCounter of this */
                void networkReset() {
                    timeStepCounter = 0;
                }

                /* completly resets this */
                virtual void reset() {
                    this->doReset();
                    learningTimeCounter = 0;
                    timeStepCounter = 0;
                    sign = sgn(gradient.getWeight());
                }

                /* returns the current simulation time step of this */
                int getSimulationTimeStep() {
                    return timeStepCounter;
                }

                /* sets the current simulation time step of this */
                void setSimulationTimeStep(int timeStepCounter) {
                    this->learningTimeCounter += timeStepCounter - this->timeStepCounter;
                    this->timeStepCounter = timeStepCounter;
                }

                /* prepares counters for an backpropagation update */
                void prepareBackPropagationUpdate() {
                    this->learningTimeCounter = -1;
                    this->timeStepCounter = 1;
                }

                /* updates this */
                void updateGPU() {
                    FloatType weightChange = this->calcWeightChange();
                    if (std::isnan(weightChange))
                        log_err("NaN encounterd in gradient update", LOG_W);
                    else if (this->getLearnRate() != 0)
                        gradient.getSynapse().addWeightChange(weightChange);
                    gradient.reset();
                }

                /* updates this and resetts the gradients */
                void updateWeights() {
                    FloatType weightChange = this->calcWeightChange();
                    if (std::isnan(weightChange))
                        log_err("NaN encounterd in gradient update", LOG_W);
                    else if (this->getLearnRate() != 0)
                        gradient.getSynapse().addWeightChange(weightChange);
                    gradient.reset();
                }

                /* updates this ussing deep rewiring */
                bool updateDeepRewiring(FloatType noise) {

                    FloatType weight = sign * gradient.getSynapse().getWeight() + 
                                       sign * this->computeChange() + 
                                       this->getLearnRate() * noise;
                    
                    gradient.getSynapse().setWeight(weight * sign);
                    gradient.reset();
                    return weight > 0;
                }

                /* returns the gradient this optimizes */
                BasicGradient &getGradient() { return gradient; }
                
                /* should save neccesarry values for model restoring to the given file */
                virtual void save(int) { }

                /* should load neccesarry values for model restoring from the given file */
                virtual void load(int, bool) { }
         };

     }

}

#endif /* __BASIC_OPTIMIZER__ */
