#ifndef __REALISTIC_ROBOT_ARM_V2_INFERENCE_NETWORK_H__
#define __REALISTIC_ROBOT_ARM_V2_INFERENCE_NETWORK_H__
#include "LongShortTermMemoryEligibilityNetwork.h"
#include "RobotArmInferenceNetworkOptions.h"
#include "RealisticRobotArmSimulationV2.h"

/* Network for prediction the poses of a simulated many joints robot arm */
namespace SNN {

    /* forward declarations */
    namespace Interfaces { class BasicOptimizer;         }
    namespace Gradients  { class CurrentInputGradient; }
    
    namespace Networks {

        class RealisticRobotArmV2InferenceNetwork: public LongShortTermMemoryEligibilityNetwork {

            private:

                /* the Options of this */
                Options::RobotArmInferenceNetworkOptions &opts;

                /* the actual robot arm simulation */
                std::shared_ptr<Visualizations::RealisticRobotArmSimulationV2> simulation;

                /* input error gradients for each joint and angle */
                std::vector<std::shared_ptr<Gradients::CurrentInputGradient>> xAngleGradients;
                std::vector<std::shared_ptr<Gradients::CurrentInputGradient>> yAngleGradients;
                std::vector<std::shared_ptr<Gradients::CurrentInputGradient>> heightGradients;

                /* input error gradient optimizers for each joint and angle */
                std::vector<std::shared_ptr<Interfaces::BasicOptimizer>> xAngleOptimizers;
                std::vector<std::shared_ptr<Interfaces::BasicOptimizer>> yAngleOptimizers;
                std::vector<std::shared_ptr<Interfaces::BasicOptimizer>> heightOptimizers;

                /* random number generator */
                rand128_t *rand;

                /* the initial random state */
                rand128_t initialRandomState;

                /* clears the train set of this */
                void clearTrainSet();

                /* initializes the train set of this */
                void initTrainSet();

                /* initializes the train set of for inference testing */
                void initTrainSetInference();

                /* calculate the prediction of the current inference pssition */
                void calculatePrediction();

                /* should perform aditional resets */
                virtual void doReset();

                /* returns the poison distributed next spike time for the given spike rate */
                unsigned nextSpikeTime(FloatType spikeRate);

                std::shared_ptr<Interfaces::BasicOptimizer> createInferenceOptimizer(
                    std::shared_ptr<Interfaces::BasicGradient> gradient
                );

                /* arm reset value */
                FloatType resetValue;

            public:

                /* constructor */
                RealisticRobotArmV2InferenceNetwork(
                    Options::RobotArmInferenceNetworkOptions &opts,
                    unsigned numAdaptiveInputSynapses = -1,
                    unsigned numAdaptiveHiddenSynapses = -1,
                    std::vector<FloatType> inputWeights      = std::vector<FloatType>(),
                    std::vector<FloatType> hiddenWeights     = std::vector<FloatType>(), 
                    std::vector<FloatType> outputWeights     = std::vector<FloatType>(), 
                    std::vector<FloatType> feedbackWeights   = std::vector<FloatType>(), 
                    std::vector<unsigned> inputWeightsIn     = std::vector<unsigned>(),
                    std::vector<unsigned> hiddenWeightsIn    = std::vector<unsigned>(), 
                    std::vector<unsigned> outputWeightsIn    = std::vector<unsigned>(), 
                    std::vector<unsigned> feedbackWeightsIn  = std::vector<unsigned>(), 
                    std::vector<unsigned> inputWeightsOut    = std::vector<unsigned>(), 
                    std::vector<unsigned> hiddenWeightsOut   = std::vector<unsigned>(), 
                    std::vector<unsigned> outputWeightsOut   = std::vector<unsigned>(), 
                    std::vector<unsigned> feedbackWeightsOut = std::vector<unsigned>() 
                );

                /* should return the network error */
                virtual FloatType getError();

                /* sets the initial random state vrom the given RealisticRobotArmV2InferenceNetwork */
                void setRandState(RealisticRobotArmV2InferenceNetwork &other) {
                    this->simulation->setInitialRandomState(other.simulation->getInitialRandomState());
                    this->setInitialRandomState(other.getInitialRandomState());
                    this->initialRandomState = other.initialRandomState;
                }

                /* sets the initial random state from the given random seed */
                void setRandState(uint64_t seed) {
                    rand128_t *randState = new_rand128_t(seed);
                    rand128_t *selfRand = new_rand128_t(rand128(randState));
                    rand128_t *lsnnRand = new_rand128_t(rand128(randState));
                    rand128_t *simulationRand = new_rand128_t(rand128(randState));

                    this->initialRandomState = *selfRand;
                    this->setInitialRandomState(*lsnnRand);
                    this->simulation->setInitialRandomState(*simulationRand);

                    free(randState);
                    free(selfRand);
                    free(lsnnRand);
                    free(simulationRand);
                }

                /* returns a simulation checksum (to check for reproducable training data) */
                FloatType getSimulationChecksum() {
                    return simulation->getChecksum();
                }

                /* run one backpropagated inference process */
                void runInference(unsigned epochs, std::string file = "");

                /* sets the simulation of this */
                void setSimulation(std::shared_ptr<Visualizations::RealisticRobotArmSimulationV2> simulation) {
                    this->simulation = simulation;
                }

        };

    }

}

#endif /* __REALISTIC_ROBOT_ARM_V2_INFERENCE_NETWORK_H__ */
