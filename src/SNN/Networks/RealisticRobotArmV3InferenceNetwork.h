#ifndef __REALISTIC_ROBOT_ARM_V3_INFERENCE_NETWORK_H__
#define __REALISTIC_ROBOT_ARM_V3_INFERENCE_NETWORK_H__
#include "LongShortTermMemoryEligibilityNetwork.h"
#include "RobotArmInferenceNetworkOptions.h"
#include "RealisticRobotArmSimulationV3.h"

/* Network for prediction the poses of a simulated many joints robot arm */
namespace SNN {

    /* forward declarations */
    namespace Interfaces { class BasicOptimizer;         }
    namespace Gradients  { class CurrentInputGradient; }
    
    namespace Networks {

        class RealisticRobotArmV3InferenceNetwork: public LongShortTermMemoryEligibilityNetwork {

            private:

                /* the Options of this */
                Options::RobotArmInferenceNetworkOptions &opts;

                /* the actual robot arm simulation */
                std::shared_ptr<Visualizations::RealisticRobotArmSimulationV3> simulation;

                /* input error gradients for each joint and angle */
                std::vector<std::shared_ptr<Gradients::CurrentInputGradient>> gearState1Gradients;
                std::vector<std::shared_ptr<Gradients::CurrentInputGradient>> gearState2Gradients;
                std::vector<std::shared_ptr<Gradients::CurrentInputGradient>> gearState3Gradients;

                /* input error gradient optimizers for each joint and angle */
                std::vector<std::shared_ptr<Interfaces::BasicOptimizer>> gearState1Optimizers;
                std::vector<std::shared_ptr<Interfaces::BasicOptimizer>> gearState2Optimizers;
                std::vector<std::shared_ptr<Interfaces::BasicOptimizer>> gearState3Optimizers;

                /* clears the train set of this */
                void clearTrainSet();

                /* initializes the train set of this */
                void initTrainSet();

                /* initializes the train set of for inference testing */
                void initTrainSetInference();

                /* calculate the prediction of the current inference pssition */
                void calculatePrediction();

                std::shared_ptr<Interfaces::BasicOptimizer> createInferenceOptimizer(
                    std::shared_ptr<Interfaces::BasicGradient> gradient
                );

                /* arm reset value */
                FloatType resetValue;

            public:

                /* constructor */
                RealisticRobotArmV3InferenceNetwork(
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

                /* run one backpropagated inference process */
                void runInference(unsigned epochs, std::string file = "");

                /* sets the simulation of this */
                void setSimulation(std::shared_ptr<Visualizations::RealisticRobotArmSimulationV3> simulation) {
                    this->simulation = simulation;
                }

        };

    }

}

#endif /* __REALISTIC_ROBOT_ARM_V3_INFERENCE_NETWORK_H__ */
