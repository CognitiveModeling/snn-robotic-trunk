#ifndef __REALISTIC_ROBOT_ARM_V2_PREDICTOR_NETWORK_H__
#define __REALISTIC_ROBOT_ARM_V2_PREDICTOR_NETWORK_H__
#include "LongShortTermMemoryEligibilityNetwork.h"
#include "RobotArmPredictorNetworkOptions.h"
#include "RealisticRobotArmSimulationV2.h"

/* Network for prediction the poses of a simulated many joints robot arm */
namespace SNN {

    /* forward declarations */
    namespace Interfaces { class BasicOptimizer; }
    
    namespace Networks {

        struct SampleV2 {
            std::vector<FloatType> xAngle;
            std::vector<FloatType> yAngle;
            std::vector<FloatType> height;
            std::vector<FloatType> jointPosX; 
            std::vector<FloatType> jointPosY; 
            std::vector<FloatType> jointPosZ; 
            std::vector<FloatType> jointUpX; 
            std::vector<FloatType> jointUpY; 
            std::vector<FloatType> jointUpZ;
            std::vector<FloatType> jointXDirectionX; 
            std::vector<FloatType> jointXDirectionY; 
            std::vector<FloatType> jointXDirectionZ;
        };

        class RealisticRobotArmV2PredictorNetwork: public LongShortTermMemoryEligibilityNetwork {

            private:

                /* the Options of this */
                Options::RobotArmPredictorNetworkOptions &opts;

                /* the actual robot arm simulation */
                std::shared_ptr<Visualizations::RealisticRobotArmSimulationV2> simulation;

                /* angles one batch */
                std::vector<std::vector<FloatType>> xAngles;
                std::vector<std::vector<FloatType>> yAngles;
                std::vector<std::vector<FloatType>> heights;
                std::vector<std::vector<FloatType>> xPos, yPos, zPos;
                std::vector<std::vector<FloatType>> qwPos, qxPos, qyPos, qzPos;

                /* last angles for one simulation run */
                std::vector<FloatType> xAnglesLast;
                std::vector<FloatType> yAnglesLast;
                std::vector<FloatType> heightsLast;

                /* the latest error in euclidean distance */
                FloatType latestError;

                /* indicates that the error should be recalculated */
                bool recalculateError;

                /* random number generator */
                rand128_t *rand;

                /* the current epoch */
                unsigned epochCounter;

                /* simulation data file */
                FILE *simulationDataFile;
                unsigned dataStart, dataEnd;
                std::vector<SampleV2 *> trainset;
                unsigned trainsetIndex;

                /* initializes the train set of this */
                void initTrainSet();

                /* visualize the prediction */
                void visualizePrediction();
   
                /* calculate the prediction error */
                FloatType getDistanceError(bool denormalize = true);

                /* calculate the prediction error for orientation */
                FloatType getRotationError();

                /* should perform aditional resets */
                virtual void doReset();

            public:

                /* constructor */
                RealisticRobotArmV2PredictorNetwork(
                    Options::RobotArmPredictorNetworkOptions &opts,
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

                /* sets the simulation of this */
                void setSimulation(std::shared_ptr<Visualizations::RealisticRobotArmSimulationV2> simulation) {
                    this->simulation = simulation;
                }

                /* should return the network error */
                virtual FloatType getError();

                /* trains the network for the given number of epochs returns the averaged error 
                 * for the last 5% epochs */
                FloatType train(unsigned epochs, bool evaluate = false, int loglevel = LOG_D);

        };

    }

}

#endif /* __REALISTIC_ROBOT_ARM_V2_PREDICTOR_NETWORK_H__ */
