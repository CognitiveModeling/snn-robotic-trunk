/**
 * global network options
 */
#ifndef __FULL_ROBOT_ARM_PREDICTOR_NETWORK_OPTIONS_H__
#define __FULL_ROBOT_ARM_PREDICTOR_NETWORK_OPTIONS_H__

#include <assert.h>

namespace SNN {

    namespace Options {
        
        class RobotArmPredictorNetworkOptions: public LongShortTermMemoryEligibilityNetworkOptions {

            public:

            /* the number of joints for the roboter arm */
            addOption(unsigned, numJoints, 5);

            /* wether to visualize the inferred possitions */
            addOption(bool, visualizeInference, false);

            /* wether to actualy visualize the ronto arm simulation */
            addOption(bool, visualizeRobotArm, false);

            /* error weights for the different output types */
            addOption(FloatType, positionErrorWeight, 1);
            addOption(FloatType, upVectorErrorWeight, 1);
            addOption(FloatType, xDirectionErrorWeight, 1);

            /* wether to use a cloc input or not */
            addOption(bool, clockInput, false);

            /* the number of timesteps of the cloc input */
            addOption(unsigned, clockTimeSteps, 0);

            /* the number of decoding timesteps fore on value */
            addOption(unsigned, decodingTimesteps, 1);

            /* the percentate of edje cases to include into the training */
            addOption(FloatType, edjeCasePercent, 0.25);

            /* wethet to use a smart trainset for training or not */
            addOption(bool, useSmartTrainSet, false);

            /* the max trainset size */
            addOption(unsigned, maxTrainSetSize, 30000);

            /* the initial train set size */
            addOption(unsigned, initialTrainSetSize, 3000);

            /* wether to save simulation data */
            addOption(bool, saveSimulation, false);
            addOption(bool, loadSimulation, false);
            addOption(std::string, simulationFile, "");

            /* wether to use quaternions for rotation or not */
            addOption(bool, quaternionRotation, false);

            /* wether to save the training poses in a simulation file */
            addOption(std::string, trainingPosesFile, "");

            /* wether to generate a dataset or not */
            addOption(bool, datasetGeneration, false);

        };
    }
}

#endif /* __FULL_ROBOT_ARM_PREDICTOR_NETWORK_OPTIONS_H__ */
