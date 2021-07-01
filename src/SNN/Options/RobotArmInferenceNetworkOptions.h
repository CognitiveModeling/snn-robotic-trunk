/**
 * global network options
 */
#ifndef __FULL_ROBOT_ARM_INFERENCE_NETWORK_OPTIONS_H__
#define __FULL_ROBOT_ARM_INFERENCE_NETWORK_OPTIONS_H__
#include "LongShortTermMemoryEligibilityNetworkOptions.h"

#include <assert.h>
#include <string>

namespace SNN {

    namespace Options {
        
        class RobotArmInferenceNetworkOptions: public LongShortTermMemoryEligibilityNetworkOptions {

            public:

            /* the number of joints for the roboter arm */
            addOption(unsigned, numJoints, 5);

            /* the max angle for any joint */
            addOption(FloatType, maxAngle, 20);

            /* wether to visualize the inferred possitions */
            addOption(bool, visualizeInference, false);

            /* wether to actualy visualize the ronto arm simulation */
            addOption(bool, visualizeRobotArm, false);

            /* wether to use uniform normalization for absolute positions or not */
            addOption(bool, uniformNormalization, true);

            /* wether to monitor training process */
            addOption(bool, monitorTraining, false);

            /* error weights for the different output types */
            addOption(FloatType, positionErrorWeight, 1);
            addOption(FloatType, upVectorErrorWeight, 1);
            addOption(FloatType, xDirectionErrorWeight, 1);

            /* wether to use a cloc input or not */
            addOption(bool, clockInput, false);

            /* the number of timesteps of the cloc input */
            addOption(unsigned, clockTimeSteps, 0);

            /* the number of cloc neurons */
            addOption(unsigned, numClockNeurons, 0);

            /* the spike rate of clock neurons in Hz */
            addOption(FloatType, clockNeuronSpikeRate, 1);

            /* the number of decoding timesteps fore on value */
            addOption(unsigned, decodingTimesteps, 1);

            /* wether to visualize inference gradients */
            addOption(bool, gradientMonitor, false);

            /* wether to visualize inference angles */
            addOption(bool, angleMonitor, false);

            /* wether to visualize inference input errors over time */
            addOption(bool, inputErrorMonitor, false);

            /* start and end neuron for input error monitor */
            addOption(unsigned, inputErrorMonitorStart, 0);
            addOption(unsigned, inputErrorMonitorEnd, 1);

            /* the optimizer type for the main loss */
            addOption(OptimizerType, inferenceOptimizer, OptimizerType::Adam);

            /* the learn rate */
            addOption(FloatType, inferenceLearnRate, 0.01)

            /* the momentum for SignDampedMomentumOptimizer */
            addOption(FloatType, inferenceMomentum, 0.1);

            /* the learn rate decay */
            addOption(FloatType, inferenceLearnRateDecay, 1.0)

            /* the learn rate decay intervall (in batches) */
            addOption(unsigned, inferenceLearnRateDecayIntervall, 100)

            /* the optimizer type for the regularizer loss */
            addOption(OptimizerType, inferenceRegularizerOptimizer, OptimizerType::Adam);

            /* the regularizer lern rate */
            addOption(FloatType, inferenceRegularizerLearnRate, 0.001);

            /* the momentum for SignDampedMomentumOptimizer (regularizer) */
            addOption(FloatType, inferenceRegularizerMomentum, 0.1);

            /* the learn rate decay for the regularizer */
            addOption(FloatType, inferenceRegularizerLearnRateDecay, 1.0)

            /* the learn rate decay intervall (in batches) for the regularizer */
            addOption(unsigned, inferenceRegularizerLearnRateDecayIntervall, 100)

            /* wether to add an output decay regularizer */
            addOption(bool, outputRegularizer, false);

            /* wether to use the input values directly as input curet to the nertwork */
            addOption(bool, currentInput, false);

            /* wether to start inference with as sraigh arm (zero angles) */
            addOption(bool, zeroStart, false);

            /* initial epochs in wich to perform relaxation */
            addOption(unsigned , relaxationEpochs, 250);

            /* initial relaxation learnrate */
            addOption(FloatType, relaxationLearnRate, 250);

            /* wether to use interactive target */
            addOption(bool, interactiveTarget, false);

            /* cmd inference target */
            addOption(bool, cmdTarget, false);
            addOption(FloatType, targetX, 0, false, true);
            addOption(FloatType, targetY, 0, false, true);
            addOption(FloatType, targetZ, 0, false, true);
            addOption(FloatType, targetXUp, 0, false, true);
            addOption(FloatType, targetYUp, 0, false, true);
            addOption(FloatType, targetZUp, 0, false, true);
            addOption(FloatType, targetXXDirection, 0, false, true);
            addOption(FloatType, targetYXDirection, 0, false, true);
            addOption(FloatType, targetZXDirection, 0, false, true);

            /* wether to send inferred inputs to a connected arduino */
            addOption(bool, arduino, false);

            /* the arduino port */
            addOption(std::string, arduinoPort, "");

            /* wether to use quaternions for rotation or not */
            addOption(bool, quaternionRotation, false);
        };
    }
}

#endif /* __FULL_ROBOT_ARM_INFERENCE_NETWORK_OPTIONS_H__ */
