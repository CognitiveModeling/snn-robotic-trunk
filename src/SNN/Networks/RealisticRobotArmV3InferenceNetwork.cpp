#include "RealisticRobotArmV3InferenceNetwork.h"
#include "CurrentInputGradient.h"
#include "AdamOptimizer.h"
#include "AMSGradOptimizer.h"
#include "SignDampedMomentumOptimizer.h"
#include "StochasticGradientDescentOptimizer.h"
#include "FeedbackWeightsSerializer.h"
#include "BasicSynapseSerializer.h"
#include "arduino-serial-lib.h"

static void sendByte(int arduino, uint8_t byte) {

    int code = -1;
    uint8_t received = ~byte;
    while (code != 0 || byte != received) {
        if (serialport_writebyte(arduino, byte))
            serialport_flush(arduino);
        
        code = serialport_readbyte(arduino, &received, 100);
        //printf("sendByte[%d]: %02x / %02x\n", code, byte, received);
    }
}

static void sendComand(int arduino, float pX, float pY, float pL, int pIndex) {

    uint8_t x = uint8_t(roundf((pX + 1) * 127.5));
    uint8_t y = uint8_t(roundf((pY + 1) * 127.5));
    uint8_t l = uint8_t(roundf((pL + 1) * 127.5));
    uint8_t index = pIndex;
    //printf("send comand: (%f, %f, %f, %d) => (%u, %u, %u, %u)\n", pX, pY, pL, pIndex, x, y, l, index);

    sendByte(arduino, 0x10 | (x & 0x0F));
    sendByte(arduino, 0x20 | ((x >> 4) & 0x0F));

    sendByte(arduino, 0x30 | (y & 0x0F));
    sendByte(arduino, 0x40 | ((y >> 4) & 0x0F));

    sendByte(arduino, 0x50 | (l & 0x0F));
    sendByte(arduino, 0x60 | ((l >> 4) & 0x0F));

    sendByte(arduino, 0x70 | (index & 0x0F));
    sendByte(arduino, 0xF0);
}


using namespace SNN;
using namespace Networks;
using namespace Gradients;
using namespace Optimizers;
using namespace Interfaces;
using namespace Options;
using namespace Visualizations;

using std::string;
using std::vector;
using std::shared_ptr;
using std::make_shared;
using std::static_pointer_cast;

/* constructor */
RealisticRobotArmV3InferenceNetwork::RealisticRobotArmV3InferenceNetwork(
    RobotArmInferenceNetworkOptions &opts,
    unsigned numAdaptiveInputSynapses,
    unsigned numAdaptiveHiddenSynapses,
    std::vector<FloatType> inputWeights,
    std::vector<FloatType> hiddenWeights,
    std::vector<FloatType> outputWeights,
    std::vector<FloatType> feedbackWeights,
    std::vector<unsigned> inputWeightsIn,
    std::vector<unsigned> hiddenWeightsIn,
    std::vector<unsigned> outputWeightsIn,
    std::vector<unsigned> feedbackWeightsIn,
    std::vector<unsigned> inputWeightsOut,
    std::vector<unsigned> hiddenWeightsOut,
    std::vector<unsigned> outputWeightsOut,
    std::vector<unsigned> feedbackWeightsOut
) : LongShortTermMemoryEligibilityNetwork(
        static_cast<LongShortTermMemoryEligibilityNetworkOptions &>(opts),
        numAdaptiveInputSynapses,
        numAdaptiveHiddenSynapses,
        inputWeights,
        hiddenWeights,
        outputWeights,
        feedbackWeights,
        inputWeightsIn,
        hiddenWeightsIn,
        outputWeightsIn,
        feedbackWeightsIn,
        inputWeightsOut,
        hiddenWeightsOut,
        outputWeightsOut,
        feedbackWeightsOut
    ),
    opts(opts) {

    this->resetValue = 0;

    if (!opts.quaternionRotation()) {
        this->targetWeightAt(0) = opts.positionErrorWeight();
        this->targetWeightAt(1) = opts.positionErrorWeight();
        this->targetWeightAt(2) = opts.positionErrorWeight();

        this->targetWeightAt(3) = opts.upVectorErrorWeight();
        this->targetWeightAt(4) = opts.upVectorErrorWeight();
        this->targetWeightAt(5) = opts.upVectorErrorWeight();

        this->targetWeightAt(6) = opts.xDirectionErrorWeight();
        this->targetWeightAt(7) = opts.xDirectionErrorWeight();
        this->targetWeightAt(8) = opts.xDirectionErrorWeight();
    }

    FloatType timePerValue = opts.numSimulationTimesteps() / FloatType(opts.numJoints());
    for (unsigned j = 0; j < opts.numJoints(); j++) {
        gearState1Gradients.push_back(
            make_shared<CurrentInputGradient>(
                j * timePerValue,
                (j + 1) * timePerValue - 1,
                0,
                0,
                opts.numInputNeurons(),
                this->inputErrorsOverTime
            )
        );
        gearState2Gradients.push_back(
            make_shared<CurrentInputGradient>(
                j * timePerValue,
                (j + 1) * timePerValue - 1,
                1,
                1,
                opts.numInputNeurons(),
                this->inputErrorsOverTime
            )
        );
        gearState3Gradients.push_back(
            make_shared<CurrentInputGradient>(
                j * timePerValue,
                (j + 1) * timePerValue - 1,
                2,
                2,
                opts.numInputNeurons(),
                this->inputErrorsOverTime
            )
        );

        gearState1Optimizers.push_back(this->createInferenceOptimizer(static_pointer_cast<BasicGradient>(gearState1Gradients.back())));
        gearState2Optimizers.push_back(this->createInferenceOptimizer(static_pointer_cast<BasicGradient>(gearState2Gradients.back())));
        gearState3Optimizers.push_back(this->createInferenceOptimizer(static_pointer_cast<BasicGradient>(gearState3Gradients.back())));
    }
}

shared_ptr<BasicOptimizer> RealisticRobotArmV3InferenceNetwork::createInferenceOptimizer(
    shared_ptr<BasicGradient> gradient
) {
    OptimizerType type               = opts.inferenceOptimizer();
    FloatType learnRate              = opts.inferenceLearnRate();
    FloatType momentum               = opts.inferenceMomentum();
    FloatType learnRateDecay         = opts.inferenceLearnRateDecay();
    unsigned learnRateDecayIntervall = opts.inferenceLearnRateDecayIntervall();

    if (type == OptimizerType::StochasticGradientDescent) {
        return static_pointer_cast<BasicOptimizer>(
            make_shared<StochasticGradientDescentOptimizer>(
                *gradient, 
                opts.optimizerUpdateInterval(),
                learnRate,
                learnRateDecay,
                learnRateDecayIntervall
            )
        );
    } else if (type == OptimizerType::Adam) {
        return static_pointer_cast<BasicOptimizer>(
            make_shared<AdamOptimizer>(
                *gradient, 
                opts.optimizerUpdateInterval(),
                learnRate,
                learnRateDecay,
                learnRateDecayIntervall,
                opts.adamBetaOne(),
                opts.adamBetaTwo()
            )
        );
    } else if (type == OptimizerType::AMSGrad) {
        return static_pointer_cast<BasicOptimizer>(
            make_shared<AMSGradOptimizer>(
                *gradient, 
                opts.optimizerUpdateInterval(),
                learnRate,
                learnRateDecay,
                learnRateDecayIntervall,
                opts.adamBetaOne(),
                opts.adamBetaTwo()
            )
        );
    } else if (type == OptimizerType::SignDampedMomentum) {
        return static_pointer_cast<BasicOptimizer>(
            make_shared<SignDampedMomentumOptimizer>(
                *gradient, 
                opts.optimizerUpdateInterval(),
                learnRate,
                momentum,
                learnRateDecay,
                learnRateDecayIntervall
            )
        );
    }

    assert(false);
    return nullptr;
}

/* clears the train set of this */
void RealisticRobotArmV3InferenceNetwork::clearTrainSet() {

    for (unsigned b = 0; b < opts.trainSetSize(); b++) {
        for (unsigned t = 0; t < opts.numSimulationTimesteps(); t++) {
            for (unsigned n = 0; n < opts.numInputNeurons(); n++)
                this->inputAt(b, t, n) = 0;
            
            for (unsigned n = 0; n < opts.numOutputNeurons(); n++)
                    this->targetAt(b, t, n) = 0;

            this->errorMaskAt(b, t) = 0;
        }
    }
}

/* initializes the train set of this */
void RealisticRobotArmV3InferenceNetwork::initTrainSet() {

    clearTrainSet();
    FloatType gearState1 = 0, gearState2 = 0, gearState3 = 0;
    FloatType jointPosX = 0, jointPosY = 0, jointPosZ = 0;
    FloatType jointUpX = 0, jointUpY = 0, jointUpZ = 0;
    FloatType jointXDirectionX = 0, jointXDirectionY = 0, jointXDirectionZ = 0;
    FloatType timePerValue = opts.numSimulationTimesteps() / FloatType(opts.numJoints());
    FloatType clockTime = opts.clockTimeSteps();

    for (unsigned j = 0; j < opts.numJoints(); j++) {
        simulation->getInferredGearStates(j, gearState1, gearState2, gearState3);

        simulation->getInferredPosition(
            j, 
            jointPosX, 
            jointPosY, 
            jointPosZ, 
            jointUpX, 
            jointUpY, 
            jointUpZ,
            jointXDirectionX, 
            jointXDirectionY, 
            jointXDirectionZ
        );

        if (opts.numInputNeurons() > 3 + j) {
            for (unsigned t = timePerValue * (j + 1) - clockTime; t < timePerValue * (j + 1); t++) {
                this->inputAt(0, t, 3 + j) = 1;
            }
        }
        for (unsigned t = timePerValue * j; t < timePerValue * (j + 1); t++) {
            this->inputAt(0, t, 0) = gearState1;
            this->inputAt(0, t, 1) = gearState2;
            this->inputAt(0, t, 2) = gearState3;
        }

        for (unsigned t = (j + 1) * timePerValue - opts.decodingTimesteps();
             t < (j + 1) * timePerValue; t++) {

            this->targetAt(0, t, 0) = jointPosX;
            this->targetAt(0, t, 1) = jointPosY;
            this->targetAt(0, t, 2) = jointPosZ;
            if (opts.quaternionRotation()) {
                FloatType qw, qx, qy, qz;
                toQuaternion(
                    jointUpX, 
                    jointUpY, 
                    jointUpZ,
                    jointXDirectionX, 
                    jointXDirectionY, 
                    jointXDirectionZ,
                    qw, qx, qy, qz
                );
                this->targetAt(0, t, 3) = qw;
                this->targetAt(0, t, 4) = qx;
                this->targetAt(0, t, 5) = qy;
                this->targetAt(0, t, 6) = qz;
            } else {
                this->targetAt(0, t, 3) = jointUpX;
                this->targetAt(0, t, 4) = jointUpY;
                this->targetAt(0, t, 5) = jointUpZ;
                this->targetAt(0, t, 6) = jointXDirectionX;
                this->targetAt(0, t, 7) = jointXDirectionY;
                this->targetAt(0, t, 8) = jointXDirectionZ;
            }
            this->errorMaskAt(0, t) = 1;
        }
    }
}

/* correct for prediction errors and initialize inference trainset */
void RealisticRobotArmV3InferenceNetwork::initTrainSetInference() {

    clearTrainSet();
    FloatType timePerValue = opts.numSimulationTimesteps() / FloatType(opts.numJoints());
    unsigned numOutputs = opts.numOutputNeurons();
    FloatType clockTime = opts.decodingTimesteps();

    FloatType targetX = 0, targetY = 0, targetZ = 0;
    FloatType targetUpX = 0, targetUpY = 0, targetUpZ = 0;
    FloatType targetXDirectionX = 0, targetXDirectionY = 0, targetXDirectionZ = 0;
    FloatType gearState1, gearState2, gearState3;

    simulation->getPosition(
        opts.numJoints() - 1, 
        targetX, 
        targetY, 
        targetZ, 
        targetUpX, 
        targetUpY, 
        targetUpZ,
        targetXDirectionX, 
        targetXDirectionY, 
        targetXDirectionZ
    );

    if (opts.cmdTarget()) {
        targetX = opts.targetX();
        targetY = opts.targetY();
        targetZ = opts.targetZ();
        targetUpX = opts.targetXUp();
        targetUpY = opts.targetYUp();
        targetUpZ = opts.targetZUp();
        targetXDirectionX = opts.targetXXDirection();
        targetXDirectionY = opts.targetYXDirection();
        targetXDirectionZ = opts.targetZXDirection();
        simulation->setInferenceTarget(
            targetX, 
            targetY, 
            targetZ, 
            targetUpX, 
            targetUpY, 
            targetUpZ,
            targetXDirectionX, 
            targetXDirectionY, 
            targetXDirectionZ
        );
    }


    if (opts.interactiveTarget()) {
        simulation->getInferenceTarget(
            targetX, 
            targetY, 
            targetZ, 
            targetUpX, 
            targetUpY, 
            targetUpZ,
            targetXDirectionX, 
            targetXDirectionY, 
            targetXDirectionZ
        );

        this->resetValue = 0;;

        string key = simulation->getKey();
        if (key == "Left") 
            targetX -= 1;
        else if (key == "Right") 
            targetX += 1;
        else if (key == "Up") 
            targetY += 1;
        else if (key == "Down") 
            targetY -= 1;
        else if (key == "a") 
            targetZ -= 1;
        else if (key == "d") 
            targetZ += 1;
        else if (key == "z")
            this->resetValue = 1;
        else if (key != "")
            printf("unknown key: %s\n", key.c_str());

        simulation->setInferenceTarget(
            targetX, 
            targetY, 
            targetZ, 
            targetUpX, 
            targetUpY, 
            targetUpZ,
            targetXDirectionX, 
            targetXDirectionY, 
            targetXDirectionZ
        );
    }

    FloatType actualX = 0, actualY = 0, actualZ = 0;
    FloatType actualUpX = 0, actualUpY = 0, actualUpZ = 0;
    FloatType actualXDirectionX = 0, actualXDirectionY = 0, actualXDirectionZ = 0;

    for (unsigned j = 0; j < opts.numJoints(); j++) {
        simulation->getInferredGearStates(j, gearState1, gearState2, gearState3);

        for (unsigned t = timePerValue * j; t < timePerValue * (j + 1); t++) {
            this->inputAt(0, t, 0) = gearState1;
            this->inputAt(0, t, 1) = gearState2;
            this->inputAt(0, t, 2) = gearState3;
        }

        if (opts.numInputNeurons() > 3 + j) {
            for (unsigned t = timePerValue * (j + 1) - clockTime; t < timePerValue * (j + 1); t++) {
                this->inputAt(0, t, 3 + j) = 1;
            }
        }
    }

    simulation->getInferredPosition(
        opts.numJoints() - 1, 
        actualX, 
        actualY, 
        actualZ, 
        actualUpX, 
        actualUpY, 
        actualUpZ,
        actualXDirectionX, 
        actualXDirectionY, 
        actualXDirectionZ
    );


    FloatType predictedX = 0, predictedY = 0, predictedZ = 0;
    FloatType predictedUpX = 0, predictedUpY = 0, predictedUpZ = 0;
    FloatType predictedXDirectionX = 0, predictedXDirectionY = 0, predictedXDirectionZ = 0;
    FloatType qw = 0, qx = 0, qy = 0, qz = 0;

    FloatType *outputsOverTime = this->getGPUOutput(0);
    for (unsigned t = opts.numJoints() * timePerValue - opts.decodingTimesteps();
         t < opts.numJoints() * timePerValue; t++) {

        predictedX += outputsOverTime[t * numOutputs + 0];
        predictedY += outputsOverTime[t * numOutputs + 1];
        predictedZ += outputsOverTime[t * numOutputs + 2];
        if (opts.quaternionRotation()) {
            qw += outputsOverTime[t * numOutputs + 3];
            qx += outputsOverTime[t * numOutputs + 4];
            qy += outputsOverTime[t * numOutputs + 5];
            qz += outputsOverTime[t * numOutputs + 6];
        } else {
            predictedUpX         += outputsOverTime[t * numOutputs + 3];
            predictedUpY         += outputsOverTime[t * numOutputs + 4];
            predictedUpZ         += outputsOverTime[t * numOutputs + 5];
            predictedXDirectionX += outputsOverTime[t * numOutputs + 6];
            predictedXDirectionY += outputsOverTime[t * numOutputs + 7];
            predictedXDirectionZ += outputsOverTime[t * numOutputs + 8];
        }
    }

    predictedX /= opts.decodingTimesteps();
    predictedY /= opts.decodingTimesteps();
    predictedZ /= opts.decodingTimesteps();
    if (opts.quaternionRotation()) {
        toVectors(
            predictedUpX,
            predictedUpY,
            predictedUpZ,
            predictedXDirectionX,
            predictedXDirectionY,
            predictedXDirectionZ,
            qw / opts.decodingTimesteps(), 
            qx / opts.decodingTimesteps(), 
            qy / opts.decodingTimesteps(), 
            qz / opts.decodingTimesteps()
        );
    } else {
        predictedUpX         /= opts.decodingTimesteps();
        predictedUpY         /= opts.decodingTimesteps();
        predictedUpZ         /= opts.decodingTimesteps();
        predictedXDirectionX /= opts.decodingTimesteps();
        predictedXDirectionY /= opts.decodingTimesteps();
        predictedXDirectionZ /= opts.decodingTimesteps();
    }


    /* set the corrected target possition for the predictor */
    for (unsigned t = opts.numJoints() * timePerValue - opts.decodingTimesteps();
         t < opts.numJoints() * timePerValue; t++) {
        
        this->targetAt(0, t, 0) = predictedX + (targetX - actualX);
        this->targetAt(0, t, 1) = predictedY + (targetY - actualY);
        this->targetAt(0, t, 2) = predictedZ + (targetZ - actualZ);
		if (opts.quaternionRotation()) {
			FloatType qw, qx, qy, qz;
			toQuaternion(
				predictedUpX + (targetUpX - actualUpX), 
				predictedUpY + (targetUpY - actualUpY), 
				predictedUpZ + (targetUpZ - actualUpZ), 
				predictedXDirectionX + (targetXDirectionX - actualXDirectionX), 
				predictedXDirectionY + (targetXDirectionY - actualXDirectionY), 
				predictedXDirectionZ + (targetXDirectionZ - actualXDirectionZ), 
				qw, qx, qy, qz
			);
			this->targetAt(0, t, 3) = qw;
			this->targetAt(0, t, 4) = qx;
			this->targetAt(0, t, 5) = qy;
			this->targetAt(0, t, 6) = qz; 
		} else {
			this->targetAt(0, t, 3) = predictedUpX + (targetUpX - actualUpX);
			this->targetAt(0, t, 4) = predictedUpY + (targetUpY - actualUpY); 
			this->targetAt(0, t, 5) = predictedUpZ + (targetUpZ - actualUpZ); 
			this->targetAt(0, t, 6) = predictedXDirectionX + (targetXDirectionX - actualXDirectionX); 
			this->targetAt(0, t, 7) = predictedXDirectionY + (targetXDirectionY - actualXDirectionY); 
			this->targetAt(0, t, 8) = predictedXDirectionZ + (targetXDirectionZ - actualXDirectionZ); 
		}
        this->errorMaskAt(0, t) = 1;
    }
}

/* run one backpropagated inference process */
void RealisticRobotArmV3InferenceNetwork::runInference(unsigned epochs, string file) {

    int fd = -1;
    if (file != "")
        fd = open(file.c_str(), O_WRONLY|O_CREAT|O_TRUNC, 00777);

    if (fd > 0) {
        assert(write(fd, &epochs, sizeof(unsigned)) == sizeof(unsigned));

        unsigned numJoints = opts.numJoints();
        assert(write(fd, &numJoints, sizeof(unsigned)) == sizeof(unsigned));
    }

    int arduino = 0;
    if (opts.arduino()) {
        printf("opening arduino at %s\n", opts.arduinoPort().c_str());
        arduino = serialport_init(opts.arduinoPort().c_str(), 115200);
        serialport_flush(arduino);
    }

    for (auto &optimizer: hiddenOptimizers) 
        optimizer->setLearnRate(0);
    for (auto &optimizer: inputOptimizers) 
        optimizer->setLearnRate(0);
    for (auto &optimizer: hiddenRegularizers) 
        optimizer->setLearnRate(0);
    for (auto &optimizer: inputRegularizers) 
        optimizer->setLearnRate(0);

    printf("opts.batchSize() = %u\n", opts.batchSize());
    printf("opts.trainSetSize() = %u\n", opts.trainSetSize());

    FloatType minInferenceError = 1000000, minInferenceOrientationError = 1000000, avgPredictionError = 0, minNMSE = 1000000;
    FloatType initialError = -1;
    FloatType avgSum = 0;
    FloatType regularizeFactor = opts.inferenceRegularizerLearnRate();

    assert(opts.useInputErrors());
    this->reset();


    if (opts.interactiveTarget()) {
        for (unsigned j = 0; j < opts.numJoints(); j++) 
            simulation->setInferredGearStates(j, 0, 0, 0);

        FloatType jointPosX = 0, jointPosY = 0, jointPosZ = 0;
        FloatType jointUpX = 0, jointUpY = 0, jointUpZ = 0;
        FloatType jointXDirectionX = 0, jointXDirectionY = 0, jointXDirectionZ = 0;
        
        simulation->getInferredPosition(
            opts.numJoints() - 1, 
            jointPosX, 
            jointPosY, 
            jointPosZ, 
            jointUpX, 
            jointUpY, 
            jointUpZ,
            jointXDirectionX, 
            jointXDirectionY, 
            jointXDirectionZ
        );
        simulation->setInferenceTarget(
            jointPosX, 
            jointPosY, 
            jointPosZ, 
            jointUpX, 
            jointUpY, 
            jointUpZ,
            jointXDirectionX, 
            jointXDirectionY, 
            jointXDirectionZ
        );
    } else {

        simulation->randomPose();

        /* choose a random start pose for the inference */
        for (unsigned j = 0; j < opts.numJoints(); j++) {
            if (opts.zeroStart()) {
                simulation->setInferredGearStates(j, 0, 0, 0);
            } else {
                FloatType gearState1, gearState2, gearState3;
                simulation->getGearStates(j, gearState1, gearState2, gearState3);
                simulation->setInferredGearStates(j, gearState1, gearState2, gearState3);
            }
        }

        /* choose a random target pose for the inference */
        simulation->randomPose();
    }

    for (unsigned epoch = 0; epoch < epochs; epoch++) {

        this->initTrainSet();

        if (resetValue > 0) {

            for (float r = 0.1; r <= 1.001; r += 0.1) {
                for (unsigned j = 0; j < opts.numJoints(); j++) {
                    FloatType gearState1, gearState2, gearState3;
                    simulation->getInferredGearStates(j, gearState1, gearState2, gearState3);
                    gearState1 *= 1 - r;
                    gearState2 *= 1 - r;
                    gearState3 *= 1 - r;

                    if (opts.arduino() && arduino > 0) 
                        sendComand(arduino, gearState1, gearState2, gearState3, j);
                }
                printf("resetting: %f\n", r);
                usleep(10000);
                resetValue = 0;
            }

            for (unsigned j = 0; j < opts.numJoints(); j++) {
                simulation->setInferredGearStates(j, 0, 0, 0);
                gearState1Optimizers[j]->reset();
                gearState2Optimizers[j]->reset();
                gearState3Optimizers[j]->reset();
            }

            FloatType jointPosX = 0, jointPosY = 0, jointPosZ = 0;
            FloatType jointUpX = 0, jointUpY = 0, jointUpZ = 0;
            FloatType jointXDirectionX = 0, jointXDirectionY = 0, jointXDirectionZ = 0;
            
            simulation->getInferredPosition(
                opts.numJoints() - 1, 
                jointPosX, 
                jointPosY, 
                jointPosZ, 
                jointUpX, 
                jointUpY, 
                jointUpZ,
                jointXDirectionX, 
                jointXDirectionY, 
                jointXDirectionZ
            );
            simulation->setInferenceTarget(
                jointPosX, 
                jointPosY, 
                jointPosZ, 
                jointUpX, 
                jointUpY, 
                jointUpZ,
                jointXDirectionX, 
                jointXDirectionY, 
                jointXDirectionZ
            );
        }

        this->updateGPU();

        /* regularize for smal predictionerror */
        for (unsigned j = 0; j < opts.numJoints(); j++) {
            gearState1Gradients[j]->setScallingFactor(regularizeFactor);
            gearState2Gradients[j]->setScallingFactor(regularizeFactor);
            gearState3Gradients[j]->setScallingFactor(regularizeFactor);

            gearState1Gradients[j]->update();
            gearState2Gradients[j]->update();
            gearState3Gradients[j]->update();
            gearState1Gradients[j]->networkReset();
            gearState2Gradients[j]->networkReset();
            gearState3Gradients[j]->networkReset();

            gearState1Gradients[j]->setScallingFactor(1);
            gearState2Gradients[j]->setScallingFactor(1);
            gearState3Gradients[j]->setScallingFactor(1);
        }

        this->initTrainSetInference();
        this->updateGPU();
        this->calculatePrediction();

        for (unsigned j = 0; j < opts.numJoints(); j++) {
            gearState1Gradients[j]->update();
            gearState2Gradients[j]->update();
            gearState3Gradients[j]->update();
            gearState1Gradients[j]->networkReset();
            gearState2Gradients[j]->networkReset();
            gearState3Gradients[j]->networkReset();

            FloatType gearState1, gearState2, gearState3;
            simulation->getInferredGearStates(j, gearState1, gearState2, gearState3);
            
            gearState1 += gearState1Optimizers[j]->calcWeightChange();
            gearState2 += gearState2Optimizers[j]->calcWeightChange();
            gearState3 += gearState3Optimizers[j]->calcWeightChange();

            if (gearState1 < 0.0) gearState1 = 0.0;
            if (gearState1 > 1.0) gearState1 = 1.0;

            if (gearState2 < 0.0) gearState2 = 0.0;
            if (gearState2 > 1.0) gearState2 = 1.0;

            if (gearState3 < 0.0) gearState3 = 0.0;
            if (gearState3 > 1.0) gearState3 = 1.0;

            //printf("gradients %u: (%5.2f, %5.2f, %5.2f), angles: (%5.2f, %5.2f, %5.2f)\n", j, 
            //    gearState1Gradients[j]->getAccumulatedGradient(), 
            //    gearState2Gradients[j]->getAccumulatedGradient(), 
            //    gearState3Gradients[j]->getAccumulatedGradient(),
            //    gearState1, gearState2, gearState3
            //);
            gearState1Gradients[j]->reset();
            gearState2Gradients[j]->reset();
            gearState3Gradients[j]->reset();
            simulation->setInferredGearStates(j, gearState1, gearState2, gearState3);

            if (opts.arduino() && arduino > 0)
                sendComand(arduino, gearState1, gearState2, gearState3, j);
        }

        FloatType inferenceError  = simulation->getInferenceError();
        FloatType inferenceOrientationError  = simulation->getInferenceOrientationError();
        FloatType predictionError = simulation->getInferencePredictionError();

        minInferenceError  = std::min(minInferenceError, inferenceError);
        minInferenceOrientationError = std::min(minInferenceOrientationError, inferenceOrientationError);
        minNMSE            = std::min(minNMSE, this->getNormalizedMeanSquaredError());
        avgPredictionError = avgPredictionError * 0.99 + predictionError;
        avgSum             = avgSum * 0.99 + 1;

        if (initialError < 0)
            initialError = inferenceError;
 
        for (unsigned j = 0; j < opts.numJoints(); j++) {
            FloatType inferenceLearnRate = opts.inferenceLearnRate() * log(1 + minInferenceError) / log(1 + initialError);
            regularizeFactor = opts.inferenceRegularizerLearnRate() * log(1 + minInferenceError) / log(1 + initialError);
            gearState1Optimizers[j]->setLearnRate(inferenceLearnRate);
            gearState2Optimizers[j]->setLearnRate(inferenceLearnRate);
            gearState3Optimizers[j]->setLearnRate(inferenceLearnRate);
        }

        log_str(
            "Epoch: " + itoa(epoch) + ", Inference-Error: " + ftoa(inferenceError) + 
            " / " + ftoa(minInferenceError) + ", Orientation-Error: " + ftoa(inferenceOrientationError) + 
            " / " + ftoa(minInferenceOrientationError) + ", Prediction-Error: " + ftoa(predictionError) + " / " +
            ftoa(avgPredictionError / avgSum) + ", NMSE: " + ftoa(this->getNormalizedMeanSquaredError()) + " / " +
            ftoa(minNMSE) + ", inference-lr: " + ftoa(gearState1Optimizers[0]->getLearnRate(), 5) + 
            ", regularizer: " + ftoa(regularizeFactor, 5) + 
            ", reset: " + ftoa(resetValue)
        );
        
        if (fd > 0) {
            simulation->saveState(fd);
        }
    }

    FloatType targetX = 0, targetY = 0, targetZ = 0;
    FloatType targetUpX = 0, targetUpY = 0, targetUpZ = 0;
    FloatType targetXDirectionX = 0, targetXDirectionY = 0, targetXDirectionZ = 0;
    FloatType gearState1, gearState2, gearState3;

    simulation->getPosition(
        opts.numJoints() - 1, 
        targetX, 
        targetY, 
        targetZ, 
        targetUpX, 
        targetUpY, 
        targetUpZ,
        targetXDirectionX, 
        targetXDirectionY, 
        targetXDirectionZ
    );
    if (opts.cmdTarget()) {
        targetX = opts.targetX();
        targetY = opts.targetY();
        targetZ = opts.targetZ();
        targetUpX = opts.targetXUp();
        targetUpY = opts.targetYUp();
        targetUpZ = opts.targetZUp();
        targetXDirectionX = opts.targetXXDirection();
        targetXDirectionY = opts.targetYXDirection();
        targetXDirectionZ = opts.targetZXDirection();
    }
    printf("target: (%f, %f, %f)\n", targetX, targetY, targetZ);
    for (unsigned j = 0; j < opts.numJoints(); j++) {
        simulation->getInferredGearStates(j, gearState1, gearState2, gearState3);
        if (j < opts.numJoints() - 1)
            printf("{%f, %f, %f}, ", gearState1, gearState2, gearState3);
        else
            printf("{%f, %f, %f}},\n", gearState1, gearState2, gearState3);
    }

    close(fd);
    if (opts.arduino() && arduino > 0) {
        serialport_flush(arduino);
        serialport_close(arduino);
    }
}

/* calculate the prediction of the current inference pssition */
void RealisticRobotArmV3InferenceNetwork::calculatePrediction() {

    FloatType timePerValue = opts.numSimulationTimesteps() / FloatType(opts.numJoints());
    unsigned numOutputs = opts.numOutputNeurons();

    for (unsigned j = 0; j < opts.numJoints(); j++) {

        FloatType x = 0, y = 0, z = 0;
        FloatType xUp = 0, yUp = 0, zUp = 0;
        FloatType xXDirection = 0, yXDirection = 0, zXDirection = 0;
        FloatType qw = 0, qx = 0, qy = 0, qz = 0;
        unsigned counter = 0;


        for (unsigned b = 0; b < opts.trainSetSize(); b++) {
            FloatType *outputsOverTime = this->getGPUOutput(b);
            for (unsigned t = (j + 1) * timePerValue - opts.decodingTimesteps();
                 t < (j + 1) * timePerValue; t++, counter++) {

                x += outputsOverTime[t * numOutputs + 0];
                y += outputsOverTime[t * numOutputs + 1];
                z += outputsOverTime[t * numOutputs + 2];
                if (opts.quaternionRotation()) {
                    qw += outputsOverTime[t * numOutputs + 3];
                    qx += outputsOverTime[t * numOutputs + 4];
                    qy += outputsOverTime[t * numOutputs + 5];
                    qz += outputsOverTime[t * numOutputs + 6];
                } else {
                    xUp         += outputsOverTime[t * numOutputs + 3];
                    yUp         += outputsOverTime[t * numOutputs + 4];
                    zUp         += outputsOverTime[t * numOutputs + 5];
                    xXDirection += outputsOverTime[t * numOutputs + 6];
                    yXDirection += outputsOverTime[t * numOutputs + 7];
                    zXDirection += outputsOverTime[t * numOutputs + 8];
                }
            }
        }

        x /= counter;
        y /= counter;
        z /= counter;
        if (opts.quaternionRotation()) {
            toVectors(
                xUp,
                yUp,
                zUp,
                xXDirection,
                yXDirection,
                zXDirection,
                qw / counter, 
                qx / counter, 
                qy / counter, 
                qz / counter
            );
        } else {
            xUp         /= counter;
            yUp         /= counter;
            zUp         /= counter;
            xXDirection /= counter;
            yXDirection /= counter;
            zXDirection /= counter;
        }
        
        simulation->setPrediction(j, x, y, z, xUp, yUp, zUp, xXDirection, yXDirection, zXDirection);
    }
}

/* should return the network error */
FloatType RealisticRobotArmV3InferenceNetwork::getError() {
    
    FloatType error = simulation->getInferenceError();
    return error;
}

