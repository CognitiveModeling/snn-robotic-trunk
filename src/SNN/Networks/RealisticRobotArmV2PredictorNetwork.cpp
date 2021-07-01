#include "RealisticRobotArmV2PredictorNetwork.h"
#include "CurrentInputGradient.h"
#include "AdamOptimizer.h"
#include "SignDampedMomentumOptimizer.h"
#include "StochasticGradientDescentOptimizer.h"

using namespace SNN;
using namespace Networks;
using namespace Gradients;
using namespace Interfaces;
using namespace Options;
using namespace Optimizers;
using namespace Visualizations;

using std::vector;
using std::shared_ptr;
using std::make_shared;
using std::static_pointer_cast;

static int myrandom (int i) { return rand128() % i;}

/* constructor */
RealisticRobotArmV2PredictorNetwork::RealisticRobotArmV2PredictorNetwork(
    RobotArmPredictorNetworkOptions &opts,
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
    opts(opts),
    xAngles(opts.batchSize(), vector<FloatType>(opts.numJoints(), 0)),
    yAngles(opts.batchSize(), vector<FloatType>(opts.numJoints(), 0)),
    heights(opts.batchSize(), vector<FloatType>(opts.numJoints(), 0)),
    xPos(opts.batchSize(), vector<FloatType>(opts.numJoints(), 0)),
    yPos(opts.batchSize(), vector<FloatType>(opts.numJoints(), 0)),
    zPos(opts.batchSize(), vector<FloatType>(opts.numJoints(), 0)),
    qwPos(opts.batchSize(), vector<FloatType>(opts.numJoints(), 0)),
    qxPos(opts.batchSize(), vector<FloatType>(opts.numJoints(), 0)),
    qyPos(opts.batchSize(), vector<FloatType>(opts.numJoints(), 0)),
    qzPos(opts.batchSize(), vector<FloatType>(opts.numJoints(), 0)),
    latestError(100000000.0),
    recalculateError(false),
    rand(new_rand128_t()),
    epochCounter(0) {
    
    xAnglesLast = vector<FloatType>(opts.numJoints(), 0);
    yAnglesLast = vector<FloatType>(opts.numJoints(), 0);
    heightsLast = vector<FloatType>(opts.numJoints(), 0);

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
    FloatType clockTime = opts.decodingTimesteps();

    // clear inputs and targets
    for (unsigned b = 0; b < opts.trainSetSize(); b++) {
        for (unsigned t = 0; t < opts.numSimulationTimesteps(); t++) {
            for (unsigned n = 0; n < opts.numInputNeurons(); n++)
                this->inputAt(b, t, n) = 0;
            
            for (unsigned n = 0; n < opts.numOutputNeurons(); n++)
                    this->targetAt(b, t, n) = 0;

            this->errorMaskAt(b, t) = 0;
        }
    }
    
    // set clock inputs
    for (unsigned b = 0; b < opts.trainSetSize(); b++) {
        for (unsigned j = 0; j < opts.numJoints(); j++) {
            if (opts.numInputNeurons() == 4) {
                for (unsigned t = timePerValue * (j + 1) - clockTime; t < timePerValue * (j + 1); t++) {
                    this->inputAt(b, t, 3) = 1;
                }
            } else if (opts.numInputNeurons() > 3 + j) {
                for (unsigned t = timePerValue * (j + 1) - clockTime; t < timePerValue * (j + 1); t++) {
                    this->inputAt(b, t, 3 + j) = 1;
                }
            }
        }
    }

    if (opts.saveSimulation()) {
        this->simulationDataFile = fopen(opts.simulationFile().c_str(), "w");

        for (unsigned j = 0; j < opts.numJoints(); j++) {
            fprintf(this->simulationDataFile, "x-angle-%d,", j);
            fprintf(this->simulationDataFile, "y-angle-%d,", j);
            fprintf(this->simulationDataFile, "height-%d,", j);
            fprintf(this->simulationDataFile, "x-pos-%d,", j);
            fprintf(this->simulationDataFile, "y-pos-%d,", j);
            fprintf(this->simulationDataFile, "z-pos-%d,", j);
            fprintf(this->simulationDataFile, "x-up-direction-%d,", j);
            fprintf(this->simulationDataFile, "y-up-direction-%d,", j);
            fprintf(this->simulationDataFile, "z-up-direction-%d,", j);
            fprintf(this->simulationDataFile, "x-right-direction-%d,", j);
            fprintf(this->simulationDataFile, "y-right-direction-%d,", j);
            fprintf(this->simulationDataFile, "z-right-direction-%d,", j);
        }
        fprintf(this->simulationDataFile, "\n");
        log_str("Saveing simulation to file: " + opts.simulationFile());
    }

    if (opts.loadSimulation()) {
        this->trainsetIndex = 0;
        this->simulationDataFile = fopen(opts.simulationFile().c_str(), "r");

        int tmp;
        for (int j = 0; j < int(opts.numJoints()); j++) {
            assert(fscanf(this->simulationDataFile, "x-angle-%d,", &tmp) == 1);           assert(tmp == j);
            assert(fscanf(this->simulationDataFile, "y-angle-%d,", &tmp) == 1);           assert(tmp == j);
            assert(fscanf(this->simulationDataFile, "height-%d,", &tmp) == 1);            assert(tmp == j);
            assert(fscanf(this->simulationDataFile, "x-pos-%d,", &tmp) == 1);             assert(tmp == j);
            assert(fscanf(this->simulationDataFile, "y-pos-%d,", &tmp) == 1);             assert(tmp == j);
            assert(fscanf(this->simulationDataFile, "z-pos-%d,", &tmp) == 1);             assert(tmp == j);
            assert(fscanf(this->simulationDataFile, "x-up-direction-%d,", &tmp) == 1);    assert(tmp == j);
            assert(fscanf(this->simulationDataFile, "y-up-direction-%d,", &tmp) == 1);    assert(tmp == j);
            assert(fscanf(this->simulationDataFile, "z-up-direction-%d,", &tmp) == 1);    assert(tmp == j);
            assert(fscanf(this->simulationDataFile, "x-right-direction-%d,", &tmp) == 1); assert(tmp == j);
            assert(fscanf(this->simulationDataFile, "y-right-direction-%d,", &tmp) == 1); assert(tmp == j);
            assert(fscanf(this->simulationDataFile, "z-right-direction-%d,", &tmp) == 1); assert(tmp == j);
        }
        fseek(this->simulationDataFile, 1, SEEK_CUR);
        this->dataStart = ftell(this->simulationDataFile);
        log_str("Loading simulation from file: " + opts.simulationFile());
        
        this->dataEnd = this->dataStart;
        while (fgetc(this->simulationDataFile) != EOF)
            this->dataEnd++;

        fseek(this->simulationDataFile, this->dataStart, SEEK_SET);

        while (true) {
            SampleV2 *s = new SampleV2();
            for (unsigned j = 0; j < opts.numJoints(); j++) {
                double xAngle = 0; 
                double yAngle = 0; 
                double height = 0;
                double jointPosX = 0; 
                double jointPosY = 0; 
                double jointPosZ = 0; 
                double jointUpX = 0; 
                double jointUpY = 0; 
                double jointUpZ = 0;
                double jointXDirectionX = 0; 
                double jointXDirectionY = 0; 
                double jointXDirectionZ = 0;
                assert(fscanf(
                    this->simulationDataFile, 
                    "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", 
                    &xAngle,
                    &yAngle,
                    &height,
                    &jointPosX, 
                    &jointPosY, 
                    &jointPosZ, 
                    &jointUpX, 
                    &jointUpY, 
                    &jointUpZ,
                    &jointXDirectionX, 
                    &jointXDirectionY, 
                    &jointXDirectionZ
                ) == 12);
                s->xAngle.push_back(xAngle);
                s->yAngle.push_back(yAngle);
                s->height.push_back(height);
                s->jointPosX.push_back(jointPosX); 
                s->jointPosY.push_back(jointPosY); 
                s->jointPosZ.push_back(jointPosZ); 
                s->jointUpX.push_back(jointUpX); 
                s->jointUpY.push_back(jointUpY); 
                s->jointUpZ.push_back(jointUpZ);
                s->jointXDirectionX.push_back(jointXDirectionX); 
                s->jointXDirectionY.push_back(jointXDirectionY); 
                s->jointXDirectionZ.push_back(jointXDirectionZ);
            }
            this->trainset.push_back(s);

            fseek(this->simulationDataFile, 1, SEEK_CUR);
            if (ftell(this->simulationDataFile) > this->dataEnd - 10) {
                log_str("Data loaded", LOG_I);
                break;
            } 
        }
    }

}

/* should perform aditional resets */
void RealisticRobotArmV2PredictorNetwork::doReset() {

    this->LongShortTermMemoryEligibilityNetwork::doReset();

    latestError = 100000000.0;
    recalculateError = false;
    *rand = initialRandomState;
    simulation->reset();
}


/* initializes the train set of this */
void RealisticRobotArmV2PredictorNetwork::initTrainSet() {

    simulation->deactivateRenderer();

    FloatType xAngle = 0, yAngle = 0, height = 0;
    FloatType jointPosX = 0, jointPosY = 0, jointPosZ = 0;
    FloatType jointUpX = 0, jointUpY = 0, jointUpZ = 0;
    FloatType jointXDirectionX = 0, jointXDirectionY = 0, jointXDirectionZ = 0;
    FloatType timePerValue = opts.numSimulationTimesteps() / FloatType(opts.numJoints());

    for (unsigned b = 0; b < opts.trainSetSize(); b++) {
        
        simulation->randomPose(1.0, opts.edjeCasePercent());
                
        for (unsigned j = 0; j < opts.numJoints(); j++) {
            simulation->getAngles(j, xAngle, yAngle, height);

            if (opts.loadSimulation()) {
                xAngle = trainset[trainsetIndex]->xAngle[j];
                yAngle = trainset[trainsetIndex]->yAngle[j];
                height = trainset[trainsetIndex]->height[j];
            }
            if (opts.saveSimulation()) 
                fprintf(this->simulationDataFile, "%f,%f,%f,", xAngle,yAngle, height);

            xAngles[b][j] = xAngle;
            yAngles[b][j] = yAngle;
            heights[b][j] = height;

            simulation->getPosition(
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

            if (opts.loadSimulation()) {
                jointPosX = trainset[trainsetIndex]->jointPosX[j];
                jointPosY = trainset[trainsetIndex]->jointPosY[j];
                jointPosZ = trainset[trainsetIndex]->jointPosZ[j];
                jointUpX = trainset[trainsetIndex]->jointUpX[j];
                jointUpY = trainset[trainsetIndex]->jointUpY[j];
                jointUpZ = trainset[trainsetIndex]->jointUpZ[j];
                jointXDirectionX = trainset[trainsetIndex]->jointXDirectionX[j];
                jointXDirectionY = trainset[trainsetIndex]->jointXDirectionY[j];
                jointXDirectionZ = trainset[trainsetIndex]->jointXDirectionZ[j];
            }
            if (opts.saveSimulation()) {
                fprintf(
                    this->simulationDataFile, 
                    "%f,%f,%f,%f,%f,%f,%f,%f,%f,", 
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

            xPos[b][j] = jointPosX;
            yPos[b][j] = jointPosY;
            zPos[b][j] = jointPosZ;

            for (unsigned t = timePerValue * j; t < timePerValue * (j + 1); t++) {
                this->inputAt(b, t, 0) = xAngle;
                this->inputAt(b, t, 1) = yAngle;
                this->inputAt(b, t, 2) = height;
            }

            for (unsigned t = (j + 1) * timePerValue - opts.decodingTimesteps();
                 t < (j + 1) * timePerValue; t++) {

                this->targetAt(b, t, 0) = jointPosX;
                this->targetAt(b, t, 1) = jointPosY;
                this->targetAt(b, t, 2) = jointPosZ;
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
                    qwPos[b][j] = qw;
                    qxPos[b][j] = qx;
                    qyPos[b][j] = qy;
                    qzPos[b][j] = qz;
                    this->targetAt(b, t, 3) = qw;
                    this->targetAt(b, t, 4) = qx;
                    this->targetAt(b, t, 5) = qy;
                    this->targetAt(b, t, 6) = qz;
                } else {
                    this->targetAt(b, t, 3) = jointUpX;
                    this->targetAt(b, t, 4) = jointUpY;
                    this->targetAt(b, t, 5) = jointUpZ;
                    this->targetAt(b, t, 6) = jointXDirectionX;
                    this->targetAt(b, t, 7) = jointXDirectionY;
                    this->targetAt(b, t, 8) = jointXDirectionZ;
                }
                this->errorMaskAt(b, t) = 1;
            }
        }
        if (opts.loadSimulation()) {
            trainsetIndex++;
            if (trainsetIndex >= trainset.size()) {
                trainsetIndex = 0;
                std::random_shuffle(trainset.begin(), trainset.end(), myrandom);
            }
        }
        if (opts.saveSimulation()) {
            fprintf(this->simulationDataFile, "\n");
        }
    }

    simulation->activateRenderer();
}

/* calculate the prediction error for absolute positioning */
FloatType RealisticRobotArmV2PredictorNetwork::getDistanceError(bool denormalize) {

    FloatType error = 0;
    FloatType timePerValue = opts.numSimulationTimesteps() / FloatType(opts.numJoints());
    unsigned numOutputs = opts.numOutputNeurons();

    for (unsigned b = 0; b < opts.trainSetSize(); b++) {
        FloatType *outputsOverTime = this->getGPUOutput(b);

        for (unsigned j = 0; j < opts.numJoints(); j++) {
            FloatType predictedX = 0, predictedY = 0, predictedZ = 0;
            FloatType actualX = 0, actualY = 0, actualZ = 0;
            unsigned counter = 0;

            for (unsigned t = (j + 1) * timePerValue - opts.decodingTimesteps();
                 t < (j + 1) * timePerValue; t++, counter++) {

                predictedX += outputsOverTime[t * numOutputs + 0];
                predictedY += outputsOverTime[t * numOutputs + 1];
                predictedZ += outputsOverTime[t * numOutputs + 2];
            }

            predictedX /= counter;
            predictedY /= counter;
            predictedZ /= counter;

            if (opts.loadSimulation()) {
                actualX = xPos[b][j];
                actualY = yPos[b][j];
                actualZ = zPos[b][j];
            }
            
            if (denormalize) {
                simulation->denormalize(predictedX, predictedY, predictedZ);
                simulation->denormalize(actualX, actualY, actualZ);
            }
            
            error += sqrt(
                pow(actualX - predictedX, 2) + 
                pow(actualY - predictedY, 2) + 
                pow(actualZ - predictedZ, 2)
            );
        }
    }
    
    return error / (opts.trainSetSize() * opts.numJoints());
}

/* calculate the prediction error for orientation */
FloatType RealisticRobotArmV2PredictorNetwork::getRotationError() {

    FloatType error = 0;
    FloatType timePerValue = opts.numSimulationTimesteps() / FloatType(opts.numJoints());
    unsigned numOutputs = opts.numOutputNeurons();
    const FloatType PI = 3.141592653589793;

    for (unsigned b = 0; b < opts.trainSetSize(); b++) {
        FloatType *outputsOverTime = this->getGPUOutput(b);

        for (unsigned j = 0; j < opts.numJoints(); j++) {
            FloatType predictedW = 0, predictedX = 0, predictedY = 0, predictedZ = 0;
            FloatType actualW = 0, actualX = 0, actualY = 0, actualZ = 0;
            unsigned counter = 0;

            for (unsigned t = (j + 1) * timePerValue - opts.decodingTimesteps();
                 t < (j + 1) * timePerValue; t++, counter++) {

                predictedW += outputsOverTime[t * numOutputs + 3];
                predictedX += outputsOverTime[t * numOutputs + 4];
                predictedY += outputsOverTime[t * numOutputs + 5];
                predictedZ += outputsOverTime[t * numOutputs + 6];
            }

            predictedW /= counter;
            predictedX /= counter;
            predictedY /= counter;
            predictedZ /= counter;

            FloatType len = sqrt(pow(predictedW, 2) + pow(predictedX, 2) + pow(predictedY, 2) + pow(predictedZ, 2));
            predictedW /= len;
            predictedX /= len;
            predictedY /= len;
            predictedZ /= len;

            if (opts.loadSimulation()) {
                actualW = qwPos[b][j];
                actualX = qxPos[b][j];
                actualY = qyPos[b][j];
                actualZ = qzPos[b][j];
            }
            
            FloatType err = 360 / PI * acos(fabs(
                predictedW * actualW + 
                predictedX * actualX + 
                predictedY * actualY + 
                predictedZ * actualZ 
            ));
            if (!isnan(err))
                error += err;
        }
    }
    
    return error / (opts.trainSetSize() * opts.numJoints());
}

/* visualize the prediction for relative positioning */
void RealisticRobotArmV2PredictorNetwork::visualizePrediction() {

    FloatType timePerValue = opts.numSimulationTimesteps() / FloatType(opts.numJoints());
    unsigned numOutputs = opts.numOutputNeurons();

    for (unsigned j = 0; j < opts.numJoints(); j++) {

        FloatType x = 0, y = 0, z = 0;
        FloatType xUp = 0, yUp = 0, zUp = 0;
        FloatType xXDirection = 0, yXDirection = 0, zXDirection = 0;
        FloatType qw = 0, qx = 0, qy = 0, qz = 0;
        unsigned counter = 0;


        FloatType *outputsOverTime = this->getGPUOutput(0);
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
        
        simulation->setAngles(j, xAnglesLast[j], yAnglesLast[j], heightsLast[j]);
        simulation->setPrediction(j, x, y, z, xUp, yUp, zUp, xXDirection, yXDirection, zXDirection);
    }
}

/* should return the network error */
FloatType RealisticRobotArmV2PredictorNetwork::getError() {

    if (recalculateError) {
        latestError = getDistanceError();
    }

    recalculateError = false;
    return latestError;
}

/* trains the network for the given number of epochs */
FloatType RealisticRobotArmV2PredictorNetwork::train(unsigned epochs, bool evaluate, int loglevel) {
    if (!opts.saveSimulation()) 
        log_str("RealisticRobotArmV2PredictorNetwork training started", LOG_I);

    FloatType minNMSE  = 1;
    FloatType avgNMSE  = 0;
    FloatType avgError = 0;
    FloatType avgNormError = 0;
    FloatType avgRotError = 0;
    FloatType avgSum = 0;
    FloatType learnRate = opts.learnRate();
    for (unsigned i = 0; i < epochs; i++) {

        recalculateError = true;
        epochCounter = i;

        for (unsigned n = 0; n < xAnglesLast.size(); n++) {
            xAnglesLast[n] = xAngles[0][n];
            yAnglesLast[n] = yAngles[0][n];
            heightsLast[n] = heights[0][n];
        }

        initTrainSet();
        //visualizePrediction();
        //simulation->saveScreenshot("simulation-prediction-" + itoa(i) + ".png", 1);
        if (opts.saveSimulation()) {
            log_str("Dataset generation: " + ftoa((i+1) * 100.0 / epochs) + "%", loglevel);
            continue;
        }

        this->updateGPU();

        avgNMSE = avgNMSE * 0.99 + this->getNormalizedMeanSquaredError();
        avgError = avgError * 0.99 + this->getError();
        avgNormError = avgNormError * 0.99 + this->getDistanceError(false);
        avgRotError = avgRotError * 0.99 + this->getRotationError();
        avgSum = avgSum * 0.99 + 1;

        log_str(
            "Epoch: " + itoa(i) + ", Error^2: " + ftoa(this->getError()) + " / " +
            ftoa(avgError / avgSum) + ", Rot-Error: " + ftoa(this->getRotationError()) + " / " +
            ftoa(avgRotError / avgSum) + ", Norm-Error^2: " + ftoa(this->getDistanceError(false)) + " / " +
            ftoa(avgNormError / avgSum) + ", NMSE: " + ftoa(this->getNormalizedMeanSquaredError(), 6) + " / " +
            ftoa(avgNMSE / avgSum, 6) + ((opts.pruneStart() >= 0) ? ", synapses: " + itoa(this->getNumActiveSynases()) : "") +
            ", lr: " + ftoa(this->getInputOptimizers()[0]->getLearnRate(), 6) + 
            ", fr-lr: " + ftoa(this->getInputRegularizers()[0]->getLearnRate(), 6) + 
            ", fr: " + ftoa(this->getAverageFiringRate() * 1000), loglevel
        );

        if (avgSum > 10 && opts.adaptiveLearningRate()) {
            //learnRate = std::min(opts.learnRate() * sqrt(avgNMSE / avgSum), learnRate);
            minNMSE = std::min(minNMSE, avgNMSE / avgSum);
            learnRate = opts.learnRate() * pow(minNMSE, 0.5);
            //if (sin(i / 300.0) > 0)
            //    learnRate += opts.learnRate() * pow(minNMSE, 0.5) * sin(i / 300.0);
            //else
            //    learnRate += opts.learnRate() * pow(minNMSE, 0.5) * sin(i / 300.0) / 2;
            for (unsigned o = 0; o < this->getNumOptimizers(); o++)
                this->getOptimizer(o).setLearnRate(learnRate);
        }

    }

    if (!opts.saveSimulation()) {
        log_str("Epochs " + itoa(epochs) + ", Euclidean-Error: " + ftoa(avgError) + " / " +
                ftoa(avgError) + ", fr: " + ftoa(this->getAverageFiringRate() * 1000), loglevel);
    }
    
    if (opts.saveSimulation() || opts.loadSimulation())
        fclose(this->simulationDataFile);
    
    if (evaluate) {
        this->save("RealisticRobotArmV2PredictorNetwork-Error" + ftoa(avgError / avgSum) + "-time" + itoa(gettime_usec() / 1000000) + 
                   "-joints" + itoa(opts.numJoints()) + "-hidden" + itoa(opts.numHiddenNeurons()) + ".snn");
        exit(EXIT_SUCCESS);
    }


    return avgError;
}
