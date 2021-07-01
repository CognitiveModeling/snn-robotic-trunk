#include "RealisticRobotArmV3PredictorNetwork.h"
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
RealisticRobotArmV3PredictorNetwork::RealisticRobotArmV3PredictorNetwork(
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
    
    if (opts.quaternionRotation()) {
        this->targetWeightAt(0) = opts.positionErrorWeight();
        this->targetWeightAt(1) = opts.positionErrorWeight();
        this->targetWeightAt(2) = opts.positionErrorWeight();
    } else {
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
            fprintf(this->simulationDataFile, "1st-gear-state-%d,", j);
            fprintf(this->simulationDataFile, "2nd-gear-state-%d,", j);
            fprintf(this->simulationDataFile, "3rd-gear-state-%d,", j);
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
            assert(fscanf(this->simulationDataFile, "1st-gear-state-%d,", &tmp) == 1);           assert(tmp == j);
            assert(fscanf(this->simulationDataFile, "2nd-gear-state-%d,", &tmp) == 1);           assert(tmp == j);
            assert(fscanf(this->simulationDataFile, "3rd-gear-state-%d,", &tmp) == 1);            assert(tmp == j);
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
            SampleV3 *s = new SampleV3();
            for (unsigned j = 0; j < opts.numJoints(); j++) {
                double gearState1 = 0; 
                double gearState2 = 0; 
                double gearState3 = 0;
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
                    &gearState1,
                    &gearState2,
                    &gearState3,
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
                s->gearState1.push_back(gearState1);
                s->gearState2.push_back(gearState2);
                s->gearState3.push_back(gearState3);
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

    poseFD = 0;
    if (opts.trainingPosesFile() != "") {
        poseFD = open(opts.trainingPosesFile().c_str(), O_WRONLY|O_CREAT|O_TRUNC, 00777);
        if (poseFD <= 0)
            log_err("failed to open training pose file", LOG_E);

        unsigned epochs = 100000;
        unsigned numJoints = opts.numJoints();
        assert(write(poseFD, &epochs, sizeof(unsigned)) == sizeof(unsigned));
        assert(write(poseFD, &numJoints, sizeof(unsigned)) == sizeof(unsigned));
    }
}

/* should perform aditional resets */
void RealisticRobotArmV3PredictorNetwork::doReset() {

    this->LongShortTermMemoryEligibilityNetwork::doReset();

    latestError = 100000000.0;
    recalculateError = false;
    *rand = initialRandomState;
    simulation->reset();
}


/* initializes the train set of this */
void RealisticRobotArmV3PredictorNetwork::initTrainSet() {

    simulation->deactivateRenderer();

    FloatType gearState1 = 0, gearState2 = 0, gearState3 = 0;
    FloatType jointPosX = 0, jointPosY = 0, jointPosZ = 0;
    FloatType jointUpX = 0, jointUpY = 0, jointUpZ = 0;
    FloatType jointXDirectionX = 0, jointXDirectionY = 0, jointXDirectionZ = 0;
    FloatType timePerValue = opts.numSimulationTimesteps() / FloatType(opts.numJoints());

    for (unsigned b = 0; b < opts.trainSetSize(); b++) {
        
        bool setted = false;
        while (!setted && !opts.loadSimulation()) {
            log_str("computing batch: " + itoa(b + 1) + " / " + itoa(opts.trainSetSize()) + " (" + ftoa(FloatType((b+1)*100) / opts.trainSetSize(),1) + " %)");
            setted = simulation->randomPose(1.0, opts.edjeCasePercent());
        }
        if (poseFD != 0) {
            bool setted = false;
            while (!setted) {
                setted = true;
                for (unsigned j = 0; j < opts.numJoints(); j++) {
                    simulation->getGearStates(j, gearState1, gearState2, gearState3);
                    setted &= simulation->setInferredGearStates(j, gearState1, gearState2, gearState3);
                }
            }
            simulation->saveState(poseFD);
        }
                
        for (unsigned j = 0; j < opts.numJoints(); j++) {
            simulation->getGearStates(j, gearState1, gearState2, gearState3);

            if (opts.loadSimulation()) {
                gearState1 = trainset[trainsetIndex]->gearState1[j];
                gearState2 = trainset[trainsetIndex]->gearState2[j];
                gearState3 = trainset[trainsetIndex]->gearState3[j];
            }
            if (opts.saveSimulation()) 
                fprintf(this->simulationDataFile, "%f,%f,%f,", gearState1,gearState2, gearState3);

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
                this->inputAt(b, t, 0) = gearState1;
                this->inputAt(b, t, 1) = gearState2;
                this->inputAt(b, t, 2) = gearState3;
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
FloatType RealisticRobotArmV3PredictorNetwork::getDistanceError(bool denormalize) {

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
FloatType RealisticRobotArmV3PredictorNetwork::getRotationError() {

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

/* should return the network error */
FloatType RealisticRobotArmV3PredictorNetwork::getError() {

    if (recalculateError) {
        latestError = getDistanceError();
    }

    recalculateError = false;
    return latestError;
}

/* trains the network for the given number of epochs */
FloatType RealisticRobotArmV3PredictorNetwork::train(unsigned epochs, bool evaluate, bool exportSpikes, int loglevel) {
    if (!opts.saveSimulation()) 
        log_str("RealisticRobotArmV3PredictorNetwork training started", LOG_I);

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

        initTrainSet();
        if (opts.saveSimulation()) {
            log_str("Dataset generation: " + ftoa((i+1) * 100.0 / epochs) + "%", loglevel);
            continue;
        }

        this->updateGPU();

        if (exportSpikes) {
            int fd = open(std::string("LSNN-spikes-epoch" + itoa(i)).c_str(), O_WRONLY|O_CREAT|O_TRUNC, 00777);
            this->saveSpikes(fd);
            close(fd);
        }

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

    if (poseFD > 0)
        close(poseFD);
    
    if (evaluate) {
        this->save("RealisticRobotArmV3PredictorNetwork-Error" + ftoa(avgError) + "-j" + itoa(opts.numJoints()) + "-time" + itoa(gettime_usec() / 1000000) + ".snn");
        exit(EXIT_SUCCESS);
    }


    return avgError;
}
