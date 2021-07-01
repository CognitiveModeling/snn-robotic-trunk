#include "RealisticRobotArmV2InferenceNetwork.h"
#include "RealisticRobotArmV3InferenceNetwork.h"
#include "utils.h"
#include "Threads.h"
#include "arduino-serial-lib.h"
#include "LongShortTermMemorySparseKernelCaller.h"

using namespace SNN;
using namespace Networks;
using namespace Options;
using namespace Interfaces;
using namespace Visualizations;
using namespace Kernels;
using namespace GPU;

using std::vector;
using std::shared_ptr;
using std::make_shared;
using std::static_pointer_cast;


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

static void sendComandV2(int arduino, float pX, float pY, float pL, int pIndex) {

    uint8_t x = uint8_t(roundf((pX + 1) * 127.5));
    uint8_t y = uint8_t(roundf((pY + 1) * 127.5));
    uint8_t l = uint8_t(roundf((pL + 1) * 127.5));
    uint8_t index = pIndex;

    sendByte(arduino, 0x10 | (x & 0x0F));
    sendByte(arduino, 0x20 | ((x >> 4) & 0x0F));

    sendByte(arduino, 0x30 | (y & 0x0F));
    sendByte(arduino, 0x40 | ((y >> 4) & 0x0F));

    sendByte(arduino, 0x50 | (l & 0x0F));
    sendByte(arduino, 0x60 | ((l >> 4) & 0x0F));

    sendByte(arduino, 0x70 | (index & 0x0F));
    sendByte(arduino, 0xF0);

    printf("send comand: (%f, %f, %f, %d) => (%u, %u, %u, %u)\n", pX, pY, pL, pIndex, x, y, l, index);
}

static void sendComandV3(int arduino, float gear1, float gear2, float gear3, int index) {

    float gears[3] = { gear1, gear2, gear3 };
    if (index == 1 || index == 4) {
        gears[0] = gear2;
        gears[1] = gear3;
        gears[2] = gear1;
    }
    if (index == 2 || index == 5) {
        gears[0] = gear3;
        gears[1] = gear1;
        gears[2] = gear2;
    }

    uint8_t g1 = uint8_t(roundf((gears[0] * 180)));
    uint8_t g2 = uint8_t(roundf((gears[1] * 180)));
    uint8_t g3 = uint8_t(roundf((gears[2] * 180)));
    uint8_t i = index;

    sendByte(arduino, 0x10 | (g1 & 0x0F));
    sendByte(arduino, 0x20 | ((g1 >> 4) & 0x0F));

    sendByte(arduino, 0x30 | (g2 & 0x0F));
    sendByte(arduino, 0x40 | ((g2 >> 4) & 0x0F));

    sendByte(arduino, 0x50 | (g3 & 0x0F));
    sendByte(arduino, 0x60 | ((g3 >> 4) & 0x0F));

    sendByte(arduino, 0x70 | (i & 0x0F));
    sendByte(arduino, 0xF0);

    printf("send comand: (%f, %f, %f, %d) => (%u, %u, %u, %u)\n", gear1, gear2, gear3, index, g1, g2, g3, i);
}


/* rescalles the given simulation (time domain) for better visualization */
void rescaleSimulationGearBased(
    std::shared_ptr<BasicRobotArmSimulation> simulation, 
    FloatType minDistance,
    unsigned epochs,
    int inFile,
    int outFile
) {
    
    vector<FloatType> gear1Last(simulation->getNumJoints(), -1);
    vector<FloatType> gear2Last(simulation->getNumJoints(), -1);
    vector<FloatType> gear3Last(simulation->getNumJoints(), -1);
    unsigned newEpochs = 0;
    for (unsigned i = 0; i < epochs; i++) {
        simulation->loadState(inFile);

        FloatType maxDistance = 0;
        for (unsigned j = 0; j < simulation->getNumJoints(); j++) {
            FloatType g1, g2, g3;

            static_pointer_cast<RealisticRobotArmSimulationV3>(simulation)->getInferredGearStates(j, g1, g2, g3);
            maxDistance = std::max(maxDistance, fabs(gear1Last[j] - g1));
            maxDistance = std::max(maxDistance, fabs(gear2Last[j] - g2));
            maxDistance = std::max(maxDistance, fabs(gear3Last[j] - g3));
        }

        if (maxDistance > minDistance || epochs - 1 == i) {
            simulation->saveState(outFile);
            newEpochs++;

            for (unsigned j = 0; j < simulation->getNumJoints(); j++) 
                static_pointer_cast<RealisticRobotArmSimulationV3>(simulation)->getInferredGearStates(j, gear1Last[j], gear2Last[j], gear3Last[j]);
        }
        log_str("rescalling: " + itoa(newEpochs) + " | " + itoa(i) + " / " + itoa(epochs));
    }

    lseek(outFile, 0, SEEK_SET);
    assert(write(outFile, &newEpochs, sizeof(unsigned)) == sizeof(unsigned));
    close(outFile);
}

/* rescalles the given simulation (time domain) for better visualization */
void rescaleSimulationAngleBased(
    std::shared_ptr<BasicRobotArmSimulation> simulation, 
    FloatType minDistance,
    unsigned epochs,
    int inFile,
    int outFile
) {
    
    vector<FloatType> gear1Last(simulation->getNumJoints(), -1);
    vector<FloatType> gear2Last(simulation->getNumJoints(), -1);
    vector<FloatType> gear3Last(simulation->getNumJoints(), -1);
    unsigned newEpochs = 0;
    for (unsigned i = 0; i < epochs; i++) {
        simulation->loadState(inFile);

        FloatType maxDistance = 0;
        for (unsigned j = 0; j < simulation->getNumJoints(); j++) {
            FloatType g1, g2, g3;

            static_pointer_cast<RealisticRobotArmSimulationV2>(simulation)->getInferredAngles(j, g1, g2, g3);
            maxDistance = std::max(maxDistance, fabs(gear1Last[j] - g1));
            maxDistance = std::max(maxDistance, fabs(gear2Last[j] - g2));
            maxDistance = std::max(maxDistance, fabs(gear3Last[j] - g3));
        }

        if (maxDistance > minDistance || epochs - 1 == i) {
            simulation->saveState(outFile);
            newEpochs++;

            for (unsigned j = 0; j < simulation->getNumJoints(); j++) 
                static_pointer_cast<RealisticRobotArmSimulationV2>(simulation)->getInferredAngles(j, gear1Last[j], gear2Last[j], gear3Last[j]);
        }
        log_str("rescalling: " + itoa(newEpochs) + " | " + itoa(i) + " / " + itoa(epochs));
    }

    lseek(outFile, 0, SEEK_SET);
    assert(write(outFile, &newEpochs, sizeof(unsigned)) == sizeof(unsigned));
    close(outFile);
}

/* rescalles the given simulation (time domain) for better visualization */
void rescaleSimulationFast(
    std::shared_ptr<BasicRobotArmSimulation> simulation, 
    FloatType minDistance,
    unsigned epochs,
    int inFile,
    int outFile
) {
    
    FloatType lastError = 100000000;
    unsigned newEpochs  = 0;
    FloatType minError  = 100000000;
    unsigned minPos     = 0;

    for (unsigned i = 0; i < epochs; i++) {
        simulation->loadState(inFile);
        unsigned curPos = lseek(inFile, 0, SEEK_CUR);

        FloatType curError = simulation->getInferenceError();
        FloatType distance = lastError - curError;

        if (distance > minDistance) {
            simulation->saveState(outFile);
            newEpochs++;
            lastError = curError;
        }
        log_str("rescalling: " + itoa(newEpochs) + " | " + itoa(i) + " / " + itoa(epochs) + " => " + ftoa(lastError) + " / " + ftoa(curError));
        
        curError *= simulation->getInferenceOrientationError();
        if (curError < minError) {
            minError = curError;
            minPos   = curPos;
        }
    }

    lseek(inFile, minPos, SEEK_SET);
    simulation->loadState(inFile);
    simulation->saveState(outFile);
    newEpochs++;
    log_str("rescalled: " + itoa(newEpochs) + " => " + ftoa(simulation->getInferenceError()) + " - " + ftoa(simulation->getInferenceOrientationError()));

    lseek(outFile, 0, SEEK_SET);
    assert(write(outFile, &newEpochs, sizeof(unsigned)) == sizeof(unsigned));
    close(outFile);
}

/* rescalles the given simulation (time domain) for better visualization */
void rescaleSimulationReal(
    std::shared_ptr<BasicRobotArmSimulation> simulation, 
    FloatType minDistance,
    unsigned epochs,
    int inFile,
    int outFile
) {
    
    FloatType lastX = 10000, lastY = 10000, lastZ = 10000;
    unsigned newEpochs = 0;
    for (unsigned i = 0; i < epochs; i++) {
        simulation->loadState(inFile);

        FloatType curX, curY, curZ, curUpX, curUpY, curUpZ, curXDirectionX, curXDirectionY, curXDirectionZ;
        simulation->getInferredPosition(
            simulation->getNumJoints() - 1, 
            curX, 
            curY, 
            curZ, 
            curUpX, 
            curUpY, 
            curUpZ,
            curXDirectionX, 
            curXDirectionY, 
            curXDirectionZ
        );
        simulation->denormalize(curX, curY, curZ);

        FloatType distance = sqrt(pow(curX - lastX, 2) + pow(curY - lastY, 2) + pow(curZ - lastZ, 2));
        if (distance > minDistance || epochs - 1 == i) {
            simulation->saveState(outFile);
            newEpochs++;
            lastX = curX;
            lastY = curY;
            lastZ = curZ;
        }
        log_str("rescalling: " + itoa(newEpochs) + " | " + itoa(i) + " / " + itoa(epochs));
    }

    lseek(outFile, 0, SEEK_SET);
    assert(write(outFile, &newEpochs, sizeof(unsigned)) == sizeof(unsigned));
    close(outFile);
}

void simulationToDataset(
    std::string fname, 
    unsigned epochs,
    unsigned numJoints,
    int simFile,
    std::shared_ptr<RealisticRobotArmSimulationV3> simulation) {

    FILE *simulationDataFile = fopen(fname.c_str(), "w");
    FloatType gearState1 = 0, gearState2 = 0, gearState3 = 0;
    FloatType jointPosX = 0, jointPosY = 0, jointPosZ = 0;
    FloatType jointUpX = 0, jointUpY = 0, jointUpZ = 0;
    FloatType jointXDirectionX = 0, jointXDirectionY = 0, jointXDirectionZ = 0;

    for (unsigned j = 0; j < numJoints; j++) {
        fprintf(simulationDataFile, "1st-gear-state-%d,", j);
        fprintf(simulationDataFile, "2nd-gear-state-%d,", j);
        fprintf(simulationDataFile, "3rd-gear-state-%d,", j);
        fprintf(simulationDataFile, "x-pos-%d,", j);
        fprintf(simulationDataFile, "y-pos-%d,", j);
        fprintf(simulationDataFile, "z-pos-%d,", j);
        fprintf(simulationDataFile, "x-up-direction-%d,", j);
        fprintf(simulationDataFile, "y-up-direction-%d,", j);
        fprintf(simulationDataFile, "z-up-direction-%d,", j);
        fprintf(simulationDataFile, "x-right-direction-%d,", j);
        fprintf(simulationDataFile, "y-right-direction-%d,", j);
        fprintf(simulationDataFile, "z-right-direction-%d,", j);
    }
    fprintf(simulationDataFile, "\n");
    log_str("Saveing simulation to dataset file: " + fname);

    for (unsigned e = 0; e < epochs; e++) {
        
        bool setted = false;
        while (!setted) {
            log_str("converting: " + itoa(e + 1) + " / " + itoa(epochs) + " (" + ftoa(FloatType((e+1)*100) / epochs,1) + " %)");
            setted = simulation->loadState(simFile, 10000);
        }
                
        for (unsigned j = 0; j < numJoints; j++) {
            simulation->getInferredGearStates(j, gearState1, gearState2, gearState3);
            fprintf(simulationDataFile, "%f,%f,%f,", gearState1,gearState2, gearState3);

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
            fprintf(
                simulationDataFile, 
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
        fprintf(simulationDataFile, "\n");
    }
    fclose(simulationDataFile);
}

int main(int argc, char *argv[]) {

    uint32_t seed = parse_i_arg(gettime_usec(), "--seed");
    log_str("Random seed: " + itoa(seed));
        init_rand128(seed);

    assert(freopen("/dev/null", "w", stderr) != NULL);
    LongShortTermMemorySparseKernelCaller::setDevice(parse_i_arg(0, "--device"));

    RobotArmInferenceNetworkOptions opts;
    opts.arduino(has_arg("--arduino"));
    opts.arduinoPort(parse_arg("", "--arduinoPort"));

    int arduino = 0;
    if (opts.arduino()) {
        printf("opening arduino at %s\n", opts.arduinoPort().c_str());
        arduino = serialport_init(opts.arduinoPort().c_str(), 115200);
        serialport_flush(arduino);
    }

    if (arduino != 0 && has_arg("--test-gripper")) {
        FloatType g = parse_f_arg(1, "--test-gripper");
        sendComandV3(arduino, g, g, g, 6);
        serialport_flush(arduino);
        serialport_close(arduino);  
        exit(EXIT_SUCCESS);
    }

    if (arduino != 0 && has_arg("--test-arduino")) {
        for (unsigned n = 0; n < 25; n++)
            for (unsigned i = 0; i < 6; i++) 
                sendComandV3(arduino, n / 100.0, 0, 0, i);
        while (true) Thread::yield();
    }

    if (has_arg("--replay")) {
        
        int fd = open(get_arg("--replay").c_str(), O_RDONLY);
        unsigned epochs = 0;
        unsigned numJoints = 0;

        assert(read(fd, &epochs,    sizeof(unsigned)) == sizeof(unsigned));
        assert(read(fd, &numJoints, sizeof(unsigned)) == sizeof(unsigned));

        std::shared_ptr<BasicRobotArmSimulation> simulation;
        if (has_arg("--realistic-v2")) {

            simulation = static_pointer_cast<BasicRobotArmSimulation>(
                make_shared<RealisticRobotArmSimulationV2>(
                    numJoints,
                    parse_arg("", "--base"),
                    parse_arg("", "--servo"),
                    parse_arg("", "--linearGear"),
                    parse_arg("", "--ballJoint"),
                    parse_arg("", "--gear"),
                    parse_arg("", "--stand"),
                    parse_arg("", "--gripper-base"), 
                    parse_arg("", "--gripper-servo"), 
                    parse_arg("", "--gripper-actuator"), 
                    parse_arg("", "--gripper"), 
                    parse_arg("", "--gripper-gear"), 
                    true,
                    !has_arg("--no-visualization")
                )
            );
        } else if (has_arg("--realistic-v3")) {

            simulation = static_pointer_cast<BasicRobotArmSimulation>(
                make_shared<RealisticRobotArmSimulationV3>(
                    numJoints,
                    parse_arg("", "--base"),
                    parse_arg("", "--first-module"),
                    parse_arg("", "--servo"),
                    parse_arg("", "--linearGear"),
                    parse_arg("", "--ballJoint"),
                    parse_arg("", "--gear"),
                    parse_arg("", "--stand"),
                    parse_arg("", "--gripper-base"), 
                    parse_arg("", "--gripper-servo"), 
                    parse_arg("", "--gripper-actuator"), 
                    parse_arg("", "--gripper"), 
                    parse_arg("", "--gripper-gear"), 
                    true,
                    !has_arg("--no-visualization")
                )
            );
        }
        if (has_arg("--epochs"))
            epochs = get_i_arg("--epochs");
        
        unsigned speed = parse_f_arg(0.1, "--speed") * 1000000;

        if (has_arg("--set-camera") && has_arg("--realistic-v3")) {
            double x, y, z, fx, fy, fz, ux, uy, uz;
            sscanf(get_arg("--set-camera").c_str(), "%lf:%lf:%lf:%lf:%lf:%lf:%lf:%lf:%lf", 
                &x, &y, &z, &fx, &fy, &fz, &ux, &uy, &uz
            );
            printf("Camera: %f:%f:%f:%f:%f:%f:%f:%f:%f\n",
                x, y, z, fx, fy, fz, ux, uy, uz
            );
            static_pointer_cast<RealisticRobotArmSimulationV3>(simulation)->setCamera(
                x, y, z, fx, fy, fz, ux, uy, uz
            );
        }
        if (has_arg("--set-camera") && has_arg("--realistic-v2")) {
            double x, y, z, fx, fy, fz, ux, uy, uz;
            sscanf(get_arg("--set-camera").c_str(), "%lf:%lf:%lf:%lf:%lf:%lf:%lf:%lf:%lf", 
                &x, &y, &z, &fx, &fy, &fz, &ux, &uy, &uz
            );
            printf("Camera: %f:%f:%f:%f:%f:%f:%f:%f:%f\n",
                x, y, z, fx, fy, fz, ux, uy, uz
            );
            static_pointer_cast<RealisticRobotArmSimulationV2>(simulation)->setCamera(
                x, y, z, fx, fy, fz, ux, uy, uz
            );
        }

        if (has_arg("--rescale")) {

            int outFile = open(get_arg("--rescale").c_str(), O_WRONLY|O_CREAT|O_TRUNC, 00777);

            assert(write(outFile, &epochs,    sizeof(unsigned)) == sizeof(unsigned));
            assert(write(outFile, &numJoints, sizeof(unsigned)) == sizeof(unsigned));

            if (has_arg("--rescale-fast"))
                rescaleSimulationFast(simulation, parse_f_arg(1, "--rescale-distance"), epochs, fd, outFile);
            else if (has_arg("--rescale-gears"))
                rescaleSimulationGearBased(simulation, parse_f_arg(0.01, "--rescale-distance"), epochs, fd, outFile);
            else if (has_arg("--rescale-angles"))
                rescaleSimulationAngleBased(simulation, parse_f_arg(0.01, "--rescale-distance"), epochs, fd, outFile);
            else
                rescaleSimulationReal(simulation, parse_f_arg(1, "--rescale-distance"), epochs, fd, outFile);
            return 0;
        }

        if (has_arg("--convert") && has_arg("--realistic-v3")) {
            simulationToDataset(
                get_arg("--convert"), 
                epochs, 
                numJoints,
                fd, 
                static_pointer_cast<RealisticRobotArmSimulationV3>(simulation)
            );
            return 0;
        }

        if (has_arg("--frame") && (has_arg("--realistic-v1") || has_arg("--realistic-v2") || has_arg("--realistic-v3"))) {
            for (int i = 0; i < get_i_arg("--frame") && i < int(epochs - 1); i++) {
                if (has_arg("--realistic-v2"))
                    static_pointer_cast<RealisticRobotArmSimulationV3>(simulation)->skipState(fd);
                if (has_arg("--realistic-v3"))
                    static_pointer_cast<RealisticRobotArmSimulationV3>(simulation)->skipState(fd);
                log_str("Skipping Epoch: " + itoa(i));
            }

            simulation->loadState(fd);
            for (unsigned i = 0; ; i++) {
                usleep(speed);
                if (has_arg("--screenshot"))
                    simulation->saveScreenshot("inference-epoch" + itoa(i) + ".png", 1);

                if (has_arg("--print-camera") && has_arg("--realistic-v3")) {
                    double x, y, z, fx, fy, fz, ux, uy, uz;
                    static_pointer_cast<RealisticRobotArmSimulationV3>(simulation)->getCamera(
                        x, y, z, fx, fy, fz, ux, uy, uz
                    );
                    printf("Camera: %f:%f:%f:%f:%f:%f:%f:%f:%f\n",
                        x, y, z, fx, fy, fz, ux, uy, uz
                    );
                }
                if (has_arg("--print-camera") && has_arg("--realistic-v2")) {
                    double x, y, z, fx, fy, fz, ux, uy, uz;
                    static_pointer_cast<RealisticRobotArmSimulationV2>(simulation)->getCamera(
                        x, y, z, fx, fy, fz, ux, uy, uz
                    );
                    printf("Camera: %f:%f:%f:%f:%f:%f:%f:%f:%f\n",
                        x, y, z, fx, fy, fz, ux, uy, uz
                    );
                }
            }
        }

        vector<FloatType> gear1(10, 0), gear2(10, 0), gear3(10, 0);
        vector<FloatType> nextGear1(10, 0), nextGear2(10, 0), nextGear3(10, 0);

        for (unsigned i = 0; i < epochs; i++) {
            if (!has_arg("--realistic-v1") && !has_arg("--realistic-v2") && !has_arg("--realistic-v3"))
                simulation->deactivateRenderer();

            if (has_arg("--realistic-v3") && has_arg("--no-visualization"))
                static_pointer_cast<RealisticRobotArmSimulationV3>(simulation)->loadState(fd, 0);
            if (has_arg("--realistic-v3"))
                static_pointer_cast<RealisticRobotArmSimulationV3>(simulation)->loadState(fd, 10000);
            else
                simulation->loadState(fd);

            if (!has_arg("--realistic-v1") && !has_arg("--realistic-v2") && !has_arg("--realistic-v3"))
                simulation->activateRenderer();
            log_str("Epoch: " + itoa(i) + ", Inference-Error: " + 
                    ftoa(simulation->getInferenceError()) + ", Predictor-Error: " + 
                    ftoa(simulation->getInferencePredictionError()));

            if (i == 0)
                usleep(parse_f_arg(0, "--init") * 1000000);

            usleep(speed);
            if (has_arg("--screenshot"))
                simulation->saveScreenshot("inference-epoch" + itoa(i) + ".png", 1);

            if (arduino != 0 && has_arg("--realistic-v2")) {
                for (unsigned j = 0; j < 10; j++) 
                    static_pointer_cast<RealisticRobotArmSimulationV2>(simulation)->getInferredAngles(j, nextGear1[j], nextGear2[j], nextGear3[j]);

                
                for (unsigned j = 0; j < 10; j++) {
                    sendComandV2(
                        arduino, 
                        nextGear1[j], 
                        nextGear2[j], 
                        nextGear3[j], 
                        j
                    );
                }

                for (unsigned j = 0; j < 10; j++) {
                    gear1[j] = nextGear1[j];
                    gear2[j] = nextGear2[j];
                    gear3[j] = nextGear3[j];
                }
            }

            if (arduino != 0 && has_arg("--realistic-v3")) {
                for (unsigned j = 0; j < 6; j++) 
                    static_pointer_cast<RealisticRobotArmSimulationV3>(simulation)->getInferredGearStates(j, nextGear1[j], nextGear2[j], nextGear3[j]);

                
                if (has_arg("--no-visualization")) {
                    for (unsigned j = 0; j < 6; j++) {
                        sendComandV3(
                            arduino, 
                            nextGear1[j], 
                            nextGear2[j], 
                            nextGear3[j], 
                            j
                        );
                    }
                } else {
                    for (unsigned n = 1; n <= 30; n++) {
                        for (unsigned j = 0; j < 6; j++) {
                            sendComandV3(
                                arduino, 
                                gear1[j] + (nextGear1[j] - gear1[j]) * n / 30.0, 
                                gear2[j] + (nextGear2[j] - gear2[j]) * n / 30.0, 
                                gear3[j] + (nextGear3[j] - gear3[j]) * n / 30.0, 
                                j
                            );
                        }
                    }
                }

                for (unsigned j = 0; j < 6; j++) {
                    gear1[j] = nextGear1[j];
                    gear2[j] = nextGear2[j];
                    gear3[j] = nextGear3[j];
                }
            }
        }

        if (arduino != 0 && has_arg("--realistic-v3")) {
            if (has_arg("--set-gripper")) {
                usleep(parse_f_arg(0.5, "--gripper-sleep") * 1000000);
                FloatType g = parse_f_arg(1, "--set-gripper");
                sendComandV3(arduino, g, g, g, 6);
            }

            usleep(parse_f_arg(10, "--end-pose") * 1000000);

            if (has_arg("--set-gripper")) {
                sendComandV3(arduino, 0, 0, 0, 6);
            }

            for (unsigned i = 0; i < 40; i++) {
                for (unsigned j = 0; j < 6; j++) {
                    gear1[j] = std::max(0.0, gear1[j] - 0.025);
                    gear2[j] = std::max(0.0, gear2[j] - 0.025);
                    gear3[j] = std::max(0.0, gear3[j] - 0.025);
                    sendComandV3(arduino, gear1[j], gear2[j], gear3[j], j);
                }
                if (!has_arg("--no-visualization")) {
                    for (unsigned j = 0; i % 10 == 0 && j < 6; j++) {
                        static_pointer_cast<RealisticRobotArmSimulationV3>(simulation)->setInferredGearStates(j, gear1[j], gear2[j], gear3[j]);
                    }
                }
            }
            serialport_flush(arduino);
            serialport_close(arduino);  
            exit(EXIT_SUCCESS);
        }

        if (arduino != 0 && has_arg("--realistic-v2")) {
            usleep(parse_f_arg(10, "--end-pose") * 1000000);

            bool finished = false;
            for (unsigned i = 0; i < 100 && !finished; i++) {
                finished = true;
                for (unsigned j = 0; j < 10; j++) {
                    gear3[j] = std::max(-1.0, gear3[j] - 0.02);
                    sendComandV2(arduino, gear1[j], gear2[j], gear3[j], j);
                    finished &= gear3[j] == -1.0;
                }
                if (!has_arg("--no-visualization")) {
                    for (unsigned j = 0; i % 10 == 0 && j < 6; j++) {
                        static_pointer_cast<RealisticRobotArmSimulationV2>(simulation)->setInferredAngles(j, gear1[j], gear2[j], gear3[j]);
                    }
                }
                
                log_str("Epoch: " + itoa(i) + ", retracting");
            }
            finished = false;
            for (unsigned i = 0; i < 100 && !finished; i++) {
                finished = true;
                for (unsigned j = 0; j < 10; j++) {
                    gear1[j] = std::max(0.0, fabs(gear1[j]) - 0.01) * sgn(gear1[j]);
                    gear2[j] = std::max(0.0, fabs(gear2[j]) - 0.01) * sgn(gear2[j]);
                    sendComandV2(arduino, gear1[j], gear2[j], gear3[j], j);
                    finished &= gear1[j] == 0.0 && gear2[j] == 0.0;
                }
                if (!has_arg("--no-visualization")) {
                    for (unsigned j = 0; i % 10 == 0 && j < 6; j++) {
                        static_pointer_cast<RealisticRobotArmSimulationV2>(simulation)->setInferredAngles(j, gear1[j], gear2[j], gear3[j]);
                    }
                }
                log_str("Epoch: " + itoa(i) + ", resetting");
            }
            serialport_flush(arduino);
            serialport_close(arduino);  
            exit(EXIT_SUCCESS);
        }
        while (true) Thread::yield();
    }

    //rand128_t *rand = new_rand128_t();

    opts.visualizeRobotArm(has_arg("-s", "--simulation"));
    opts.numJoints(parse_i_arg(10, "-j", "--numJoints"));
    opts.numInputNeurons(parse_i_arg(2, "--numInputs"));
    opts.numHiddenNeurons(parse_i_arg(128, "--numHidden"));
    opts.trainSetSize(parse_i_arg(1, "--batchSize"));
    opts.numSimulationTimesteps(parse_i_arg(12, "--time") * opts.numJoints());
    opts.decodingTimesteps(parse_i_arg(7, "--decodingTimesteps"));
    opts.regularizerLearnRate(parse_f_arg(0.0, "--fr-learnRate"));
    opts.regularizerLearnRateDecay(parse_f_arg(1.0, "--fr-decay"));
    opts.learnRate(parse_f_arg(0.0, "--learnRate"));
    opts.learnRateDecay(parse_f_arg(1.0, "--decay"));
    opts.positionErrorWeight(parse_f_arg(1.0, "--positionErrorWeight"));
    opts.upVectorErrorWeight(parse_f_arg(0.05, "--upVectorErrorWeight"));
    opts.xDirectionErrorWeight(parse_f_arg(0.05, "--xDirectionErrorWeight"));
    unsigned epochs = parse_i_arg(10000, "--epochs");
    opts.saveInterval(parse_i_arg(3600, "--saveInterval"));
    opts.adamBetaOne(parse_f_arg(0.9, "--beta1"));
    opts.adamBetaTwo(parse_f_arg(0.9, "--beta2"));
    opts.derivativeDumpingFactor(parse_f_arg(0.3, "--derivativeDumpingFactor"));

    if (has_arg("--gradientMonitor"))
        opts.gradientMonitor(true);

    if (has_arg("--angleMonitor"))
        opts.angleMonitor(true);

    if (has_arg("--inputErrorMonitor"))
        opts.inputErrorMonitor(true);

    if (has_arg("--outputReqularizer"))
        opts.outputRegularizer(true);

    if (!has_arg("--no-currentInput"))
        opts.currentInput(true);

    if (!has_arg("--no-zeroStart"))
        opts.zeroStart(true);

    if (has_arg("--interactiveTarget"))
        opts.interactiveTarget(true);

    opts.inferenceLearnRate(parse_f_arg(0.01, "--inferenceLearnRate"));
    opts.inferenceMomentum(parse_f_arg(0.3, "--inferenceMomentum"));
    opts.inferenceRegularizerLearnRate(parse_f_arg(0.001, "--inferenceRegularizerLearnRate"));
    opts.inferenceRegularizerMomentum(parse_f_arg(0.1, "--inferenceRegularizerMomentum"));
    opts.inputErrorMonitorStart(parse_i_arg(0, "--inputErrorMonitorStart"));
    opts.inputErrorMonitorEnd(parse_i_arg(opts.numInputNeurons(), "--inputErrorMonitorEnd"));
    opts.relaxationEpochs(parse_i_arg(2, "--relaxationEpochs"));
    opts.relaxationLearnRate(parse_f_arg(opts.inferenceLearnRate(), "--relaxationLearnRate"));

    opts.learnRateDecayIntervall(epochs / 10);
    opts.regularizerLearnRateDecayIntervall(epochs / 10);
    opts.numStandartHiddenNeurons(parse_i_arg(opts.numHiddenNeurons() / 2, "--standart-neurons"));
    opts.numAdaptiveHiddenNeurons(parse_i_arg(opts.numHiddenNeurons() / 2, "--adaptive-neurons"));
    opts.spikeThreshold(0.5);
    opts.numOutputNeurons(9);
    opts.debugPropability(parse_f_arg(0, "--debug"));
    opts.batchSize(opts.trainSetSize());
    opts.shuffleBatch(false);
    opts.regularizerOptimizer(OptimizerType::StochasticGradientDescent);

    if (has_arg("--quaternionRotation")) {
        opts.quaternionRotation(true);
        opts.numOutputNeurons(7);
    }

    if (!has_arg("--no-clockInput")) {
        
        opts.clockInput(true);
        opts.clockTimeSteps(parse_i_arg(opts.decodingTimesteps(), "--clockTimeSteps"));
        opts.numClockNeurons(parse_i_arg(opts.numJoints(), "--numClockNeurons"));
        opts.clockNeuronSpikeRate(parse_f_arg(10000, "--clockNeuronSpikeRate"));
        opts.numInputNeurons(opts.numInputNeurons() + opts.numClockNeurons());
    } 

    if (has_arg("--realistic-v2") || has_arg("--realistic-v3"))
        opts.numInputNeurons(opts.numInputNeurons() + 1);

    opts.useBackPropagation(!has_arg("--use-eligibility"));
    opts.optimizerUpdateInterval(opts.optimizerUpdateInterval() * 2);

    opts.momentum(parse_f_arg(0.1, "--momentum"));

    if (has_arg("--fr-optimizer") && get_arg("--fr-optimizer") == "SGD")
        opts.regularizerOptimizer(OptimizerType::StochasticGradientDescent);
    if (has_arg("--fr-optimizer") && get_arg("--fr-optimizer") == "Momentum")
        opts.regularizerOptimizer(OptimizerType::SignDampedMomentum);
    if (has_arg("--fr-optimizer") && get_arg("--fr-optimizer") == "Adam")
        opts.regularizerOptimizer(OptimizerType::Adam);

    if (has_arg("--optimizer") && get_arg("--optimizer") == "SGD")
        opts.optimizer(OptimizerType::StochasticGradientDescent);
    if (has_arg("--optimizer") && get_arg("--optimizer") == "Momentum")
        opts.optimizer(OptimizerType::SignDampedMomentum);
    if (has_arg("--optimizer") && get_arg("--optimizer") == "Adam")
        opts.optimizer(OptimizerType::Adam);

    if (has_arg("--inference-regularizer-optimizer") && get_arg("--inference-regularizer-optimizer") == "SGD")
        opts.inferenceRegularizerOptimizer(OptimizerType::StochasticGradientDescent);
    if (has_arg("--inference-regularizer-optimizer") && get_arg("--inference-regularizer-optimizer") == "Momentum")
        opts.inferenceRegularizerOptimizer(OptimizerType::SignDampedMomentum);
    if (has_arg("--inference-regularizer-optimizer") && get_arg("--inference-regularizer-optimizer") == "Adam")
        opts.inferenceRegularizerOptimizer(OptimizerType::Adam);

    if (has_arg("--inference-optimizer") && get_arg("--inference-optimizer") == "SGD")
        opts.inferenceOptimizer(OptimizerType::StochasticGradientDescent);
    if (has_arg("--inference-optimizer") && get_arg("--inference-optimizer") == "Momentum")
        opts.inferenceOptimizer(OptimizerType::SignDampedMomentum);
    if (has_arg("--inference-optimizer") && get_arg("--inference-optimizer") == "Adam")
        opts.inferenceOptimizer(OptimizerType::Adam);
    if (has_arg("--inference-optimizer") && get_arg("--inference-optimizer") == "AMSGrad")
        opts.inferenceOptimizer(OptimizerType::AMSGrad);

    opts.timeStepLength(parse_f_arg(1.0, "--timeStepLength"));
    opts.numSimulationTimesteps(opts.numSimulationTimesteps() / opts.timeStepLength());
    opts.optimizerUpdateInterval(opts.numSimulationTimesteps() * opts.batchSize());

    if (opts.inferenceOptimizer() == OptimizerType::SignDampedMomentum) {
        if (!has_arg("--learnRate"))
           opts.learnRate(0);
        if (!has_arg("--inferenceLearnRate"))
           opts.inferenceLearnRate(0.1);
        if (!has_arg("--fr-learnRate"))
            opts.regularizerLearnRate(0);
        if (!has_arg("--inferenceRegularizerLearnRate"))
           opts.inferenceRegularizerLearnRate(0);
    }

    if (!has_arg("--no-stats")) {
        printf("Back Propagation: %d\n", int(opts.useBackPropagation()));
        if (opts.inferenceOptimizer() == OptimizerType::Adam)
            printf("Optimizer:        Adam\n");
        if (opts.inferenceOptimizer() == OptimizerType::StochasticGradientDescent)
            printf("Optimizer:        StochasticGradientDescent\n");
        if (opts.inferenceOptimizer() == OptimizerType::SignDampedMomentum)
            printf("Optimizer:        SignDampedMomentum\n");
        if (opts.inferenceOptimizer() == OptimizerType::AMSGrad)
            printf("Optimizer:        AMSGrad\n");

        printf("numInputs:        %u\n", opts.numInputNeurons());
        printf("numHidden:        %u\n", opts.numHiddenNeurons());
        printf("numStandart:      %u\n", opts.numStandartHiddenNeurons());
        printf("numAdaptive:      %u\n", opts.numAdaptiveHiddenNeurons());
        printf("batchSize:        %u\n", opts.trainSetSize());
        printf("time:             %u\n", opts.numSimulationTimesteps());
        printf("decay:            %f\n", opts.learnRateDecay());
        printf("FR decay:         %f\n", opts.regularizerLearnRateDecay());
        printf("epochs:           %u\n", epochs);
        printf("numJoints:        %u\n", opts.numJoints());
        printf("FR factor:        %f\n", opts.regularizerLearnRate());
        printf("lernRate:         %f\n", opts.learnRate());
    }

    opts.training(false);
    opts.useInputErrors(true);
    opts.visualizeInference(true);

    if (has_arg("--cmdTarget")) {
        opts.cmdTarget(true);
        opts.targetX(parse_f_arg(0, "--targetX"));
        opts.targetY(parse_f_arg(0, "--targetY"));
        opts.targetZ(parse_f_arg(0, "--targetZ"));
        opts.targetXUp(parse_f_arg(0, "--targetXUp"));
        opts.targetYUp(parse_f_arg(0, "--targetYUp"));
        opts.targetZUp(parse_f_arg(0, "--targetZUp"));
        opts.targetXXDirection(parse_f_arg(0, "--targetXXDirection"));
        opts.targetYXDirection(parse_f_arg(0, "--targetYXDirection"));
        opts.targetZXDirection(parse_f_arg(0, "--targetZXDirection"));
    }

    if (!has_arg("--load")) {
        printf("missing network file!!\n");
        exit(EXIT_SUCCESS);
    }

    std::shared_ptr<BasicRobotArmSimulation> simulation;
    if (has_arg("--realistic-v2")) {

        simulation = static_pointer_cast<BasicRobotArmSimulation>(
            make_shared<RealisticRobotArmSimulationV2>(
                opts.numJoints(),
                parse_arg("", "--base"),
                parse_arg("", "--servo"),
                parse_arg("", "--linearGear"),
                parse_arg("", "--ballJoint"),
                parse_arg("", "--gear"),
                parse_arg("", "--stand"),
                parse_arg("", "--gripper-base"), 
                parse_arg("", "--gripper-servo"), 
                parse_arg("", "--gripper-actuator"), 
                parse_arg("", "--gripper"), 
                parse_arg("", "--gripper-gear"), 
                true,
                opts.visualizeRobotArm()
            )
        );
    } else if (has_arg("--realistic-v3")) {

        simulation = static_pointer_cast<BasicRobotArmSimulation>(
            make_shared<RealisticRobotArmSimulationV3>(
                opts.numJoints(),
                parse_arg("", "--base"),
                parse_arg("", "--first-module"),
                parse_arg("", "--servo"),
                parse_arg("", "--linearGear"),
                parse_arg("", "--ballJoint"),
                parse_arg("", "--gear"),
                parse_arg("", "--stand"),
                parse_arg("", "--gripper-base"), 
                parse_arg("", "--gripper-servo"), 
                parse_arg("", "--gripper-actuator"), 
                parse_arg("", "--gripper"), 
                parse_arg("", "--gripper-gear"), 
                true,
                opts.visualizeRobotArm()
            )
        );
    }

    if (has_arg("--realistic-v2")) {

        RealisticRobotArmV2InferenceNetwork network(opts);
        network.setSimulation(static_pointer_cast<RealisticRobotArmSimulationV2>(simulation));

        if (has_arg("--load"))
            network.load(get_arg("--load"), has_arg("--load-optimizers"));
        
        network.runInference(epochs, parse_arg("", "--save-inference"));

    } else if (has_arg("--realistic-v3")) {

        RealisticRobotArmV3InferenceNetwork network(opts);
        network.setSimulation(static_pointer_cast<RealisticRobotArmSimulationV3>(simulation));

        if (has_arg("--load"))
            network.load(get_arg("--load"), has_arg("--load-optimizers"));
        
        network.runInference(epochs, parse_arg("", "--save-inference"));
    } 
}
