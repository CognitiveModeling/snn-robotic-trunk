#include "RealisticRobotArmV2PredictorNetwork.h"
#include "RealisticRobotArmV3PredictorNetwork.h"
#include "utils.h"
#include "Image.h"
#include "BasicSynapseSerializer.h"
#include "FeedbackWeightsSerializer.h"
#include "LongShortTermMemorySparseKernelCaller.h"

using namespace SNN;
using namespace Networks;
using namespace Options;
using namespace Interfaces;
using namespace Visualizations;
using namespace Kernels;
using namespace GPU;

using std::string;
using std::vector;
using std::shared_ptr;
using std::make_shared;
using std::static_pointer_cast;

int main(int argc, char *argv[]) {

    uint32_t seed = parse_i_arg(gettime_usec(), "--seed");
    log_str("Random seed: " + itoa(seed));
        init_rand128(seed);

    if (has_arg("--convert-spikes")) {
        
        int fd = open(get_arg("--convert-spikes").c_str(), O_RDONLY);
        int width  = parse_i_arg(0, "--numSimulationTimesteps"); 
        int height = parse_i_arg(0, "--numHidden"); 
        FloatType scaleX = parse_f_arg(1, "--scaleX");
        FloatType scaleY = parse_f_arg(1, "--scaleY");
        int spikeSize = parse_i_arg(0, "--spikeRadius");
        bool transparent = has_arg("--transparent");
        int red = parse_i_arg(35, "--red");
        int green = parse_i_arg(55, "--green");
        int blue = parse_i_arg(59, "--blue");
        if (fd > 0) {
            vector<vector<int>> spikes;

            for (int t = 0; t < width; t++) {

                spikes.push_back(vector<int>());
                int size = 0;
                assert(read(fd, &size, sizeof(int)) == sizeof(int));
                for (int i = 0; i < size; i++) {
                    int value = 0;
                    assert(read(fd, &value, sizeof(int)) == sizeof(int));
                    spikes.back().push_back(value);
                }
            }

            Image img(
                width*scaleX+2*(spikeSize - std::min(scaleX, scaleY)/2) + 2, 
                scaleY*height+2*(spikeSize - std::min(scaleX, scaleY)/2) + 2, 
                4, int(!transparent) * 255
            );

            for (int x = 0; x < width; x++) {
                for (int y: spikes[x]) {
                    
                    int xPos = scaleX*x + scaleX * 0.5;
                    int yPos = scaleY*y + scaleY * 0.5;
                    for (int px = xPos; px <= xPos + 2*spikeSize + 2; px++) {
                        for (int py = yPos; py <= yPos + 2*spikeSize + 2; py++) {
                            if (px >= 0 && px < img.getWidth() && py >= 0 && py < img.getHeight()) {
                                
                                FloatType l = sqrt(pow(px - xPos - spikeSize - 1, 2) + 
                                              pow(py - yPos - spikeSize - 1, 2)) / (1 + spikeSize);

                                int offset = py * img.getWidth() * 4 + px * 4;

                                if (transparent) {
                                    img[offset + 3] = std::max<int>(img[py * img.getWidth() * 4 + px * 4 + 3], 255 * (1 - (tanh(l*6-3)/2 + 0.5)));
                                    img[offset + 0] = red;
                                    img[offset + 1] = green;
                                    img[offset + 2] = blue;
                                } else {
                                    int tmp = std::min<int>(img[py * img.getWidth() * 4 + px * 4], 255 * (tanh(l*6-3)/2 + 0.5));
                                    img[offset + 0] = tmp;
                                    img[offset + 1] = tmp;
                                    img[offset + 2] = tmp;
                                }
                            }
                        }
                    }
                }
            }
            img.save("spikes-" + itoa(parse_i_arg(0, "--spike-img-index")) + ".png");
            close(fd);
        }
        return 0;
    }       

    if (has_arg("--plot-spikes")) {
        
        int fd = open(get_arg("--plot-spikes").c_str(), O_RDONLY);
        int width  = parse_i_arg(0, "--numSimulationTimesteps"); 
        if (fd > 0) {
            vector<vector<int>> spikes;

            for (int t = 0; t < width; t++) {

                spikes.push_back(vector<int>());
                int size = 0;
                assert(read(fd, &size, sizeof(int)) == sizeof(int));
                for (int i = 0; i < size; i++) {
                    int value = 0;
                    assert(read(fd, &value, sizeof(int)) == sizeof(int));
                    spikes.back().push_back(value);
                }
            }

            for (int x = 0; x < width; x++) {
                for (int y: spikes[x]) {
                    printf("(%d,%d)\n", x, y);
                }
            }
        }
        return 0;
    }
                    

    LongShortTermMemorySparseKernelCaller::setDevice(parse_i_arg(0, "--device"));

    RobotArmPredictorNetworkOptions opts;

    opts.visualizeRobotArm(has_arg("-s", "--simulation"));
    opts.numJoints(parse_i_arg(10, "-j", "--numJoints"));
    opts.numInputNeurons(parse_i_arg(2, "--numInputs"));
    opts.numHiddenNeurons(parse_i_arg(128, "--numHidden"));
    opts.trainSetSize(parse_i_arg(32, "--batchSize"));
    opts.numSimulationTimesteps(parse_i_arg(12, "--time") * opts.numJoints());
    opts.decodingTimesteps(parse_i_arg(7, "--decodingTimesteps"));
    opts.regularizerLearnRate(parse_f_arg(0.001, "--fr-learnRate"));
    opts.regularizerLearnRateDecay(parse_f_arg(1.0, "--fr-decay"));
    opts.learnRateDecay(parse_f_arg(1.0, "--decay"));
    opts.positionErrorWeight(parse_f_arg(1.0, "--positionErrorWeight"));
    opts.upVectorErrorWeight(parse_f_arg(0.05, "--upVectorErrorWeight"));
    opts.xDirectionErrorWeight(parse_f_arg(0.05, "--xDirectionErrorWeight"));
    unsigned epochs = parse_i_arg(10000, "--epochs");
    opts.saveInterval(parse_i_arg(3600, "--saveInterval"));
    opts.edjeCasePercent(parse_f_arg(0.25, "--edjeCasePercent"));
    opts.refactoryPeriod(parse_i_arg(5, "--refactoryPeriod"));
    opts.adamWeightDecay(parse_i_arg(0, "--weightDecay"));
    opts.trainingPosesFile(parse_arg("", "--save-training-poses"));
    
    if (has_arg("--saveSimulation")) {
        opts.saveSimulation(true);
        opts.simulationFile(get_arg("--saveSimulation"));
    }
    if (has_arg("--loadSimulation")) {
        opts.loadSimulation(true);
        opts.simulationFile(get_arg("--loadSimulation"));
    }

    if (has_arg("--adaptive-lr"))
        opts.adaptiveLearningRate(true);

    if (has_arg("--smart-trainset"))
        opts.useSmartTrainSet(true);

    if (has_arg("--no-training")) {
        opts.learnRate(0);
        opts.regularizerLearnRate(0);
    }

    opts.training(!has_arg("--no-training"));
    opts.learnRateDecayIntervall(parse_i_arg(std::max<int>(1, epochs / 10), "--decay-interval"));
    opts.regularizerLearnRateDecayIntervall(parse_i_arg(std::max<int>(1, epochs / 10), "--decay-interval"));
    opts.numStandartHiddenNeurons(parse_i_arg(opts.numHiddenNeurons() / 2, "--standart-neurons"));
    opts.numAdaptiveHiddenNeurons(parse_i_arg(opts.numHiddenNeurons() / 2, "--adaptive-neurons"));
    opts.spikeThreshold(0.5);
    opts.learnRate(parse_f_arg(0.001, "--learnRate"));
    opts.numOutputNeurons(9);
    opts.debugPropability(parse_f_arg(0, "--debug"));
    opts.batchSize(opts.trainSetSize());
    opts.shuffleBatch(false);
    opts.regularizerOptimizer(OptimizerType::StochasticGradientDescent);
    opts.derivativeDumpingFactor(parse_f_arg(0.3, "--derivativeDumpingFactor"));
    opts.adamBetaOne(parse_f_arg(0.9, "--beta1"));
    opts.adamBetaTwo(parse_f_arg(0.999, "--beta2"));
    opts.adamEpsilon(parse_f_arg(1e-8, "--epsilon"));
    opts.inputWeightFactor(parse_f_arg(1.0, "--inputWeightFactor"));
    opts.hiddenWeightFactor(parse_f_arg(1.0, "--hiddenWeightFactor"));
    opts.outputWeightFactor(parse_f_arg(1.0, "--outputWeightFactor"));
    opts.pruneStart(parse_i_arg(-1, "--pruneStart"));
    opts.pruneIntervall(parse_i_arg(1000, "--pruneIntervall"));
    opts.pruneStrength(parse_f_arg(2.0, "--pruneStrength"));
    opts.useInputErrors(true);
    opts.eprop3Interval(parse_i_arg(12, "--eprop3Interval"));
    opts.targetFiringRate(parse_f_arg(10, "--targetFiringRate")/1000.0);

    if (has_arg("--quaternionRotation")) {
        opts.quaternionRotation(true);
        opts.numOutputNeurons(7);
    }

    if (has_arg("--symetricEprop1"))
        opts.symetricEprop1(true);

    if (!has_arg("--no-clockInput")) {
        
        opts.clockInput(true);
        opts.clockTimeSteps(opts.decodingTimesteps());
        opts.numInputNeurons(opts.numInputNeurons() + parse_i_arg(opts.numJoints(), "--numClockNeurons"));
    } 

    if (has_arg("--realistic-v2"))
        opts.numInputNeurons(opts.numInputNeurons() + 1);

    if (has_arg("--realistic-v3")) {
        opts.numInputNeurons(opts.numInputNeurons() + 1);
        opts.edjeCasePercent(parse_f_arg(0.5, "--edjeCasePercent"));
    }


    if (has_arg("--back-propagation")) {
        opts.useBackPropagation(true);
        opts.optimizerUpdateInterval(opts.optimizerUpdateInterval() * 2);
    }

    if (has_arg("--eligibility-back-propagation")) {
        opts.useBackPropagation(true);
        opts.useEligibilityBackPropagation(true);
        opts.optimizerUpdateInterval(opts.optimizerUpdateInterval() * 2);
    }

    if (has_arg("--izhikevich")) {
        opts.timeStepLength(0.25);
        opts.optimizerUpdateInterval(opts.optimizerUpdateInterval() * 4);
        opts.numSimulationTimesteps(opts.numSimulationTimesteps() * 4);
        opts.decodingTimesteps(opts.decodingTimesteps() * 4);
        opts.izhikevich(true);
        opts.inputWeightFactor(parse_f_arg(20, "--inputWeightFactor"));
        opts.hiddenWeightFactor(parse_f_arg(20, "--hiddenWeightFactor"));
    }

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
    if (has_arg("--optimizer") && get_arg("--optimizer") == "AMSGrad")
        opts.optimizer(OptimizerType::AMSGrad);

    opts.timeStepLength(parse_f_arg(1.0, "--timeStepLength"));
    opts.numSimulationTimesteps(opts.numSimulationTimesteps() / opts.timeStepLength());
    opts.optimizerUpdateInterval(opts.numSimulationTimesteps() * opts.batchSize());

    if (has_arg("--generate-dataset") && has_arg("--dataset-name")) {
        opts.datasetGeneration(true);
        opts.saveSimulation(true);
        opts.simulationFile(get_arg("--dataset-name"));
        opts.numHiddenNeurons(2);
        opts.numAdaptiveHiddenNeurons(1);
        opts.numStandartHiddenNeurons(1);
        epochs = parse_i_arg(opts.batchSize(), "--generate-dataset") / opts.batchSize();
    }

    if (!has_arg("--no-stats") && !opts.datasetGeneration()) {
        printf("Back Propagation: %d\n", int(opts.useBackPropagation()));
        if (opts.optimizer() == OptimizerType::Adam)
            printf("Optimizer:        Adam\n");
        if (opts.optimizer() == OptimizerType::StochasticGradientDescent)
            printf("Optimizer:        StochasticGradientDescent\n");
        if (opts.optimizer() == OptimizerType::SignDampedMomentum)
            printf("Optimizer:        SignDampedMomentum\n");

        if (opts.regularizerOptimizer() == OptimizerType::Adam)
            printf("FR Optimizer:     Adam\n");
        if (opts.regularizerOptimizer() == OptimizerType::StochasticGradientDescent)
            printf("FR Optimizer:     StochasticGradientDescent\n");
        if (opts.regularizerOptimizer() == OptimizerType::SignDampedMomentum)
            printf("FR Optimizer:     SignDampedMomentum\n");


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

    std::shared_ptr<LongShortTermMemoryEligibilityNetwork> network;

    if (has_arg("--sparse-network")) {

        FloatType inputConnectifity = parse_f_arg(1, "--input-connectifity");
        FloatType outputConnectifity = parse_f_arg(1, "--output-connectifity");
        FloatType hiddenConnectifity = parse_f_arg(1, "--hidden-connectifity");
        FloatType feedbackConnectifity = parse_f_arg(1, "--feedback-connectifity");
        unsigned numAdaptiveInputSynapses        = 0;
        unsigned numAdaptiveHiddenSynapses       = 0;
        std::vector<FloatType> inputWeights      = std::vector<FloatType>();
        std::vector<FloatType> hiddenWeights     = std::vector<FloatType>(); 
        std::vector<FloatType> outputWeights     = std::vector<FloatType>(); 
        std::vector<FloatType> feedbackWeights   = std::vector<FloatType>(); 
        std::vector<unsigned> inputWeightsIn     = std::vector<unsigned>();
        std::vector<unsigned> hiddenWeightsIn    = std::vector<unsigned>(); 
        std::vector<unsigned> outputWeightsIn    = std::vector<unsigned>(); 
        std::vector<unsigned> feedbackWeightsIn  = std::vector<unsigned>(); 
        std::vector<unsigned> inputWeightsOut    = std::vector<unsigned>(); 
        std::vector<unsigned> hiddenWeightsOut   = std::vector<unsigned>(); 
        std::vector<unsigned> outputWeightsOut   = std::vector<unsigned>(); 
        std::vector<unsigned> feedbackWeightsOut = std::vector<unsigned>();

        for (unsigned i = 0; i < opts.numInputNeurons(); i++) {
            for (unsigned h = 0; h < opts.numStandartHiddenNeurons(); h++) {
                if (rand_range_d128(0.0, 1.0) < inputConnectifity) {
                    inputWeights.push_back(rand128n(0.0, 1.0) / sqrt(opts.numInputNeurons() * inputConnectifity));
                    inputWeightsIn.push_back(i);
                    inputWeightsOut.push_back(h);
                }
            }
        }
        for (unsigned i = 0; i < opts.numInputNeurons(); i++) {
            for (unsigned h = opts.numStandartHiddenNeurons(); h < opts.numHiddenNeurons(); h++) {
                if (rand_range_d128(0.0, 1.0) < inputConnectifity) {
                    inputWeights.push_back(rand128n(0.0, 1.0) / sqrt(opts.numInputNeurons() * inputConnectifity));
                    inputWeightsIn.push_back(i);
                    inputWeightsOut.push_back(h);
                    numAdaptiveInputSynapses++;
                }
            }
        }
        for (unsigned hi = 0; hi < opts.numHiddenNeurons(); hi++) {
            for (unsigned ho = 0; ho < opts.numStandartHiddenNeurons(); ho++) {
                if (rand_range_d128(0.0, 1.0) < hiddenConnectifity) {
                    hiddenWeights.push_back(rand128n(0.0, 1.0) / sqrt(opts.numHiddenNeurons() * hiddenConnectifity));
                    hiddenWeightsIn.push_back(hi);
                    hiddenWeightsOut.push_back(ho);
                }
            }
        }
        for (unsigned hi = 0; hi < opts.numHiddenNeurons(); hi++) {
            for (unsigned ho = opts.numStandartHiddenNeurons(); ho < opts.numHiddenNeurons(); ho++) {
                if (rand_range_d128(0.0, 1.0) < hiddenConnectifity) {
                    hiddenWeights.push_back(rand128n(0.0, 1.0) / sqrt(opts.numHiddenNeurons() * hiddenConnectifity));
                    hiddenWeightsIn.push_back(hi);
                    hiddenWeightsOut.push_back(ho);
                    numAdaptiveHiddenSynapses++;
                }
            }
        }
        for (unsigned h = 0; h < opts.numHiddenNeurons(); h++) {
            for (unsigned o = 0; o < opts.numOutputNeurons(); o++) {
                if (rand_range_d128(0.0, 1.0) < outputConnectifity) {
                    outputWeights.push_back(rand128n(0.0, 1.0) / sqrt(opts.numHiddenNeurons() * outputConnectifity));
                    outputWeightsIn.push_back(h);
                    outputWeightsOut.push_back(o);
                }
                if (rand_range_d128(0.0, 1.0) < feedbackConnectifity) {
                    feedbackWeights.push_back(rand128n(0.0, 1.0) / sqrt(opts.numHiddenNeurons() * feedbackConnectifity));
                    feedbackWeightsIn.push_back(h);
                    feedbackWeightsOut.push_back(o);
                }
            }
        }
        printf("num input Synapses:   %lu\n", inputWeights.size());
        printf("num hidden Synapses:   %lu\n", hiddenWeights.size());
        printf("num output Synapses:   %lu\n", outputWeights.size());
        printf("num feedback Synapses: %lu\n", feedbackWeights.size());

        printf("num adaptive input Synapses:    %u\n", numAdaptiveInputSynapses); 
        printf("num adaptive hidden Synapses:   %u\n", numAdaptiveHiddenSynapses);
        if (has_arg("--realistic-v2")) {
            network = std::make_shared<RealisticRobotArmV2PredictorNetwork>(
                opts,
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
            );
        } else if (has_arg("--realistic-v3")) {
            network = std::make_shared<RealisticRobotArmV3PredictorNetwork>(
                opts,
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
            );
        } 
    } else {

        if (has_arg("--realistic-v2")) {
            network = std::make_shared<RealisticRobotArmV2PredictorNetwork>(opts);
        } else if (has_arg("--realistic-v3")) {
            network = std::make_shared<RealisticRobotArmV3PredictorNetwork>(opts);
        } 
    }

    std::shared_ptr<BasicRobotArmSimulation> simulation;
    if (has_arg("--realistic-v2")) {
        printf("Realistic Simulation V2\n");

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
                false,
                opts.visualizeRobotArm()
            )
        );

    } else if (has_arg("--realistic-v3")) {
        printf("Realistic Simulation V3\n");

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
                has_arg("--save-training-poses"),
                opts.visualizeRobotArm()
            )
        );
    }

    if (has_arg("--random-pose-screenshot")) {
        for (int i = 0;;i++) {
            
            log_str("Epoch: " + itoa(i));
            simulation->randomPose();

            char path[256];
            sprintf(path, "simulation-random-pose-%05d.png", i);
            simulation->saveScreenshot(std::string(path), 1);
        }
    }

    if (has_arg("--realistic-v2")) {
        std::static_pointer_cast<RealisticRobotArmV2PredictorNetwork>(network)->setSimulation(std::static_pointer_cast<RealisticRobotArmSimulationV2>(simulation));
    } else if (has_arg("--realistic-v3")) {
        std::static_pointer_cast<RealisticRobotArmV3PredictorNetwork>(network)->setSimulation(std::static_pointer_cast<RealisticRobotArmSimulationV3>(simulation));
    } 

    if (has_arg("--load"))
        network->load(get_arg("--load"), has_arg("--load-optimizers"), has_arg("--load-learnRate"));

    if (has_arg("--sparse"))
        network->makeSparse();

    if (has_arg("--createImages")) {
        shared_ptr<BasicSynapseSerializer> inputSynapses  = network->getInputSynapsesSerializer();
        shared_ptr<BasicSynapseSerializer> hiddenSynapses = network->getHiddenSynapsesSerializer();
        shared_ptr<BasicSynapseSerializer> outputSynapses = network->getOutputSynapsesSerializer();
        shared_ptr<FeedbackWeightsSerializer> feedbackSynapses = network->getFeedbackSynapsesSerializer();

        inputSynapses->serialize();
        hiddenSynapses->serialize();
        outputSynapses->serialize();
        feedbackSynapses->serialize();

        inputSynapses->saveAsImage("LSNN-I2H.png", opts.numInputNeurons(), opts.numHiddenNeurons());
        hiddenSynapses->saveAsImage("LSNN-H2H.png", opts.numHiddenNeurons(), opts.numHiddenNeurons());
        outputSynapses->saveAsImage("LSNN-H2O.png", opts.numHiddenNeurons(), opts.numOutputNeurons());
        feedbackSynapses->saveAsImage("LSNN-O2H.png", opts.numHiddenNeurons(), opts.numOutputNeurons());
    }

    if (has_arg("--save-weights")) {
        int fdi = open(string("LSNN-input.weights").c_str(), O_CREAT|O_TRUNC|O_WRONLY, 00777);
        int fdh = open(string("LSNN-hidden.weights").c_str(), O_CREAT|O_TRUNC|O_WRONLY, 00777);
        int fdo = open(string("LSNN-output.weights").c_str(), O_CREAT|O_TRUNC|O_WRONLY, 00777);

        int numInputs  = network->getInputSynapsesSerializer()->size();
        int numHidden  = network->getHiddenSynapsesSerializer()->size();
        int numOutputs = network->getOutputSynapsesSerializer()->size();

        assert(write(fdi, &numInputs, sizeof(int)) == sizeof(int));
        assert(write(fdh, &numHidden, sizeof(int)) == sizeof(int));
        assert(write(fdo, &numOutputs, sizeof(int)) == sizeof(int));

        network->getInputSynapsesSerializer()->save(fdi);
        network->getHiddenSynapsesSerializer()->save(fdh);
        network->getOutputSynapsesSerializer()->save(fdo);

        close(fdi);
        close(fdh);
        close(fdo);
    }

    if (has_arg("--realistic-v2")) {
        std::static_pointer_cast<RealisticRobotArmV2PredictorNetwork>(network)->train(epochs, has_arg("--evaluate"), LOG_I);
    } else if (has_arg("--realistic-v3")) {
        std::static_pointer_cast<RealisticRobotArmV3PredictorNetwork>(network)->train(epochs, has_arg("--evaluate"), has_arg("--save-spikes"), LOG_I);
    } 
}
