#ifndef __BASIC_NEURON__
#define __BASIC_NEURON__
#include "utils.h"
#include "BasicNetworkOptions.h"
/**
 * A Base class for LeakyIntegrateAndFireNeurons and AdaptiveLeakyIntegrateAndFireNeurons
 */
namespace SNN {

    namespace Interfaces {

        class BasicNeuron {

            private:

                /* te simulation batch counter */
                int batchCounter;

                /* the simulation time step counter */
                int timeStepCounter;

                /* the length of one simulation timestep in milliseconds */
                FloatType timeStepLength;

                /* the batch size */
                unsigned batchSize;

                /* the number of simulation timesteps */
                unsigned numSimulationTimesteps;

                /* the number this has fired */
                unsigned numSpikes;

                /* the target firing rate */
                FloatType targetFiringRate;

                /* the firing rate during the last batch (bevor the last reset) */
                FloatType firingRate;

                /* the id of this */
                uint64_t id;

                /* vector of voltage for back propagation */
                std::vector<FloatType> derivatives;

                /* vector of neuron outputs (spikes) over time */
                std::vector<FloatType> outputsOverTime;

                /* vector of delta errors over time */
                std::vector<FloatType> errorsOverTime;

            protected:

                /* indicates wether beack propagation or synaptic delays are enabled or not */
                bool storeValuesOverTime;

                /* increas or decreas timestep */
                void incTime() { timeStepCounter++; }
                void decTime() { timeStepCounter--; }

            private:

                /* the delta error of this (back propagation)
                 * (voltage component since others are not needed) */
                FloatType deltaErrorVoltage;

                /* weighted summ of delta errors from connected neurons */
                FloatType learnSignal;

                /* learnsignal from the last backpropagation step */
                FloatType learnSignalLast;

                /* eligibility based error of this */
                FloatType eligibilityError;

                /* eligibility based error of this (inputs) */
                FloatType eligibilityErrorInput;

                /* should compute the next timestep */
                virtual void doUpdate() = 0;

                /* should reset this */
                virtual void doReset() = 0;

            public:

                /* constructor */
                BasicNeuron(BasicNetworkOptions &opts): 
                    batchCounter(0),
                    timeStepCounter(0), 
                    timeStepLength(opts.timeStepLength()),
                    batchSize(opts.batchSize()),
                    numSimulationTimesteps(opts.numSimulationTimesteps()),
                    numSpikes(0),
                    targetFiringRate(opts.targetFiringRate()),
                    firingRate(opts.targetFiringRate()),
                    id(genUID()),
                    storeValuesOverTime(opts.useBackPropagation() | opts.useSynapticDelays()),
                    deltaErrorVoltage(0),
                    learnSignal(0),
                    learnSignalLast(0),
                    eligibilityError(0),
                    eligibilityErrorInput(0) {
                    
                    assert(timeStepLength < 100 && timeStepLength > 0.01 && 
                           "timstep length should be in milliseconds");
                }

                /* virtual destructor for correct polymophism */
                virtual ~BasicNeuron() { }

                /* should reset this */
                virtual void reset() {
                    batchCounter = 0;
                    timeStepCounter = 0;
                    numSpikes = 0;
                    firingRate = targetFiringRate;
                    deltaErrorVoltage = 0;
                    learnSignal = 0;
                    learnSignalLast = 0;
                    eligibilityError = 0;
                    eligibilityErrorInput = 0;
                    derivatives.clear();
                    outputsOverTime.clear();
                    errorsOverTime.clear();
                    this->doReset();
                }

                /* should reset neccesary parameters for a next run */
                virtual void networkReset() {
                    batchCounter++;
                    this->doReset();

                    /* calculates the firing rate in firings per millisecond */
                    if (batchCounter % batchSize == 0) {
                        calculateFiringRate();
                        numSpikes = 0;
                    }
                    timeStepCounter   = 0;
                    deltaErrorVoltage = 0;
                    learnSignal       = 0;
                    learnSignalLast   = 0;
                    eligibilityError  = 0;
                    eligibilityErrorInput = 0;
                    derivatives.clear();
                    outputsOverTime.clear();
                    errorsOverTime.clear();
                }

                /* calculates the firing rate in spike per millisecond */
                void calculateFiringRate() {
                    firingRate = numSpikes / (batchSize * numSimulationTimesteps * timeStepLength);
                }
                FloatType calculateFiringRate(unsigned numSpikes) {
                    return numSpikes / (numSimulationTimesteps * timeStepLength);
                }

                /* sets the firing rate */
                void setFiringRate(FloatType firingRate) {
                    this->firingRate = firingRate;
                }

                /* should add a input current to this */
                virtual void addCurrent(FloatType current) = 0;

                /* computes the next timestep */
                void update() {
                    this->doUpdate();
                    timeStepCounter++;
                    assert(timeStepCounter > 0);

                    if (this->fired())
                        numSpikes++;

                    if (storeValuesOverTime) {
                        derivatives.push_back(this->getDerivative());
                        outputsOverTime.push_back(this->getOutput());
                        errorsOverTime.push_back(0);
                    }

                    eligibilityError = eligibilityErrorInput;
                    eligibilityErrorInput = 0;
                }

                /* collect the weighted delta error */
                virtual void addError(FloatType error) {
                    learnSignal += error;
                }

                /* collect the eligibility error */
                void addEligibilityError(FloatType error) {
                    eligibilityErrorInput += error;
                }

                /* should compute the voltage component of next delta error 
                 * (we need voltage component only since other components 
                 *  will get zero weighted during learn signal computations)
                 */
                virtual FloatType computeDeltaErrorVoltage(
                    FloatType learnSignal, 
                    FloatType derivative
                ) = 0;

                /* should compute the reset value for the learn signal */
                virtual FloatType resetLearnSignal() {
                    return 0;
                }

                /* back propagation of errors (calculate delta error) */
                virtual void backPropagate() {
                    assert(timeStepCounter > 0);
                    assert(storeValuesOverTime);
                    timeStepCounter--;

                    deltaErrorVoltage = this->computeDeltaErrorVoltage(
                        learnSignal, 
                        derivatives[timeStepCounter]
                    );
                    learnSignalLast = learnSignal;
                    learnSignal = this->resetLearnSignal();
                    errorsOverTime[timeStepCounter] = deltaErrorVoltage;
                }

                /* should set delta errors within the chield class */
                virtual void doSetDeltaError(std::vector<FloatType>) {
                    log_err("doSetDeltaError not implemented in child class", LOG_E);
                }

                /* sets the delta error of this */
                void setDeltaError(std::vector<FloatType> errors) {
                    this->doSetDeltaError(errors);
                    this->deltaErrorVoltage = errors[0];
                    this->learnSignal = this->resetLearnSignal();
                }
                
                /* returns the error of this */
                virtual FloatType getError() {
                    return deltaErrorVoltage;
                }

                /* returns the error of this Neuron from the given timestep */
                FloatType getError(int timeStep) {
                    if (timeStep >= 0 && timeStep < int(errorsOverTime.size()))
                        return errorsOverTime[timeStep];

                    return 0;
                }

                /* returns the eligibility error of this */
                FloatType getEligibilityError() {
                    return eligibilityError;
                }

                /* returns the learn signal as computed within the last backpropagation update */
                virtual FloatType getLearnSignal() {
                    return learnSignalLast;
                }

                /* sets the error of this */
                void setError(FloatType error) {
                    deltaErrorVoltage = error;
                }

                /* sets the learn signal of this */
                void setLearnSignal(FloatType learnSignal) {
                    this->learnSignal = learnSignal;
                }

                /* should return whether this neuron has fired during the current time step */
                virtual bool fired() = 0;

                /* should return the output of this Neuron 
                 * (1 if fired 0 otherwise for spiking neurons) */
                virtual FloatType getOutput() = 0;

                /* returns the output of this Neuron from th given timestep */
                FloatType getOutput(int timeStep) {
                    assert(timeStep >= 0 && timeStep < int(outputsOverTime.size()));
                    return outputsOverTime[timeStep];
                }

                /* should return the (pseudo) derivative of this */
                virtual FloatType getDerivative() = 0;

                /* should return the voltage of this */
                virtual FloatType getVoltage() = 0;

                /* should return the input current of this */
                virtual FloatType getInputCurrent() = 0;

                /* should set the voltage of this */
                virtual void setVoltage(FloatType) = 0;

                /* should set the input current of this */
                virtual void setInputCurrent(FloatType) = 0;

                /* should set the derivative of this */
                virtual void setDerivative(FloatType) = 0;

                /* returns the current simulation time step of this */
                int getSimulationTimeStep() {
                    return timeStepCounter;
                }

                /* sets the current simulation time step of this */
                void setSimulationTimeStep(int timeStepCounter) {
                    this->timeStepCounter = timeStepCounter;
                }

                /* returns the firing rate in spikes per millisecond */
                FloatType getFiringRate() {
                    return firingRate;
                }

                /* returns the number of spikes */
                unsigned getNumSpikes() {
                    return numSpikes;
                }

                /* sets the number of spikes */
                void setNumSpikes(unsigned numSpikes) {
                    this->numSpikes = numSpikes;
                }

                /* comparison */
                bool operator == (const BasicNeuron &other) { return this->id == other.id; }

                /* should return the threshold of this */
                virtual FloatType getThreshold() { return 0; }

                /* returns the pseudo-derivative of this from the given timestep */
                virtual FloatType getDerivative(unsigned t) {
                    return derivatives[t];
                }

        };

    }

}
#endif /* __BASIC_NEURON__ */
