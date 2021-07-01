#ifndef __SMART_TRAIN_SET_H__
#define __SMART_TRAIN_SET_H__

#include "Threads.h"
#include "LongShortTermMemoryEligibilityNetworkOptions.h"
#include <limits.h>

/* smart trainset that chooses the next sample propabilistically based on its previous error */
namespace SNN {

    /* forward declaration */
    class SmartTrainSet;
    
    /* storage class for a smart sample */
    class SmartSample {

        friend class SmartTrainSet;
        
        private:

            /* the options of this */
            Options::LongShortTermMemoryEligibilityNetworkOptions &opts;

            /* the targets over time for one simulatrion run */
            FloatType *targetsOverTime;

            /* the error mask over time for one simulatrion run */
            FloatType *errorMaskOverTime;

            /* the inputs over time for one simulatrion run */
            FloatType *inputsOverTime;

            /* the error of this */
            FloatType error;

        public:

            /* constructor */
            SmartSample(Options::LongShortTermMemoryEligibilityNetworkOptions &opts): opts(opts) {
                this->targetsOverTime   = (FloatType *) malloc(sizeof(FloatType) * opts.numOutputNeurons() * opts.numSimulationTimesteps());
                this->errorMaskOverTime = (FloatType *) malloc(sizeof(FloatType) * opts.numSimulationTimesteps());
                this->inputsOverTime    = (FloatType *) malloc(sizeof(FloatType) * opts.numInputNeurons() * opts.numSimulationTimesteps());
                this->error             = std::numeric_limits<FloatType>::infinity();
            }

            /* destructor */
            ~SmartSample() {
                free(this->targetsOverTime);
                free(this->errorMaskOverTime);
                free(this->inputsOverTime);
                this->error = std::nan("SmartSample");
            }

            /* copies the content of this into the given arrays */
            void copy(FloatType *targetsOverTime, FloatType *errorMaskOverTime, FloatType *inputsOverTime) {
                memcpy(targetsOverTime,   this->targetsOverTime,   sizeof(FloatType) * opts.numOutputNeurons() * opts.numSimulationTimesteps());
                memcpy(errorMaskOverTime, this->errorMaskOverTime, sizeof(FloatType) * opts.numSimulationTimesteps());
                memcpy(inputsOverTime,    this->inputsOverTime,    sizeof(FloatType) * opts.numInputNeurons() * opts.numSimulationTimesteps());
            }

            /* updates the error of this */
            void updateError(FloatType error) {
                assert(error > 0);
                this->error = error;
            }

            /* returns the error of this */
            FloatType getError() { return this->error; }

    };

    class SmartTrainSet: protected Thread {

        private:

            /* the options of this */
            Options::LongShortTermMemoryEligibilityNetworkOptions &opts;
        
            /* the training smaples of this */
            std::vector<std::shared_ptr<SmartSample>> samples;

            /* the max size of this */
            unsigned maxSize;

            /* the max reachted sized */
            unsigned maxReachSize;

            /* the running statistics (mean + std of the error distribution of this  */
            RunningStatistics stats;

            /* synconization mutex */
            Mutex mutex;

            /* indicates that this is running */
            bool running;

            /* thread function child class */
            virtual void threadFunction(void *) {
                while (running) {
                    if (samples.size() >= maxSize)
                        Thread::yield();
                    else {

                        std::shared_ptr<SmartSample> sample = std::make_shared<SmartSample>(opts);
                        this->createSample(
                            sample->targetsOverTime,
                            sample->errorMaskOverTime,
                            sample->inputsOverTime
                        );

                        mutex.lock();
                        samples.push_back(sample);
                        maxReachSize = std::max<unsigned>(samples.size(), maxReachSize);
                        mutex.unlock();
                    }
                }
            }

        protected:
            
            /* child function to calculate a new sample */
            virtual void createSample(
                FloatType *targetsOverTime,
                FloatType *errorMaskOverTime,
                FloatType *inputsOverTime
            ) = 0;

        public:

            /* constructor */
            SmartTrainSet(
                Options::LongShortTermMemoryEligibilityNetworkOptions &opts, 
                unsigned maxSize,
                bool singleThreaded = false
            ) : opts(opts), maxSize(maxSize), maxReachSize(1), running(!singleThreaded) { 
                if (!singleThreaded)
                    this->run();    
            }

            /* destructor */
            ~SmartTrainSet() {
                if (this->running) {
                    this->running = false;
                    this->join();
                }
            }

            /* returns the size of this */
            unsigned size() { return samples.size(); }

            /* returns the randomly choosen next sample */
            std::shared_ptr<SmartSample> getNextSample() {
                mutex.lock();
                assert(samples.size() > 1);

                FloatType randError = 0;
                if (stats.getSum() > 0)
                    randError = rand128n(stats.getMean(), stats.getSTD());

                for (unsigned i = 1;/* loop indefinitly */; i++) {
                    unsigned index = rand128() % samples.size();

                    if (samples[index]->getError() > randError || i > maxReachSize) {

                        std::shared_ptr<SmartSample> sample = samples[index];
                        samples[index] = samples.back();
                        samples.resize(samples.size() - 1);
                        mutex.unlock();
                        return sample;
                    }

                    if (i % 1000 == 0)
                        randError = rand128n(stats.getMean(), stats.getSTD());
                }
            }

            /* readds the given sample after and error update */
            void readd(std::shared_ptr<SmartSample> sample) {
                assert(sample->getError() < std::numeric_limits<FloatType>::infinity());
                assert(sample->getError() > 0);
                assert(!std::isnan(sample->getError()));

                stats.add(sample->getError(), maxReachSize);

                if (sample->getError() > stats.getMean() - stats.getSTD() * 2) {
                    mutex.lock();
                    samples.push_back(sample);
                    maxReachSize = std::max<unsigned>(samples.size(), maxReachSize);
                    mutex.unlock();
                }
            }

            /* returns the statisical mena and standart derivation */
            FloatType getMean() { return stats.getMean(); }
            FloatType getSTD() { return stats.getSTD(); }
            FloatType getSum() { return stats.getSum(); }

    };

}

#endif
