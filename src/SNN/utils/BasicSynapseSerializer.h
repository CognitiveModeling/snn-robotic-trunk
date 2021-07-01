#ifndef __BASIC_SYNAPSE_SERIALIZER__
#define __BASIC_SYNAPSE_SERIALIZER__
#include <vector>
#include <memory>
#include "BasicSynapse.h"
#include "BasicSerializer.h"
#include "utils.h"
#include "Image.h"

namespace SNN {

    class BasicSynapseSerializer: public BasicSerializer {
        
        private:

            /* synapses to serialize */
            std::vector<std::shared_ptr<Interfaces::BasicSynapse>> synapses;
            
            /* array of weights for each synapse */
            FloatType *weights;

            /* array of eligibility traces for each synapse */
            FloatType *eligibilityTraces;

            /* wether to run checks or not */
            bool doChecks;

            /* check the given two values */
            void check(unsigned index, FloatType v1, FloatType v2, std::string name) {
                if (fabs(v1 - v2) > 0.000001) {
                    log_err("synapse " + itoa(index) + " " + name + ": " + ftoa(v1, 10) + 
                            " != " + ftoa(v2, 10), LOG_W);
                }
            }

        public:

            /* constructor */
            BasicSynapseSerializer(
                std::vector<std::shared_ptr<Interfaces::BasicSynapse>> synapses,
                bool doChecks = true
            ) : 
                synapses(synapses), doChecks(doChecks) {

                weights           = new FloatType[synapses.size()];
                eligibilityTraces = new FloatType[synapses.size()];
            }

            /* destructor */
            ~BasicSynapseSerializer() {
                synapses.clear();
                delete[] weights;
                delete[] eligibilityTraces;
            }

            /* copies neuron values to arrays */
            void serialize(bool gpuMode = false) {
                if (gpuMode) {
                    for (unsigned i = 0; i < synapses.size(); i++) {
                        weights[i]           = synapses[i]->getWeight();
                    }
                } else {
                    for (unsigned i = 0; i < synapses.size(); i++) {
                        weights[i]           = synapses[i]->getWeight();
                        eligibilityTraces[i] = synapses[i]->getEligibilityTrace();
                    }
                }
            }

            /* copies array values to synapses */
            void deserialize(bool gpuMode = false) {
                if (!gpuMode) {
                    for (unsigned i = 0; i < synapses.size(); i++) {
                        synapses[i]->setWeight(weights[i]);
                        synapses[i]->setEligibilityTrace(eligibilityTraces[i]);
                    }
                }
            }

            /* check wether serialized and original values are the same */
            void check(bool gpuMode = false) {

                /* skip check in gpu mode */
                if (gpuMode) return;

                if (doChecks) {
                    for (unsigned i = 0; i < synapses.size(); i++) {
                        check(i, weights[i]          , synapses[i]->getWeight(),           "weight");
                        check(i, eligibilityTraces[i], synapses[i]->getEligibilityTrace(), "eligibility-trace");
                    }
                }
            }

            /* access weights or eligibility traces */
            FloatType *getWeights()           { return weights;           }
            FloatType *getEligibilityTraces() { return eligibilityTraces; }
            
            /* writes the weights of this to the given file */
            void save(int fd) {
                for (unsigned i = 0; i < synapses.size(); i++) 
                     weights[i] = synapses[i]->getWeight();

                writeValue(fd, weights, sizeof(FloatType) * synapses.size());
            }

            /* loads the weights of this from the given file */
            void load(int fd) {
                readValue(fd, weights, sizeof(FloatType) * synapses.size());

                for (unsigned i = 0; i < synapses.size(); i++) 
                    synapses[i]->setWeight(weights[i]);
            }

            /* returns the sitze of this */
            size_t size() const { return synapses.size(); }

            /* sets all weigh to zero */
            void clear() {
                for (unsigned i = 0; i < synapses.size(); i++)
                    weights[i] = 0;
            }

            /* sets the given weight */
            void setWeight(unsigned index, FloatType weight) {
                weights[index] = weight;
            }

            /* returns wether all weights are zero */
            bool isZero() {
                for (unsigned i = 0; i < synapses.size(); i++)
                    if (weights[i] != 0)
                        return false;

                return true;
            }

            /* saves this as an image */
            void saveAsImage(
                std::string name, 
                unsigned numInputs, 
                unsigned numOutputs, 
                unsigned magnification = 35
            ) {
                assert(numInputs * numOutputs == synapses.size());
                Image img(numInputs, numOutputs, 3);
                
                FloatType meanPos = 0, meanNeg = 0;
                FloatType meanPosSqr = 0, meanNegSqr = 0;
                unsigned meanNegValues = 0, meanPosValues = 0;
                for (unsigned i = 0; i < synapses.size(); i++) {
                    if (weights[i] > 0) {
                        meanPos    += weights[i];
                        meanPosSqr += pow(weights[i], 2);
                        meanPosValues++;
                    }
                    if (weights[i] < 0) {
                        meanNeg    += weights[i];
                        meanNegSqr += pow(weights[i], 2);
                        meanNegValues++;
                    }
                }
                FloatType stdPos = 0, stdNeg = 0;
                if (meanPosValues > 1) {
                    meanPos    /= meanPosValues;
                    meanPosSqr /= meanPosValues;
                    stdPos = sqrt((FloatType(meanPosValues) / (meanPosValues - 1)) * (meanPosSqr - pow(meanPos, 2)));
                }
                if (meanNegValues > 1) {
                    meanNeg    /= meanNegValues;
                    meanNegSqr /= meanNegValues;
                    stdNeg = sqrt((FloatType(meanNegValues) / (meanNegValues - 1)) * (meanNegSqr - pow(meanNeg, 2)));
                }
                
                for (unsigned i = 0; i < numInputs; i++) {
                    for (unsigned o = 0; o < numOutputs; o++) {
                        unsigned index = i * numOutputs + o;
                        if (weights[index] > 0)
                            img.getPixelGreen(i, o) = std::min(unsigned(255 * weights[index] / (meanPos + 2 * stdPos)), 255u);
                        if (weights[index] < 0)
                            img.getPixelRed(i, o) = std::min(unsigned(255 * weights[index] / (meanNeg - 2 * stdNeg)), 255u);
                    }
                }

                img.upscal(magnification);
                img.save(name);
            }


            /* removes all weights with contibute statistically less */
            void prune(FloatType alpha, unsigned nInputs, unsigned nOutputs) {
                for (unsigned i = 0; i < nInputs; i++) {
                    unsigned n = 0;
                    FloatType mean = 0;
                    FloatType meanSqr = 0;
                    for (unsigned o = 0; o < nOutputs; o++) {
                        if (weights[i * nOutputs + o] != 0) {
                            n++;
                            mean    += fabs(weights[i * nOutputs + o]);
                            meanSqr += pow(weights[i * nOutputs + o], 2);
                        }
                    }
                    if (n > 1) {
                        mean /= n;
                        meanSqr /= n;
                        FloatType std = (n / FloatType(n - 1)) * (meanSqr - pow(mean, 2));

                        for (unsigned o = 0; o < nOutputs; o++) 
                            if (fabs(weights[i * nOutputs + o]) < mean - std * alpha) 
                                weights[i * nOutputs + o] = 0;
                    }
                }
                for (unsigned i = 0; i < synapses.size(); i++) 
                    synapses[i]->setWeight(weights[i]);
            }

            /* returns the number of active synapses */
            unsigned getNumActiveSynases() {
                unsigned n = 0;
                for (unsigned i = 0; i < synapses.size(); i++) 
                    if (weights[i] != 0)
                        n++;

                return n;
            }

    };

}

#endif
