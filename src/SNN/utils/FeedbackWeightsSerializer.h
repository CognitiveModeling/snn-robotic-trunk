#ifndef __FEEDBACK_WEIGHTS_SERIALIZER__
#define __FEEDBACK_WEIGHTS_SERIALIZER__
#include <vector>
#include <memory>
#include "FixedBroadcastGradient.h"
#include "BasicSerializer.h"
#include "Image.h"

namespace SNN {

    class FeedbackWeightsSerializer: public BasicSerializer {
        
        private:

            /* feedbackWeights to serialize */
            std::shared_ptr<Gradients::BasicFeedbackWeights> feedbackWeights;
            
            /* array of feedback Weights */
            FloatType *weights;


        public:

            /* constructor */
            FeedbackWeightsSerializer(std::shared_ptr<Gradients::BasicFeedbackWeights> feedbackWeights) : 
                feedbackWeights(feedbackWeights) {

                weights = new FloatType[feedbackWeights->size()];
            }

            /* destructor */
            FeedbackWeightsSerializer() {
                delete[] weights;
            }

            /* copies neuron values to arrays */
            void serialize(bool gpuMode = false) {
                (void) gpuMode;
                for (unsigned i = 0; i < feedbackWeights->size(); i++) {
                    weights[i] = feedbackWeights->getWeight(i);
                }
            }

            /* copies array values to feedbackWeights */
            void deserialize(bool gpuMode = false) {
                if (!gpuMode) {
                    for (unsigned i = 0; i < feedbackWeights->size(); i++) {
                        feedbackWeights->setWeight(weights[i], i);
                    }
                }
            }

            /* access weights */
            FloatType *getWeights() { return weights; }

            /* writes the weights of this to the given file */
            void save(int fd) {
                serialize();
                writeValue(fd, weights, sizeof(FloatType) * feedbackWeights->size());
            }

            /* loads the weights of this from the given file */
            void load(int fd) {
                readValue(fd, weights, sizeof(FloatType) * feedbackWeights->size());
                deserialize();
            }

            /* returns the sitze of this */
            size_t size() const { 
                return feedbackWeights->size();
            }

            /* sets all weigh to zero */
            void clear() {
                for (unsigned i = 0; i < this->size(); i++)
                    weights[i] = 0;
            }

            /* sets the given weight */
            void setWeight(unsigned index, FloatType weight) {
                weights[index] = weight;
            }

            /* returns wether all weights are zero */
            bool isZero() {
                for (unsigned i = 0; i < this->size(); i++)
                    if (weights[i] != 0)
                        return false;

                return true;
            }

            /* saves this as an image */
            void saveAsImage(
                std::string name, 
                unsigned numInputs, 
                unsigned numOutputs, 
                unsigned magnification = 25
            ) {
                assert(numInputs * numOutputs == feedbackWeights->size());
                Image img(numInputs, numOutputs, 3);
                
                FloatType meanPos = 0, meanNeg = 0;
                FloatType meanPosSqr = 0, meanNegSqr = 0;
                unsigned meanNegValues = 0, meanPosValues = 0;
                for (unsigned i = 0; i < feedbackWeights->size(); i++) {
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
    };

}

#endif
