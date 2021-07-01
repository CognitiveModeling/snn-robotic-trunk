#ifndef __BASIC_GRADIENT_SERIALIZER__
#define __BASIC_GRADIENT_SERIALIZER__
#include <vector>
#include <memory>
#include <string>
#include "BasicGradient.h"
#include "BasicSerializer.h"
#include "utils.h"

namespace SNN {

    class BasicGradientSerializer: public BasicSerializer {
        
        private:

            /* gradients to serialize */
            std::vector<std::shared_ptr<Interfaces::BasicGradient>> gradients;
            
            /* array of gradients */
            FloatType *gradientsArray;

            /* the name of this gradient */
            std::string name;

            /* check the given two values */
            void check(unsigned index, FloatType v1, FloatType v2) {
#if 1
                while (fabs(v1) > 1 || fabs(v2) > 1) {
                    v1 /= 2;
                    v2 /= 2;
                }
#endif
#if FloatTypeSize == 32
                if (fabs(v1 - v2) > 1e-3) {
                    log_err("gradient " + name + " " + itoa(index) + ", device: " + ftoa(v1, 10) + 
                            " != host: " + ftoa(v2, 10), LOG_W);
                }
#else
                if (fabs(v1 - v2) > 1e-3) {
                    log_err("gradient " + name + " " + itoa(index) + ", device: " + ftoa(v1, 10) + 
                            " != host: " + ftoa(v2, 10), LOG_W);
                }
#endif
            }

        public:

            /* constructor */
            BasicGradientSerializer(
                std::vector<std::shared_ptr<Interfaces::BasicGradient>> gradients,
                std::string name = ""
            ) : 
                gradients(gradients), name(name) {

                gradientsArray = new FloatType[gradients.size()];
            }

            /* destructor */
            ~BasicGradientSerializer() {
                gradients.clear();
                delete[] gradientsArray;
            }

            /* copies neuron values to arrays */
            void serialize(bool gpuMode = false) {
                if (!gpuMode) {
                    for (unsigned i = 0; i < gradients.size(); i++) {
                        gradientsArray[i] = gradients[i]->getAccumulatedGradient();
                    }
                }
            }

            /* copies array values to gradients */
            void deserialize(bool gpuMode = false) {
                (void) gpuMode;
                for (unsigned i = 0; i < gradients.size(); i++) {
                    gradients[i]->setAccumulatedGradient(gradientsArray[i]);
                }
            }

            /* check wether serialized and original values are the same */
            void check(bool gpuMode = false) {

                /* gradients are always checked */
                (void) gpuMode;

                for (unsigned i = 0; i < gradients.size(); i++) {
                    check(i, gradientsArray[i], gradients[i]->getAccumulatedGradient());
                }
            }

            /* access weights or eligibility traces */
            FloatType *getGradients() { return gradientsArray; }
            

    };

}

#endif
