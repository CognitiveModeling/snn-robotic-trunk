#ifndef __BASIC_NEURON_SERIALIZER__
#define __BASIC_NEURON_SERIALIZER__
#include <vector>
#include <memory>
#include "BasicNeuron.h"
#include "BasicSerializer.h"
#include "utils.h"

namespace SNN {

    class BasicNeuronSerializer: public BasicSerializer {
        
        private:

            /* neurons to serialize */
            std::vector<std::shared_ptr<Interfaces::BasicNeuron>> neurons;
            
            /* array of input currents for each neuron */
            FloatType *I;

            /* array of voltage for each neuron */
            FloatType *v;

            /* array of spike status for each neuron */
            FloatType *spikes;

            /* array of num spikes status for each neuron */
            FloatType *numSpikes;

            /* array of derivatives for each neuron */
            FloatType *derivatives;

            /* array of firing rates */
            FloatType *firingRates;

            /* array of learning signals (back propagation) */
            FloatType *learSignals;

            /* array of erros (back propagation) */
            FloatType *errors;

            /* check the given two values */
            void check(unsigned index, FloatType v1, FloatType v2, std::string name) {
                if (fabs(v1 - v2) > 0.000001) {
                    log_err("neuron " + itoa(index) + " " + name + ": " + ftoa(v1, 10) + 
                            " != " + ftoa(v2, 10), LOG_W);
                }
            }

        public:

            /* constructor */
            BasicNeuronSerializer(std::vector<std::shared_ptr<Interfaces::BasicNeuron>> neurons) : 
                neurons(neurons) {

                I           = new FloatType[neurons.size()];
                v           = new FloatType[neurons.size()];
                spikes      = new FloatType[neurons.size()];
                numSpikes   = new FloatType[neurons.size()];
                derivatives = new FloatType[neurons.size()];
                firingRates = new FloatType[neurons.size()];
                learSignals = new FloatType[neurons.size()];
                errors      = new FloatType[neurons.size()];
            }

            /* destructor */
            ~BasicNeuronSerializer() {
                neurons.clear();
                delete[] I;
                delete[] v;
                delete[] spikes;
                delete[] numSpikes;
                delete[] derivatives;
                delete[] firingRates;
                delete[] learSignals;
                delete[] errors;
            }

            /* copies neuron values to arrays */
            void serialize(bool gpuMode = false) {
                if (!gpuMode) {
                    for (unsigned i = 0; i < neurons.size(); i++) {
                        I[i]           = neurons[i]->getInputCurrent();
                        v[i]           = neurons[i]->getVoltage();
                        spikes[i]      = neurons[i]->fired();
                        numSpikes[i]   = neurons[i]->getNumSpikes();
                        derivatives[i] = neurons[i]->getDerivative();
                        firingRates[i] = neurons[i]->getFiringRate();
                        learSignals[i] = neurons[i]->getLearnSignal();
                        errors[i]      = neurons[i]->getError();
                    }
                }
            }

            /* copies array values to neurons */
            void deserialize(bool gpuMode = false) {
                if (gpuMode) {
                    for (unsigned i = 0; i < neurons.size(); i++) 
                        neurons[i]->setFiringRate(firingRates[i]);
                } else {
                    for (unsigned i = 0; i < neurons.size(); i++) {
                        neurons[i]->setInputCurrent(I[i]);
                        neurons[i]->setVoltage(v[i]);
                        neurons[i]->setNumSpikes(numSpikes[i]);
                        neurons[i]->setDerivative(derivatives[i]);
                        neurons[i]->calculateFiringRate();
                        neurons[i]->setLearnSignal(learSignals[i]);
                        neurons[i]->setError(errors[i]);
                    }
                }
            }

            /* check wether serialized and original values are the same */
            void check(bool gpuMode = false) {

                /* skip check in gpu mode */
                if (gpuMode) return;

                for (unsigned i = 0; i < neurons.size(); i++) {
                    check(i, I[i],           neurons[i]->getInputCurrent(), "current");
                    check(i, v[i],           neurons[i]->getVoltage(),      "voltage");
                    check(i, spikes[i],      neurons[i]->fired(),           "spike");
                    check(i, numSpikes[i],   neurons[i]->getNumSpikes(),    "num-spike");
                    check(i, derivatives[i], neurons[i]->getDerivative(),   "derivative");
                    check(
                        i, 
                        neurons[i]->calculateFiringRate(numSpikes[i]), 
                        neurons[i]->getFiringRate(),   
                        "firing-rate"
                    );
                    check(i, learSignals[i],   neurons[i]->getLearnSignal(), "learn-signal");
                    check(i, errors[i],        neurons[i]->getError(),       "error");
                }
            }

            /* returns the index of the given neuron */
            int indexOf(std::shared_ptr<Interfaces::BasicNeuron> neuron) {
                for (unsigned i = 0; i < neurons.size(); i++)
                    if (*neuron == *(neurons[i]))
                        return i;

                return -1;
            }

            /* access input current, voltage or spikes */
            FloatType    *getI()            { return I;           }
            FloatType    *getV()            { return v;           }
            FloatType    *getSpikes()       { return spikes;      }
            FloatType    *getNumSpikes()    { return numSpikes;   }
            FloatType    *getDerivatives()  { return derivatives; }
            FloatType    *getFiringRates()  { return firingRates; }
            FloatType    *getLearnSignals() { return learSignals; }
            FloatType    *getErrors()       { return errors;      }
            

    };

}

#endif
