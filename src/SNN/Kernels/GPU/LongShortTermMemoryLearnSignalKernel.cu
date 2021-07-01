#ifndef __LONG_SHORT_TERM_MEMORY_LEARN_SIGNAL_KERNEL__
#define __LONG_SHORT_TERM_MEMORY_LEARN_SIGNAL_KERNEL__
#include "CudaUtils.h"
/**
 * Parallel GPU Kernel for a fully connected Network of leaky integrate and fire neurons
 * combining eligibility computation with gradient computation 
 */
namespace SNN {

    namespace Kernels {

#define ERROR_MODE_REGRESSION 0
#define ERROR_MODE_CLASSIFICATION 1
#define ERROR_MODE_INFINITY 2

        namespace GPU {

            __device__ void longShortTermMemoryLearnSignalKernel(

                /* the number of hidden neurons */
                unsigned numHidden,

                /* the number of network output neurons */
                unsigned numOutputs,

                /* the error mode of this */
                unsigned errorMode,

                /* the network outputs */
                FloatType *outputs,

                /* the network target weights */
                FloatType *targetWeights,

                /* the network targets */
                FloatType *targets,

                /* the network error mask for one simulation run */
                FloatType errorMask,

                /* the network error factor for one simulation run */
                FloatType *errorFactors,

                /* the learnSignals for each output neuron */
                FloatType *learnSignals,

                /* the feedback weights */
                FloatType *feedbackWeights

            ) {
                const unsigned h = threadIdx.x;
                if (h < numHidden) {

                    /* compute learn signals */
                    FloatType learnSignal = 0;

                    if (errorMask != 0) {
                        if (errorMode == ERROR_MODE_INFINITY) {
                            for (unsigned o = 0; o < numOutputs; o++) {
                                learnSignal += targetWeights[o] * errorMask * errorFactors[o] * 
                                               feedbackWeights[h * numOutputs + o] * 
                                               -1.0 * targets[o] * 
                                               exp(-1.0 * targets[o] * outputs[o]);
                            }
                            
                        } else if (errorMode == ERROR_MODE_CLASSIFICATION) {

                            FloatType expSum = 0;
                            for (unsigned o = 0; o < numOutputs; o++) 
                                expSum += exp(outputs[o]);

                            for (unsigned o = 0; o < numOutputs; o++) {
                                const FloatType softmax = exp(outputs[o]) / expSum;

                                learnSignal += targetWeights[o] * errorMask * errorFactors[o] *
                                               feedbackWeights[h * numOutputs + o] * 
                                               (softmax - targets[o]);
                            }
                        } else if (errorMode == ERROR_MODE_REGRESSION)  {
                            for (unsigned o = 0; o < numOutputs; o++) {
                                learnSignal += targetWeights[o] * errorMask *  errorFactors[o] *
                                               feedbackWeights[h * numOutputs + o] * 
                                               (outputs[o] - targets[o]);
                            }
                        }
                    }
                        
                    learnSignals[h] = learnSignal;
                }
            }
        }
    }
}
#endif /* __LONG_SHORT_TERM_MEMORY_LEARN_SIGNAL_KERNEL__ */
