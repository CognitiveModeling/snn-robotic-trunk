#ifndef __LONG_SHORT_TERM_MEMORY_SPARSE_LEARN_SIGNAL_KERNEL__
#define __LONG_SHORT_TERM_MEMORY_SPARSE_LEARN_SIGNAL_KERNEL__
#include "CudaUtils.h"
/**
 * Parallel GPU Kernel for a fully connected Network of leaky integrate and fire neurons
 * combining eligibility computation with gradient computation 
 */
namespace SNN {

    namespace Kernels {

        namespace GPU {

            __device__ void longShortTermMemorySparseLearnSignalKernel(

                /* flags defining execution modules */
                unsigned flags,

                /* the number of network output neurons */
                unsigned numOutputs,

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

                /* the number feedback weight inputs */
                unsigned numFeedbackWeights,

                /* the feedback weights */
                FloatType *feedbackWeights,

                /* the feedback weight input indices */
                unsigned *feedbackWeightsIn,

                /* the feedback weight output indices */
                unsigned *feedbackWeightsOut

            ) {

                if (errorMask != 0) {
                    if (CONTAINS(flags, LSNN_REGRESSION_ERROR)) {
                        for (int i = threadIdx.x; i < numFeedbackWeights; i += blockDim.x) {

                            const unsigned h = feedbackWeightsIn[i];
                            const unsigned o = feedbackWeightsOut[i];

                            atomicAdd(
                                learnSignals + h, 
                                targetWeights[o] * 
                                errorFactors[o] * 
                                errorMask * 
                                feedbackWeights[i] * 
                                (outputs[o] - targets[o])
                            );
                        }
                    } else if (CONTAINS(flags, LSNN_CLASSIFICATION_ERROR)) {
                    
                        FloatType expSum = 0;
                        for (unsigned o = 0; o < numOutputs; o++) 
                            expSum += exp(outputs[o]);

                        for (int i = threadIdx.x; i < numFeedbackWeights; i += blockDim.x) {
                            const unsigned h = feedbackWeightsIn[i];
                            const unsigned o = feedbackWeightsOut[i];

                            const FloatType softmax = exp(outputs[o]) / expSum;

                            atomicAdd(
                                learnSignals + h, 
                                targetWeights[o] * errorMask * feedbackWeights[i] * (softmax - targets[o])
                            );
                        }
                    }
                }
            }
        }
    }
}
#endif /* __LONG_SHORT_TERM_MEMORY_SPARSE_LEARN_SIGNAL_KERNEL__ */
