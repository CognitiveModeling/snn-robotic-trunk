#ifndef __SPARSE_OUTPUT_ERROR_KERNEL__
#define __SPARSE_OUTPUT_ERROR_KERNEL__
#include "CudaUtils.h"
/**
 * Parallel GPU Kernel for a fully connected Network of leaky integrate and fire neurons
 * combining eligibility computation with gradient computation 
 */
namespace SNN {

    namespace Kernels {

        namespace GPU {
            
            __device__ void sparseOutputErrorKernel(

                /* flags defining execution modules */
                unsigned flags,

                /* the number of output neurons */
                unsigned numOutputs,

                /* the readout decay factor */
                FloatType readoutDecayFactor,

                /* the error mask */
                FloatType errorMask,

                /* the network error factor for one simulation run */
                FloatType *errorFactors,

                /* the network target weights */
                FloatType *targetWeights,

                /* the filtered (back propagated) output errors */
                FloatType *filteredOutputErrors,

                /* external output errors */
                FloatType *externalErrors,

                /* the outputs */
                FloatType *outputs,

                /* the targets */
                FloatType *targets

            ) {

                /* output error calculation */
                for (int i = threadIdx.x; i < numOutputs; i += blockDim.x) {

                    if (CONTAINS(flags, LSNN_REGRESSION_ERROR)) {

                        filteredOutputErrors[i] = 
                            filteredOutputErrors[i] * 
                            readoutDecayFactor +
                            (outputs[i] - targets[i]) * 
                            errorMask * 
                            errorFactors[i] * 
                            targetWeights[i];

                    } else if (CONTAINS(flags, LSNN_CLASSIFICATION_ERROR)) {

                        FloatType expSum = 1e-9;
                        for (unsigned o = 0; o < numOutputs; o++) 
                            expSum += exp(outputs[o]);

                        const FloatType softmax = exp(outputs[i]) / expSum;
                        filteredOutputErrors[i] = 
                            filteredOutputErrors[i] * 
                            readoutDecayFactor + 
                            targetWeights[i] * 
                            errorMask *
                            errorFactors[i] * 
                            (softmax - targets[i]);

                    } else if (CONTAINS(flags, LSNN_EXTERNAL_ERROR)) {
                        filteredOutputErrors[i] = 
                            filteredOutputErrors[i] * 
                            readoutDecayFactor + 
                            targetWeights[i] * 
                            errorMask *
                            errorFactors[i] * 
                            externalErrors[i];
                    }
                }
            }
        }
    }
}
#endif /* __SPARSE_OUTPUT_ERROR_KERNEL__ */
