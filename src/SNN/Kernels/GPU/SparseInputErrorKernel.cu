#ifndef __SPARSE_INPUT_ERROR_KERNEL__
#define __SPARSE_INPUT_ERROR_KERNEL__
#include "CudaUtils.h"
/**
 * Parallel GPU Kernel for a fully connected Network of leaky integrate and fire neurons
 * combining eligibility computation with gradient computation 
 */
namespace SNN {

    namespace Kernels {

        namespace GPU {
            
            __device__ void sparseInputErrorKernel(

                /* the number synaptic input weights */
                unsigned numInputWeights,

                /* the synaptic input weights */
                FloatType *inputWeights,

                /* the synaptic input weight input indices */
                unsigned *inputWeightsIn,

                /* the synaptic input weight output indices */
                unsigned *inputWeightsOut,

                /* the voltage component of the delta errors */
                FloatType *voltageError,

                /* the input errors */
                FloatType *inputErrors


            ) {
                const int weightsPerThread = ceilf(numInputWeights / float(blockDim.x));
                const int start            = threadIdx.x * weightsPerThread;
                const int end              = min(numInputWeights , (1 + threadIdx.x) * weightsPerThread);

                for (int i = start; i < end; i++)
                    atomicAdd(inputErrors + inputWeightsIn[i], voltageError[inputWeightsOut[i]] * inputWeights[i]);
            }
        }
    }
}
#endif /* __SPARSE_INPUT_ERROR_KERNEL__ */
