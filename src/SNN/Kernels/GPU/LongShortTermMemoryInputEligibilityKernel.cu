#ifndef __LONG_SHORT_TERM_MEMORY_INPUT_ELIGIBILITY_KERNEL__
#define __LONG_SHORT_TERM_MEMORY_INPUT_ELIGIBILITY_KERNEL__
#include "CudaUtils.h"
/**
 * Parallel GPU Kernel for a fully connected Network of leaky integrate and fire neurons
 * combining eligibility computation with gradient computation 
 */
namespace SNN {

    namespace Kernels {

        namespace GPU {

            __device__ void longShortTermMemoryInputEligibilityKernel(

                /* the number of input neurons */
                unsigned numInputs,

                /* the number of hidden neurons */
                unsigned numHidden,

                /* array of derivatives */
                FloatType *derivatives,

                /* the synaptic input weights */
                FloatType *inputWeights,

                /* the input errors */
                FloatType *inputErrors,

                /* the learn signals for ea.hiden neuron over time in 
                 * order to compute the input errors */ 
                FloatType *learnSignals

            ) {
                const int i = threadIdx.x;
                if (i < numInputs) {
                    FloatType inputError = 0;
                    for (unsigned h = 0; h < numHidden; h++) {
                        const unsigned index = i * numHidden + h;
                        inputError += learnSignals[h] * derivatives[h] * inputWeights[index];
                    }

                    inputErrors[i] = inputError;
                }
            }
        }
    }
}
#endif /* __LONG_SHORT_TERM_MEMORY_INPUT_ELIGIBILITY_KERNEL__ */
