#ifndef __LONG_SHORT_TERM_MEMORY_INPUT_ELIGIBILITY_LEARNSIGNAL_ERROR_KERNEL__
#define __LONG_SHORT_TERM_MEMORY_INPUT_ELIGIBILITY_LEARNSIGNAL_ERROR_KERNEL__
#include "CudaUtils.h"
/**
 * Parallel GPU Kernel for a fully connected Network of leaky integrate and fire neurons
 * combining eligibility computation with gradient computation 
 */
namespace SNN {

    namespace Kernels {

        namespace GPU {

            __device__ void longShortTermMemoryInputEligibilityLearnSignalErrorKernel(

                /* the number of input neurons */
                unsigned numInputs,

                /* the number of hidden neurons */
                unsigned numHidden,

                /* array of old derivatives (first eligibility pass) */
                FloatType *oldDerivatives,

                /* the synaptic input weights */
                FloatType *inputWeights,

                /* the input errors */
                FloatType *inputErrors,

                /* the learn signals error for ea.hiden neuron */
                FloatType *learnSignalErrors

            ) {
                const int h = threadIdx.x;

                if (h < numHidden) {
                    FloatType learnSignalError = 0;
                    for (unsigned i = 0; i < numInputs; i++) {
                    //for (unsigned i = 0; i < 2; i++) {
                        const unsigned index = i * numHidden + h;
                        learnSignalError -= inputErrors[i] * oldDerivatives[h] * inputWeights[index];
                    }

                    learnSignalErrors[h] = learnSignalError;
                }
            }
        }
    }
}
#endif /* __LONG_SHORT_TERM_MEMORY_INPUT_ELIGIBILITY_LEARNSIGNAL_ERROR_KERNEL__ */
