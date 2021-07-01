#ifndef __INPUT_OUTPUT_KERNEL__
#define __INPUT_OUTPUT_KERNEL__
#include "CudaUtils.h"
/**
 * Parallel CPU Kernel for a fully connected Network of InputOutputNeurons
 */
namespace SNN {

    namespace Kernels {

        namespace GPU {
            
            __device__ void inputOutputKernel(

                /* the number input of neurons */
                unsigned numInputs,

                /* the number output of neurons */
                unsigned numOutputs,

                /* hidden neuron voltage decay factor */
                FloatType hiddenDecayFactor,
                
                /* readout neuron voltage decay factor */
                FloatType readoutDecayFactor,

                /* the readout neuron input currents */
                FloatType *I,

                /* the readout neuron voltage */
                FloatType *v,

                /* the input neuron spikes */
                FloatType *spikes,

                /* the filtered input spikes */
                FloatType *filteredSpikes

            ) {
                if (spikes != NULL) {
                    for (int i = threadIdx.x; i < numInputs; i += blockDim.x) {
                        filteredSpikes[i] = filteredSpikes[i] * hiddenDecayFactor + spikes[i];
                    }
                }

                for (int i = threadIdx.x; i < numOutputs; i += blockDim.x) {
                    v[i] = v[i] * readoutDecayFactor + I[i];
                    I[i] = 0;
                }
            }
        }
    }
}
#endif /* __INPUT_OUTPUT_KERNEL__ */
