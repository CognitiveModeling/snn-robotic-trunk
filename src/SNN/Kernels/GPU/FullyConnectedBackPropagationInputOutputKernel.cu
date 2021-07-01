#ifndef __FULLY_CONNECTED_BACK_PROPAGATION_INPUT_OUTPUT_KERNEL__
#define __FULLY_CONNECTED_BACK_PROPAGATION_INPUT_OUTPUT_KERNEL__
#include "CudaUtils.h"
/**
 * Parallel CPU Kernel for a fully connected Network of InputOutputNeurons
 */
namespace SNN {

    namespace Kernels {

        namespace GPU {
            
            __device__ void fullyConnectedBackPropagationInputOutputKernel(

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

                /* the number of spikes per neuron */
                FloatType *numSpikes

            ) {
                if (wrapThreads(numInputs) + numOutputs <= blockDim.x) {

                    /* input neurons */
                    if (threadIdx.x < numInputs && spikes != NULL) {
                        const int i = threadIdx.x;
                        if (spikes[i] != 0) numSpikes[i] += spikes[i];
                    }

                    /* readout voltage */
                    if (wrapThreads(numInputs) <= threadIdx.x && 
                        threadIdx.x < wrapThreads(numInputs) + numOutputs) {

                        const int i = threadIdx.x - wrapThreads(numInputs);
                        v[i] = v[i] * readoutDecayFactor + I[i];
                        I[i] = 0;
                    }
                } else {

                    const int i = threadIdx.x;

                    /* input neurons */
                    if (i < numInputs && spikes != NULL) {
                        if (spikes[i] != 0) numSpikes[i] += spikes[i];
                    }

                    /* readout voltage */
                    if (i < numOutputs) {
                        v[i] = v[i] * readoutDecayFactor + I[i];
                        I[i] = 0;
                    }

                }
            }
        }
    }
}
#endif /* __FULLY_CONNECTED_BACK_PROPAGATION_INPUT_OUTPUT_KERNEL__ */
