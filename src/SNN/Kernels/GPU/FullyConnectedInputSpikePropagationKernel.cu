#ifndef __FULLY_CONNECTED_INPUT_SPIKE_PROPAGATION_KERNEL__
#define __FULLY_CONNECTED_INPUT_SPIKE_PROPAGATION_KERNEL__
#include "CudaUtils.h"
/**
 * Parallel CPU Kernel for a fully connected Network of neurons
 * SpikePropagation version
 */
namespace SNN {

    namespace Kernels {

        namespace GPU {
            
            __device__ void fullyConnectedInputSpikePropagationKernel(

                /* the number of input neurons */
                unsigned numInputs,

                /* the number of hidden neurons */
                unsigned numHidden,

                /* the input neuron spikes */
                FloatType *inputSpikes,

                /* the neuron input currents */
                FloatType *I,

                /* the synaptic input weights */
                FloatType *inputWeights

            ) {
                cudaAssert(numHidden <= blockDim.x);
                const int h = threadIdx.x;
                if (h >= numHidden) return;

                /* input synapses */
                for (unsigned i = 0; i < numInputs; i++) 
                    if (inputSpikes[i] != 0)
                        I[h] += inputWeights[i * numHidden + h] * inputSpikes[i];

            }
        }
    }
}
#endif /* __FULLY_CONNECTED_INPUT_SPIKE_PROPAGATION_KERNEL__ */
