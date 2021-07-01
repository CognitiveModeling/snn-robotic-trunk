#ifndef __SPARSE_CONNECTED_INPUT_SPIKE_PROPAGATION_KERNEL__
#define __SPARSE_CONNECTED_INPUT_SPIKE_PROPAGATION_KERNEL__
#include "CudaUtils.h"
/**
 * Parallel CPU Kernel for a fully connected Network of neurons
 * SpikePropagation version
 */
namespace SNN {

    namespace Kernels {

        namespace GPU {
            
            __device__ void sparseConnectedInputSpikePropagationKernel(

                /* the input neuron spikes */
                FloatType *inputSpikes,

                /* the neuron input currents */
                FloatType *I,

                /* the number synaptic input weights */
                unsigned numInputWeights,

                /* the synaptic input weights */
                FloatType *inputWeights,

                /* the synaptic input weight input indices */
                unsigned *inputWeightsIn,

                /* the synaptic input weight output indices */
                unsigned *inputWeightsOut

            ) {
                /* input synapses */
                for (int i = threadIdx.x; i < numInputWeights; i += blockDim.x)
                    if (inputSpikes[inputWeightsIn[i]] != 0)
                        atomicAdd(I + inputWeightsOut[i], inputWeights[i] * inputSpikes[inputWeightsIn[i]]);
            }
        }
    }
}
#endif /* __SPARSE_CONNECTED_INPUT_SPIKE_PROPAGATION_KERNEL__ */
