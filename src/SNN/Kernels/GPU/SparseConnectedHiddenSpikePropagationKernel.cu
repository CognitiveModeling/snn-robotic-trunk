#ifndef __SPARSE_CONNECTED_HIDDEN_SPIKE_PROPAGATION_KERNEL__
#define __SPARSE_CONNECTED_HIDDEN_SPIKE_PROPAGATION_KERNEL__
#include "CudaUtils.h"
/**
 * Parallel CPU Kernel for a fully connected Network of neurons
 * SpikePropagation version
 */
namespace SNN {

    namespace Kernels {

        namespace GPU {
            
            __device__ void sparseConnectedHiddenSpikePropagationKernel(

                /* the number of hidden neurons */
                unsigned numHidden,

                /* the hidden neuron spikes */
                FloatType *hiddenSpikes,

                /* the neuron input currents */
                FloatType *I,

                /* the number synaptic input weights */
                unsigned numHiddenWeights,

                /* the number synaptic input weights */
                unsigned numOutputWeights,

                /* the synaptic input weights */
                FloatType *hiddenWeights,

                /* the synaptic input weights */
                FloatType *outputWeights,

                /* the synaptic input weight input indices */
                unsigned *hiddenWeightsIn,

                /* the synaptic input weight input indices */
                unsigned *outputWeightsIn,

                /* the synaptic input weight output indices */
                unsigned *hiddenWeightsOut,

                /* the synaptic input weight output indices */
                unsigned *outputWeightsOut

            ) {
                FloatType *hiddenI = I;
                FloatType *outputI = I + numHidden;

                /* hidden synapses */
                for (int i = threadIdx.x; i < numHiddenWeights; i += blockDim.x)
                    if (hiddenSpikes[hiddenWeightsIn[i]] != 0)
                        atomicAdd(hiddenI + hiddenWeightsOut[i], hiddenWeights[i]);

                /* output synapses */
                for (int i = threadIdx.x; i < numOutputWeights; i += blockDim.x)
                    if (hiddenSpikes[outputWeightsIn[i]] != 0)
                        atomicAdd(outputI + outputWeightsOut[i], outputWeights[i]);
            }
        }
    }
}
#endif /* __SPARSE_CONNECTED_HIDDEN_SPIKE_PROPAGATION_KERNEL__ */
