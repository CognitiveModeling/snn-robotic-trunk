#ifndef __SPARSE_CONNECTED_SPIKE_PROPAGATION_KERNEL__
#define __SPARSE_CONNECTED_SPIKE_PROPAGATION_KERNEL__
#include "CudaUtils.h"
/**
 * Parallel CPU Kernel for a fully connected Network of neurons
 * SpikePropagation version
 */
namespace SNN {

    namespace Kernels {

        namespace GPU {
            
            __device__ void sparseConnectedSpikePropagationKernel(

                /* the number of input neurons */
                unsigned numInputs,

                /* the number of hidden neurons */
                unsigned numHidden,

                /* the number of output neurons */
                unsigned numOutputs,

                /* the input neuron spikes */
                FloatType *inputSpikes,

                /* the hidden neuron spikes */
                FloatType *hiddenSpikes,

                /* the neuron input currents */
                FloatType *I,

                /* the number synaptic input weights */
                unsigned numInputWeights,

                /* the number synaptic input weights */
                unsigned numHiddenWeights,

                /* the number synaptic input weights */
                unsigned numOutputWeights,

                /* the synaptic input weights */
                FloatType *inputWeights,

                /* the synaptic input weights */
                FloatType *hiddenWeights,

                /* the synaptic input weights */
                FloatType *outputWeights,

                /* the synaptic input weight input indices */
                unsigned *inputWeightsIn,

                /* the synaptic input weight input indices */
                unsigned *hiddenWeightsIn,

                /* the synaptic input weight input indices */
                unsigned *outputWeightsIn,

                /* the synaptic input weight output indices */
                unsigned *inputWeightsOut,

                /* the synaptic input weight output indices */
                unsigned *hiddenWeightsOut,

                /* the synaptic input weight output indices */
                unsigned *outputWeightsOut

            ) {
                FloatType *hiddenI = I;
                FloatType *outputI = I + numHidden;

                /* input synapses */
                for (int i = threadIdx.x; i < numInputWeights; i += blockDim.x)
                    if (inputSpikes[inputWeightsIn[i]])
                        atomicAdd(hiddenI + inputWeightsOut[i], inputWeights[i]);

                /* hidden synapses */
                for (int i = threadIdx.x; i < numHiddenWeights; i += blockDim.x)
                    if (hiddenSpikes[hiddenWeightsIn[i]])
                        atomicAdd(hiddenI + hiddenWeightsOut[i], hiddenWeights[i]);

                /* output synapses */
                for (int i = threadIdx.x; i < numOutputWeights; i += blockDim.x)
                    if (hiddenSpikes[outputWeightsIn[i]])
                        atomicAdd(outputI + outputWeightsOut[i], outputWeights[i]);
            }
        }
    }
}
#endif /* __SPARSE_CONNECTED_SPIKE_PROPAGATION_KERNEL__ */
