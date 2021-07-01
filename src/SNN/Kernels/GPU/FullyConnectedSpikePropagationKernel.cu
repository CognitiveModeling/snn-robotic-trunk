#ifndef __FULLY_CONNECTED_SPIKE_PROPAGATION_KERNEL__
#define __FULLY_CONNECTED_SPIKE_PROPAGATION_KERNEL__
#include "CudaUtils.h"
/**
 * Parallel CPU Kernel for a fully connected Network of neurons
 * SpikePropagation version
 */
namespace SNN {

    namespace Kernels {

        namespace GPU {
            
            __device__ void fullyConnectedSpikePropagationKernel(

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

                /* the synaptic input weights */
                FloatType *inputWeights,

                /* the synaptic input weights */
                FloatType *hiddenWeights,

                /* the synaptic input weights */
                FloatType *outputWeights
            ) {
                cudaAssert(numHidden <= blockDim.x);
                const int h = threadIdx.x;
                if (h >= numHidden) return;

                FloatType *hiddenI = I;
                FloatType *outputI = I + numHidden;

                /* input synapses */
                for (unsigned i = 0; i < numInputs; i++) 
                    if (inputSpikes[i] != 0)
                        hiddenI[h] += inputWeights[i * numHidden + h] * inputSpikes[i];


                /* hidden synapses */
                for (unsigned i = 0; i < numHidden; i++) 
                    if (hiddenSpikes[i] != 0)
                        hiddenI[h] += hiddenWeights[i * numHidden + h];

                /* output synapses */
                for (unsigned o = 0; o < numOutputs; o++) 
                    if (hiddenSpikes[h] != 0)
                        atomicAdd(outputI + o, outputWeights[h * numOutputs + o]);
            }
        }
    }
}
#endif /* __FULLY_CONNECTED_SPIKE_PROPAGATION_KERNEL__ */
