#ifndef __FULLY_CONNECTED_HIDDEN_SPIKE_PROPAGATION_KERNEL__
#define __FULLY_CONNECTED_HIDDEN_SPIKE_PROPAGATION_KERNEL__
#include "CudaUtils.h"
/**
 * Parallel CPU Kernel for a fully connected Network of neurons
 * SpikePropagation version
 */
namespace SNN {

    namespace Kernels {

        namespace GPU {
            
            __device__ void fullyConnectedHiddenSpikePropagationKernel(

                /* the number of hidden neurons */
                unsigned numHidden,

                /* the number of output neurons */
                unsigned numOutputs,

                /* the hidden neuron spikes */
                FloatType *hiddenSpikes,

                /* the neuron input currents */
                FloatType *I,

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
#endif /* __FULLY_CONNECTED_HIDDEN_SPIKE_PROPAGATION_KERNEL__ */
