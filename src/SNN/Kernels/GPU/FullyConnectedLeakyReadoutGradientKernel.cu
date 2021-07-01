#ifndef __FULLY_CONNECTED_LEAKY_READOUT_GRADIENT_KERNEL__
#define __FULLY_CONNECTED_LEAKY_READOUT_GRADIENT_KERNEL__
#include "CudaUtils.h"
/**
 * Parallel CPU Kernel for a fully connected Network of neurons
 * LeakyReadoutGradient version
 */
namespace SNN {

    namespace Kernels {

        namespace GPU {
            
            __device__ void fullyConnectedLeakyReadoutGradientKernel(
                
                /* the number of input neurons */
                unsigned numInputs,

                /* the number of output neurons */
                unsigned numOutputs,

                /* filtered input spikes */
                FloatType *filteredSpikes,

                /* the network outputs */
                FloatType *outputs,

                /* the network targets */
                FloatType *targets,

                /* the leaky readout gradients */
                FloatType *leakyReadoutGradients
            ) {

                cudaAssert(numInputs == blockDim.x);
                const int i = threadIdx.x;

                for (unsigned o = 0; o < numOutputs; o++) {
                    const unsigned index = i * numOutputs + o;

                    leakyReadoutGradients[index] += (outputs[o] - targets[o]) * filteredSpikes[i];
                }
            }
        }
    }
}
#endif /* __FULLY_CONNECTED_LEAKY_READOUT_GRADIENT_KERNEL__ */
