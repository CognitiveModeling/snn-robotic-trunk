#ifndef __LONG_SHORT_TERM_MEMORY_LEAKY_READOUT_GRADIENT_KERNEL__
#define __LONG_SHORT_TERM_MEMORY_LEAKY_READOUT_GRADIENT_KERNEL__
#include "CudaUtils.h"
/**
 * Parallel CPU Kernel for a fully connected Network of neurons
 * LeakyReadoutGradient version
 */
namespace SNN {

    namespace Kernels {

#define ERROR_MODE_REGRESSION 0
#define ERROR_MODE_CLASSIFICATION 1
#define ERROR_MODE_INFINITY 2

        namespace GPU {
            
            __device__ void longShortTermMemoryLeakyReadoutGradientKernel(
                
                /* the number of hidden neurons */
                unsigned numHidden,

                /* the number of output neurons */
                unsigned numOutputs,

                /* the error mode of this */
                unsigned errorMode,

                /* the readout decay factor */
                FloatType readoutDecayFactor,

                /* filtered (by the readout decay) hidden spikes (from last timestep)  */
                FloatType *filteredSpikes,

                /* hidden spikes from current timestep */
                FloatType *hiddenSpikes,

                /* the network outputs */
                FloatType *outputs,

                /* the network targets */
                FloatType *targets,

                /* the network error mask for one simulation run */
                FloatType errorMask,

                /* the leaky readout gradients */
                FloatType *leakyReadoutGradients
            ) {
                cudaAssert(numHidden <= blockDim.x);
                const int i = threadIdx.x;
                if (i >= numHidden) return;

                const FloatType filteredSpike = 
                    filteredSpikes[i] * readoutDecayFactor + hiddenSpikes[i];

                if (errorMask != 0) {
                    if (errorMode == ERROR_MODE_INFINITY) {
                        for (unsigned o = 0; o < numOutputs; o++) {
                            const unsigned index = o * numHidden + i;

                            leakyReadoutGradients[index] += -1.0 * targets[o] * errorMask *
                                exp(-1.0 * targets[o] * outputs[o]) * filteredSpike;
                        }
                        
                    } else if (errorMode == ERROR_MODE_CLASSIFICATION) {
                        FloatType expSum = 0;
                        for (unsigned o = 0; o < numOutputs; o++) 
                            expSum += exp(outputs[o]);

                        for (unsigned o = 0; o < numOutputs; o++) {
                            const unsigned index = o * numHidden + i;

                            const FloatType softmax = exp(outputs[o]) / expSum;

                            leakyReadoutGradients[index] += errorMask *
                                (softmax - targets[o]) * filteredSpike;
                        }
                    } else if (errorMode == ERROR_MODE_REGRESSION)  {
                        for (unsigned o = 0; o < numOutputs; o++) {
                            const unsigned index = o * numHidden + i;

                            leakyReadoutGradients[index] += errorMask *
                                (outputs[o] - targets[o]) * filteredSpike;
                        }
                    }
                }

                filteredSpikes[i] = filteredSpike;
            }
        }
    }
}
#endif /* __LONG_SHORT_TERM_MEMORY_LEAKY_READOUT_GRADIENT_KERNEL__ */
