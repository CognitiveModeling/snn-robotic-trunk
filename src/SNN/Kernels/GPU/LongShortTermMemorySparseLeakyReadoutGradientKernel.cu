#ifndef __LONG_SHORT_TERM_MEMORY_SPARSE_LEAKY_READOUT_GRADIENT_KERNEL__
#define __LONG_SHORT_TERM_MEMORY_SPARSE_LEAKY_READOUT_GRADIENT_KERNEL__
#include "CudaUtils.h"
/**
 * Parallel CPU Kernel for a fully connected Network of neurons
 * LeakyReadoutGradient version
 */
namespace SNN {

    namespace Kernels {

        namespace GPU {
            
            __device__ void longShortTermMemorySparseLeakyReadoutGradientKernel(

                /* flags defining execution modules */
                unsigned flags,

                /* the number of output neurons */
                unsigned numOutputs,
                
                /* the number output synapses */
                unsigned numOutputSynapses,

                /* the synaptic  input indices */
                unsigned *outputSynapsesIn,

                /* the synaptic output indices */
                unsigned *outputSynapsesOut,

                /* filtered input spikes */
                FloatType *filteredSpikes,

                /* hidden spikes from current timestep */
                FloatType *hiddenSpikes,

                /* the readout decay factor */
                FloatType readoutDecayFactor,

                /* the network outputs */
                FloatType *outputs,

                /* the network targets */
                FloatType *targets,

                /* the network error mask for one simulation run */
                FloatType errorMask,

                /* the leaky readout gradients */
                FloatType *leakyReadoutGradients
            ) {
                if (errorMask == 0) return;

                if (CONTAINS(flags, LSNN_REGRESSION_ERROR)) {
                    for (int index = threadIdx.x; index < numOutputSynapses; index += blockDim.x) {
                        const unsigned i = outputSynapsesIn[index];
                        const unsigned o = outputSynapsesOut[index];
                        
                        const FloatType filteredSpike = 
                            filteredSpikes[i] * readoutDecayFactor + hiddenSpikes[i];

                        leakyReadoutGradients[index] += 
                            errorMask * (outputs[o] - targets[o]) * filteredSpike;
                    }
                } else if (CONTAINS(flags, LSNN_CLASSIFICATION_ERROR)) {

                    FloatType expSum = 1e-9;
                    for (unsigned o = 0; o < numOutputs; o++) 
                        expSum += exp(outputs[o]);

                    for (int index = threadIdx.x; index < numOutputSynapses; index += blockDim.x) {
                        const unsigned i = outputSynapsesIn[index];
                        const unsigned o = outputSynapsesOut[index];

                        const FloatType softmax = exp(outputs[o]) / expSum;
                        
                        const FloatType filteredSpike = 
                            filteredSpikes[i] * readoutDecayFactor + hiddenSpikes[i];

                        leakyReadoutGradients[index] += 
                            errorMask * (softmax - targets[o]) * filteredSpike;
                    }
                }
            }
        }
    }
}
#endif /* __LONG_SHORT_TERM_MEMORY_SPARSE_LEAKY_READOUT_GRADIENT_KERNEL__ */
