#ifndef __SPARSE_BACK_PROPAGATED_GRADIENTS_KERNEL__
#define __SPARSE_BACK_PROPAGATED_GRADIENTS_KERNEL__
#include "CudaUtils.h"
/**
 * Parallel GPU Kernel for a fully connected Network of leaky integrate and fire neurons
 * combining eligibility computation with gradient computation 
 */
namespace SNN {

    namespace Kernels {

        namespace GPU {
            
            __device__ void sparseBackPropagatedGradients(

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

                /* the synaptic input weight output indices */
                unsigned *inputWeightsOut,

                /* the synaptic input weight input indices */
                unsigned *hiddenWeightsIn,

                /* the synaptic input weight input indices */
                unsigned *outputWeightsIn,

                /* the synaptic input weight output indices */
                unsigned *hiddenWeightsOut,

                /* the synaptic input weight output indices */
                unsigned *outputWeightsOut,

                /* the voltage component of the delta errors */
                FloatType *voltageError,
                
                /* the filtered (back propagated) output errors */
                FloatType *filteredOutputErrors,

                /* the input gradients */
                FloatType *inputGradients,

                /* the hidden gradients */
                FloatType *hiddenGradients,

                /* the output gradients */
                FloatType *outputGradients,

                /* the input neuron spikes */
                FloatType *inputSpikes,

                /* the hidden neuron spikes */
                FloatType *hiddenSpikes,

                /* the hidden neuron spikes from the previous timestep */
                FloatType *hiddenSpikesPrev

            ) {
                /* add errors to input gradients */
                for (unsigned i = threadIdx.x; i < numInputWeights; i += blockDim.x) {
                    inputGradients[i] += 
                        voltageError[inputWeightsOut[i]] * inputSpikes[inputWeightsIn[i]];
                }

                /* add errors to hidden gradients */
                if (hiddenSpikesPrev != NULL) {
                    for (unsigned i = threadIdx.x; i < numHiddenWeights; i += blockDim.x) {

                        if (hiddenSpikesPrev[hiddenWeightsIn[i]] != 0)
                            hiddenGradients[i] += voltageError[hiddenWeightsOut[i]];
                    }
                }

                /* add errors to output gradients */
                for (unsigned i = threadIdx.x; i < numOutputWeights; i += blockDim.x) {

                    if (hiddenSpikes[outputWeightsIn[i]] != 0)
                        outputGradients[i] += filteredOutputErrors[outputWeightsOut[i]];
                }
            }
        }
    }
}
#endif /* __SPARSE_BACK_PROPAGATED_GRADIENTS_KERNEL__ */
