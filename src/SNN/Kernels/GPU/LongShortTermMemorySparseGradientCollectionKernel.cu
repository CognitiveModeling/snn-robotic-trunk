#ifndef __LONG_SHORT_TERM_MEMORY_SPARSE_GRADIENT_COLLECTION_KERNEL__
#define __LONG_SHORT_TERM_MEMORY_SPARSE_GRADIENT_COLLECTION_KERNEL__
/**
 * Parallel kernel for a Long Short Term Spiking Network
 */
namespace SNN {

    namespace Kernels {

        namespace GPU {

            __global__ void longShortTermMemorySparseGradientCollectionKernel(

                /* flags defining execution modules */
                unsigned *flags_,

                /* the number input gradients */
                unsigned *numInputGradients_,

                /* the number hidden gradients */
                unsigned *numHiddenGradients_,

                /* the number output gradients */
                unsigned *numOutputGradients_,

                /* the number of inputs */
                unsigned *numInputs_,

                /* the number of outputs */
                unsigned *numOutputs_,

                /* the number of simulation timesteps */
                unsigned *numSimulationTimesteps_,

                /* the batch size */
                unsigned *batchSize_,

                /* the fixed braodcast gradients for input synapses */
                FloatType *inputFixedBroadcastGradients,

                /* the firing rate gradients for input synapses  */
                FloatType *inputFiringRateGradients,

                /* the fixed braodcast gradients for hidden synapses */
                FloatType *hiddenFixedBroadcastGradients,

                /* the firing rate gradients for hidden synapses */
                FloatType *hiddenFiringRateGradients,

                /* the leaky readout gradients */
                FloatType *leakyReadoutGradients,

                /* the input error gradients over time */
                FloatType *inputErrorsOverTime,

                /* the input error gradients over time (for all batches) */
                FloatType *allInputErrorsOverTime,

                /* the networks squared summed error */
                FloatType *networkError,

                /* the network summed target for each output */
                FloatType *summedTargets,

                /* the network squared summed target for each output */
                FloatType *squaredSummedTargets,

                /* the number of values summed for each output */
                FloatType *numSummedValues,

                /* the networks classification accuracy error */
                FloatType *classificationAccuracy,

                /* the networks number of classification samples */
                FloatType *classificationSamples

            ) {
                const unsigned flags                  = *flags_;
                const unsigned numInputGradients      = *numInputGradients_;
                const unsigned numHiddenGradients     = *numHiddenGradients_;
                const unsigned numOutputGradients     = *numOutputGradients_;
                const unsigned numInputs              = *numInputs_;
                const unsigned numOutputs             = *numOutputs_;
                const unsigned batchSize              = *batchSize_;
                const unsigned numSimulationTimesteps = *numSimulationTimesteps_;

                if (CONTAINS(flags, LSNN_ELIGIBILTY_GRADIENTS|LSNN_BACKPROPAGATED_GRADIENTS|LSNN_BACKPROPAGATED_ELIGIBILITY_GRADIENTS)) {

                    /* summ input gradients */
                    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; 
                         i < numInputGradients; 
                         i += gridDim.x * blockDim.x) {
                        
                        for (unsigned b = 1; b < batchSize; b++) {
                            inputFiringRateGradients[i] += 
                                inputFiringRateGradients[b * numInputGradients + i];

                            inputFixedBroadcastGradients[i] += 
                                inputFixedBroadcastGradients[b * numInputGradients + i];
                        }
                    }

                    /* summ hidden gradients */
                    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; 
                         i < numHiddenGradients; 
                         i += gridDim.x * blockDim.x) {
                        
                        for (unsigned b = 1; b < batchSize; b++) {
                            hiddenFiringRateGradients[i] += 
                                hiddenFiringRateGradients[b * numHiddenGradients + i];

                            hiddenFixedBroadcastGradients[i] += 
                                hiddenFixedBroadcastGradients[b * numHiddenGradients + i];
                        }
                    }
                }

                if (CONTAINS(flags, LSNN_BACKPROPAGATED_GRADIENTS|LSNN_READOUT_FORWARD_GRAIENTS)) {

                    /* summ output gradients */
                    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; 
                         i < numOutputGradients; 
                         i += gridDim.x * blockDim.x) {
                        
                        for (unsigned b = 1; b < batchSize; b++) {
                            leakyReadoutGradients[i] += 
                                leakyReadoutGradients[b * numOutputGradients + i];
                        }
                    }
                }

                if (CONTAINS(flags, LSNN_INPUT_ERRORS)) {

                    /* copy input errors */
                    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; 
                         i < numInputs * numSimulationTimesteps * batchSize; 
                         i += gridDim.x * blockDim.x) {

                        allInputErrorsOverTime[i] = inputErrorsOverTime[i];
                    }

                    /* summ input error gradients */
                    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; 
                         i < numInputs * numSimulationTimesteps; 
                         i += gridDim.x * blockDim.x) {
                        
                        for (unsigned b = 1; b < batchSize; b++) {
                            inputErrorsOverTime[i] += 
                                inputErrorsOverTime[b * numInputs * numSimulationTimesteps + i];
                        }
                    }
                }

                /* summ output errors */
                for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; 
                     i < numOutputs; 
                     i += gridDim.x * blockDim.x) {

                    networkError[batchSize * numOutputs + i]           = networkError[i];
                    summedTargets[batchSize * numOutputs + i]          = summedTargets[i];
                    squaredSummedTargets[batchSize * numOutputs + i]   = squaredSummedTargets[i];
                    numSummedValues[batchSize * numOutputs + i]        = numSummedValues[i];
                    classificationAccuracy[batchSize * numOutputs + i] = classificationAccuracy[i];
                    classificationSamples[batchSize * numOutputs + i]  = classificationSamples[i];
                }
                __syncthreads();

                for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; 
                     i < numOutputs; 
                     i += gridDim.x * blockDim.x) {
                    for (unsigned b = 1; b < batchSize; b++) {
                        atomicAdd(networkError + batchSize * numOutputs + i,           networkError[b * numOutputs + i]);
                        atomicAdd(summedTargets + batchSize * numOutputs + i,          summedTargets[b * numOutputs + i]);
                        atomicAdd(squaredSummedTargets + batchSize * numOutputs + i,   squaredSummedTargets[b * numOutputs + i]);
                        atomicAdd(numSummedValues + batchSize * numOutputs + i,        numSummedValues[b * numOutputs + i]);
                        atomicAdd(classificationAccuracy + batchSize * numOutputs + i, classificationAccuracy[b * numOutputs + i]);
                        atomicAdd(classificationSamples + batchSize * numOutputs + i,  classificationSamples[b * numOutputs + i]);
                    }
                }
            }
        }
    }
}
#endif /* __LONG_SHORT_TERM_MEMORY_SPARSE_GRADIENT_COLLECTION_KERNEL__ */
