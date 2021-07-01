#ifndef __LONG_SHORT_TERM_MEMORY_GRADIENT_COLLECTION_KERNEL__
#define __LONG_SHORT_TERM_MEMORY_GRADIENT_COLLECTION_KERNEL__
/**
 * Parallel kernel for a Long Short Term Spiking Network
 */
namespace SNN {

    namespace Kernels {

        namespace GPU {

            __global__ void longShortTermMemoryGradientCollectionKernel(

                /* the number of input neurons */
                unsigned *numInputs_,

                /* the number of (leaky integrate and fire) hiddem neurons */
                unsigned *numStandartHidden_,

                /* the number of (adaptive leaky integrate and fire) hidden neurons */
                unsigned *numAdaptiveHidden_,

                /* the number of output neurons */
                unsigned *numOutputs_,

                /* the batch size */
                unsigned *batchSize_,

                /* the number of simulation timesteps */
                unsigned *numSimulationTimesteps_,

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
                FloatType *allInputErrorsOverTime

            ) {
                unsigned numInputs                = *numInputs_;
                unsigned numOutputs               = *numOutputs_;
                unsigned batchSize                = *batchSize_;
                unsigned numSimulationTimesteps   = *numSimulationTimesteps_;
                const unsigned id                 = threadIdx.x + blockDim.x * blockIdx.x;
                const unsigned numHidden          = *numStandartHidden_ + *numAdaptiveHidden_;
                const unsigned numInputGradients  = numInputs * numHidden;
                const unsigned numHiddenGradients = numHidden * numHidden;
                const unsigned numOutputGradients = numHidden * numOutputs;
                const unsigned numInputsErrors    = numInputs * numSimulationTimesteps;

                /* summ input gradients */
                for (unsigned offset = 0; 
                     offset + id < numInputs * numHidden; 
                     offset += blockDim.x * gridDim.x) {
                    
                    const unsigned index = offset + id;
                    for (unsigned b = 1; b < batchSize; b++) {
                        inputFiringRateGradients[index] += 
                            inputFiringRateGradients[b * numInputGradients + index];

                        inputFixedBroadcastGradients[index] += 
                            inputFixedBroadcastGradients[b * numInputGradients + index];
                    }
                }

                /* summ hidden gradients */
                for (unsigned offset = 0; 
                     offset + id < numHidden * numHidden; 
                     offset += blockDim.x * gridDim.x) {
                    
                    const unsigned index = offset + id;
                    for (unsigned b = 1; b < batchSize; b++) {
                        hiddenFiringRateGradients[index] += 
                            hiddenFiringRateGradients[b * numHiddenGradients + index];

                        hiddenFixedBroadcastGradients[index] += 
                            hiddenFixedBroadcastGradients[b * numHiddenGradients + index];
                    }
                }

                /* summ output gradients */
                for (unsigned offset = 0; 
                     offset + id < numHidden * numOutputs; 
                     offset += blockDim.x * gridDim.x) {
                    
                    const unsigned index = offset + id;
                    for (unsigned b = 1; b < batchSize; b++) {
                        leakyReadoutGradients[index] += 
                            leakyReadoutGradients[b * numOutputGradients + index];
                    }
                }
                
                /* summ input error gradients */
                for (unsigned offset = 0; 
                     offset + id < numInputsErrors;
                     offset += blockDim.x * gridDim.x) {

                    const unsigned index = offset + id;
                    inputErrorsOverTime[index] = allInputErrorsOverTime[index];
                    
                    for (unsigned b = 1; b < batchSize; b++) {
                        inputErrorsOverTime[index] += 
                            allInputErrorsOverTime[b * numInputsErrors + index];
                    }
                }
#if 0                
                __syncthreads();

                /* summ hidden gradients */
                for (unsigned offset = 0; 
                     offset + id < numHidden * numHidden; 
                     offset += blockDim.x * gridDim.x) {
                    
                    const unsigned index = offset + id;
                    hiddenFiringRateGradients[index]     /= batchSize;
                    hiddenFixedBroadcastGradients[index] /= batchSize;
                }

                /* summ output gradients */
                for (unsigned offset = 0; 
                     offset + id < numHidden * numOutputs; 
                     offset += blockDim.x * gridDim.x) {
                    
                    const unsigned index = offset + id;
                    leakyReadoutGradients[index] /= batchSize;
                }
#endif
            }
        }
    }
}
#endif /* __LONG_SHORT_TERM_MEMORY_GRADIENT_COLLECTION_KERNEL__ */
