#ifndef __LONG_SHORT_TERM_MEMORY_WEIGHT_UPDATE_KERNEL__
#define __LONG_SHORT_TERM_MEMORY_WEIGHT_UPDATE_KERNEL__
/**
 * Parallel kernel for a Long Short Term Spiking Network
 */
namespace SNN {

    namespace Kernels {

        namespace GPU {

            __global__ void longShortTermMemoryWeightUpdateKernel(

                /* the learn rate of this */
                FloatType *learnRate_,

                /* the regularizer learn rate of this */
                FloatType *regularizerFactor_,

                /* the beta 1 and 2 parameters */
                FloatType *beta1_, 
                FloatType *beta2_,

                /* epsilon parameter */
                FloatType *epsilon_,
                
                /* runing product of beta parameters */
                FloatType *beta1Product_, 
                FloatType *beta2Product_,

                /* the number input gradients */
                unsigned *numInputGradients_,

                /* the number hidden gradients */
                unsigned *numHiddenGradients_,

                /* the number output gradients */
                unsigned *numOutputGradients_,

                /* the synaptic input weights */
                FloatType *inputWeights,

                /* the synaptic input weights */
                FloatType *hiddenWeights,

                /* the synaptic input weights */
                FloatType *outputWeights,

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
                
                /* Adam v and m values for the fixed braodcast gradients for input synapses */
                FloatType *inputM,
                FloatType *inputV,

                /* Adam v and m values the fixed braodcast gradients for hidden synapses */
                FloatType *hiddenM,
                FloatType *hiddenV,

                /* Adam v and m values the leaky readout gradients */
                FloatType *outputM,
                FloatType *outputV

            ) {
                const FloatType learnRate         = *learnRate_;
                const FloatType regularizerFactor = *regularizerFactor_;
                const FloatType beta1             = *beta1_;
                const FloatType beta2             = *beta2_;
                const FloatType epsilon           = *epsilon_;
                const FloatType beta1Product      = *beta1Product_;
                const FloatType beta2Product      = *beta2Product_;
                const unsigned numInputGradients  = *numInputGradients_;
                const unsigned numHiddenGradients = *numHiddenGradients_;
                const unsigned numOutputGradients = *numOutputGradients_;

                /* update input weigts */
                for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; 
                     i < numInputGradients; 
                     i += gridDim.x * blockDim.x) {

                    inputM[i] = beta1 * inputM + (1 - beta1) * inputFixedBroadcastGradients[i];
                    inputV[i] = beta2 * inputV + (1 - beta2) * pow(inputFixedBroadcastGradients[i], 2);

                    FloatType unbiased_inputM = inputM[i] / (1 - beta1Product);
                    FloatType unbiased_inputV = inputV[i] / (1 - beta2Product);

                    inputWeights[i] -= 
                        learnRate * unbiased_inputM / (sqrt(unbiased_inputV) + epsilon) -
                        regularizerFactor * inputFiringRateGradients[i];
                }

                /* update hidden weigts */
                for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; 
                     i < numHiddenGradients; 
                     i += gridDim.x * blockDim.x) {
                    
                    hiddenM[i] = beta1 * hiddenM + (1 - beta1) * hiddenFixedBroadcastGradients[i];
                    hiddenV[i] = beta2 * hiddenV + (1 - beta2) * pow(hiddenFixedBroadcastGradients[i], 2);

                    FloatType unbiased_hiddenM = hiddenM[i] / (1 - beta1Product);
                    FloatType unbiased_hiddenV = hiddenV[i] / (1 - beta2Product);

                    hiddenWeights[i] -= 
                        learnRate * unbiased_hiddenM / (sqrt(unbiased_hiddenV) + epsilon) -
                        regularizerFactor * hiddenFiringRateGradients[i];
                }

                /* update output weigts */
                for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; 
                     i < numOutputGradients; 
                     i += gridDim.x * blockDim.x) {
                    
                    outputM[i] = beta1 * outputM + (1 - beta1) * leakyReadoutGradients[i];
                    outputV[i] = beta2 * outputV + (1 - beta2) * pow(leakyReadoutGradients[i], 2);

                    FloatType unbiased_outputM = outputM[i] / (1 - beta1Product);
                    FloatType unbiased_outputV = outputV[i] / (1 - beta2Product);

                    outputWeights[i] -= 
                        learnRate * unbiased_outputM / (sqrt(unbiased_outputV) + epsilon);
                }
            }
        }
    }
}
#endif /* __LONG_SHORT_TERM_MEMORY_WEIGHT_UPDATE_KERNEL__ */
