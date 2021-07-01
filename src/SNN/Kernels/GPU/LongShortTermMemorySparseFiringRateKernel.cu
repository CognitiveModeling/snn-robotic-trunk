#ifndef __LONG_SHORT_TERM_MEMORY_SPARSE_FIRING_RATE_KERNEL__
#define __LONG_SHORT_TERM_MEMORY_SPARSE_FIRING_RATE_KERNEL__
#include "CudaUtils.h"
/**
 * Parallel kernel for a Long Short Term Spiking Network
 */
namespace SNN {

    namespace Kernels {

        namespace GPU {

            __global__ void longShortTermMemorySparseFiringRateKernel(

                /* the number of input neurons */
                unsigned *numInputs_,

                /* the number of (leaky integrate and fire) hiddem neurons */
                unsigned *numStandartHidden_,

                /* the number of (adaptive leaky integrate and fire) hidden neurons */
                unsigned *numAdaptiveHidden_,

                /* the batch size */
                unsigned *batchSize_,

                /* the number of simmulation time steps */
                unsigned *numSimulationTimesteps_,

                /* the simulation timestep length */
                FloatType *timeStepLength_,

                /* the hidden neurons firing rates */
                FloatType *firingRates,

                /* the hidden neurons number of spikes */
                FloatType *numSpikes
            ) {
                const unsigned numInputs              = *numInputs_;
                const unsigned batchSize              = *batchSize_;
                const unsigned numSimulationTimesteps = *numSimulationTimesteps_;
                const FloatType timeStepLength        = *timeStepLength_;

                const unsigned numHidden = *numStandartHidden_ + *numAdaptiveHidden_;
                
                for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; 
                     i < numHidden;
                     i += gridDim.x * blockDim.x) {

                    FloatType summedSpikes = 0;
                    for (unsigned b = 0; b < batchSize; b++)
                        summedSpikes += numSpikes[b * (numInputs + numHidden) + numInputs + i];

                    firingRates[i] = summedSpikes / (batchSize * numSimulationTimesteps * timeStepLength);
                }
            }
        }
    }
}
#endif /* __LONG_SHORT_TERM_MEMORY_SPARSE_FIRING_RATE_KERNEL__ */
