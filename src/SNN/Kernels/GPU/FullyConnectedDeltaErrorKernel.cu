#ifndef __FULLY_CONNECTED_DELTA_ERROR_KERNEL__
#define __FULLY_CONNECTED_DELTA_ERROR_KERNEL__
#include "CudaUtils.h"
/**
 * Parallel GPU Kernel for a fully connected Network of leaky integrate and fire neurons
 * combining eligibility computation with gradient computation 
 */
namespace SNN {

    namespace Kernels {

        namespace GPU {
            
            __device__ void fullyConnectedDeltaErrorKernel(

                /* the number of input neurons */
                unsigned numInputs,

                /* the number of hidden neurons */
                unsigned numHidden,

                /* the number of output neurons */
                unsigned numOutputs,

                /* the number of (leaky integrate and fire) hiddem neurons */
                unsigned numStandart,

                /* the number of (adaptive leaky integrate and fire) hidden neurons */
                unsigned numAdaptive,

                /* the firing threshold */
                FloatType firingThreshold,

                /* the factor about which the base threshold increases */
                FloatType thresholdIncrease,

                /* the decay factor for the adaptive threshold */
                FloatType adaptionDecay,

                /* the decay factor for the voltage of the neurons */
                FloatType voltageDecay,

                /* the timestep length */
                FloatType timeStepLength,

                /* the filtered (back propagated) output errors */
                FloatType *filteredOutputErrors,

                /* the network target weights */
                FloatType *targetWeights,

                /* the network error factor for one simulation run */
                FloatType *errorFactors,

                /* the voltage component of the delta errors */
                FloatType *voltageError,

                /* the adaption component of the delta errors */
                FloatType *adaptionError,

                /* the neurons spikes */
                FloatType *spikes,

                /* the hidden derivatives */
                FloatType *derivatives,

                /* the hidden voltage */
                FloatType *v,

                /* the synaptic hidden weights */
                FloatType *reccurentWeights,

                /* the synaptic output weights */
                FloatType *readoutWeights

            ) {
                const int i = threadIdx.x;
                const int ai = threadIdx.x - numStandart;;
                if (i >= numHidden) return;

                FloatType learnSignal = 0;

                /* sum errors from output neurons */
                for (unsigned o = 0; o < numOutputs; o++) {
                    learnSignal += 
                        readoutWeights[i * numOutputs + o] * 
                        filteredOutputErrors[o] *
                        errorFactors[o] *
                        targetWeights[o];
                }

                /* sum errors from other hidden neurons from futur timestep */
                for (unsigned h = 0; h < numHidden; h++) {
                    learnSignal += reccurentWeights[i * numHidden + h] * voltageError[h];
                }

                /* add own error from futur timestep */
                learnSignal -= firingThreshold * voltageError[i];

                /* add own error from futur timestep (adaptive part) */
                if (i >= numStandart) 
                    learnSignal += adaptionError[ai];

                /* compute new delta errors */
                if (i >= numStandart) {
                    adaptionError[ai] = 
                        adaptionDecay * adaptionError[ai] - 
                        learnSignal * derivatives[i] * thresholdIncrease;
                }

                __syncthreads();
                voltageError[i] = voltageError[i] * voltageDecay + learnSignal * derivatives[i];
            }
        }
    }
}
#endif /* __FULLY_CONNECTED_DELTA_ERROR_KERNEL__ */
