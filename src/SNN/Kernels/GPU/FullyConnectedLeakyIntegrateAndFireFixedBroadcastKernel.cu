#ifndef __FULLY_CONNECTED_LEAKY_INTEGRATE_AND_FIRE_FIXED_BROADCAST_KERNEL__
#define __FULLY_CONNECTED_LEAKY_INTEGRATE_AND_FIRE_FIXED_BROADCAST_KERNEL__
#include "FullyConnectedLeakyIntegrateAndFireKernel.cu"
#include "FullyConnectedInputOutputKernel.cu"
#include "FullyConnectedSpikePropagationKernel.cu"
#include "FullyConnectedLeakyIntegrateAndFireEligibilityGradientKernel.cu"
#include "FullyConnectedLeakyReadoutGradientKernel.cu"
/**
 * Parallel FULLY_CONNECTED Kernel for a fully connected Network of neurons
 * SpikePropagation version
 */
namespace SNN {

    namespace Kernels {

        namespace GPU {

            __device__ void fullyConnectedLeakyIntegrateAndFireFixedBroadcastKernel(

                /* the number of input neurons */
                unsigned numInputs,

                /* the number of hidden neurons */
                unsigned numHidden,

                /* the number of output neurons */
                unsigned numOutputs,

                /* the number of simmulation time steps */
                unsigned numSimulationTimesteps,

                /* the simulation timestep length */
                FloatType timeStepLength,

                /* neuron spike threshold */
                FloatType spikeThreshold,

                /* neuron refactory period */
                FloatType refactoryPeriod,

                /* the hidden voltage decay factor */
                FloatType hiddenDecayFactor,

                /* the readout voltage decay factor */
                FloatType readoutDecayFactor,

                /* the target firing rate */
                FloatType targetFiringRate,

                /* the firing rate gradient scalling factor */
                FloatType firingRateScallingFactor,

                /* the derivative dumping factor */
                FloatType derivativeDumpingFactor,

                /* the input neuron spikes over one simulation run */
                FloatType *inputSpikesOverTime,

                /* the hidden neurons firing rates */
                FloatType *firingRates,

                /* the hidden neurons number of spikes */
                FloatType *numSpikes,

                /* the synaptic input weights */
                FloatType *inputWeights,

                /* the synaptic input weights */
                FloatType *hiddenWeights,

                /* the synaptic input weights */
                FloatType *outputWeights,

                /* the feedback weights */
                FloatType *feedbackWeights,

                /* the network targets fore one simulation run */
                FloatType *targetsOverTime,

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
                
                /* the networks squared summed error */
                FloatType *networkError,

                /***** content managed by kernel ******/

                /* the filtered eligibility traces */
                FloatType *filteredEligibilityTraces,

                /* the filtered hidden spikes */
                FloatType *filteredSpikes,

                /* hidden derivatives */
                FloatType *derivatives,

                /* input current for hidden and output neurons */
                FloatType *I,

                /* hidden and readout voltage */
                FloatType *v,

                /* hidden spikes */
                FloatType *hiddenSpikes,

                /* time since last spike for hidden neurons */
                FloatType *timeStepsSinceLastSpike

            ) {
                cudaAssert(numHidden == blockDim.x);

                inputSpikesOverTime           += blockIdx.x * numInputs * numSimulationTimesteps;
                firingRates                   += blockIdx.x * numHidden;
                numSpikes                     += blockIdx.x * numHidden;
                targetsOverTime               += blockIdx.x * numOutputs * numSimulationTimesteps;
                inputFixedBroadcastGradients  += blockIdx.x * numInputs * numHidden;
                inputFiringRateGradients      += blockIdx.x * numInputs * numHidden;
                hiddenFixedBroadcastGradients += blockIdx.x * numHidden * numHidden;
                hiddenFiringRateGradients     += blockIdx.x * numHidden * numHidden;
                leakyReadoutGradients         += blockIdx.x * numHidden * numOutputs;
                networkError                  += blockIdx.x;
                filteredEligibilityTraces     += blockIdx.x * (numInputs * numHidden + numHidden * numHidden);
                filteredSpikes                += blockIdx.x * (numInputs + numHidden);
                derivatives                   += blockIdx.x * numHidden;
                I                             += blockIdx.x * (numHidden + numOutputs);
                v                             += blockIdx.x * (numHidden + numOutputs);
                hiddenSpikes                  += blockIdx.x * numHidden;
                timeStepsSinceLastSpike       += blockIdx.x * numHidden;

                /* clear values */
                const int h = threadIdx.x;
                filteredSpikes[h] = 0;
                numSpikes[h]      = 0;
                hiddenSpikes[h]   = 0;
                I[h]              = 0;
                v[h]              = 0;
                timeStepsSinceLastSpike[h] = 2 * refactoryPeriod;

                if (h == 0) 
                    networkError[0] = 0;

                if (h < numInputs) {
                    filteredSpikes[h + numHidden] = 0;
                }
                if (h < numOutputs) {
                    I[h + numHidden]              = 0;
                    v[h + numHidden]              = 0;
                }

                for (unsigned i = 0; i < numInputs; i++) {
                    inputFiringRateGradients[i * numHidden + h]     = 0;
                    inputFixedBroadcastGradients[i * numHidden + h] = 0;
                    filteredEligibilityTraces[i * numHidden + h]    = 0;
                }
                for (unsigned i = 0; i < numHidden; i++) {
                    hiddenFiringRateGradients[i * numHidden + h]                         = 0;
                    hiddenFixedBroadcastGradients[i * numHidden + h]                     = 0;
                    filteredEligibilityTraces[numInputs * numHidden + i * numHidden + h] = 0;
                }
                for (unsigned i = 0; i < numOutputs; i++) {
                    leakyReadoutGradients[h * numOutputs + i] = 0;
                }
                __syncthreads();

                for (unsigned t = 0; t < numSimulationTimesteps; t++) {
                   
                    fullyConnectedLeakyIntegrateAndFireKernel(
                        0,
                        numHidden,
                        spikeThreshold,
                        hiddenDecayFactor,
                        refactoryPeriod,
                        derivativeDumpingFactor,
                        hiddenSpikes,
                        filteredSpikes + numInputs,
                        numSpikes,
                        I,
                        v,
                        derivatives,
                        timeStepsSinceLastSpike
                    );
                    fullyConnectedInputOutputKernel(
                        numInputs, 
                        numOutputs,
                        hiddenDecayFactor,
                        readoutDecayFactor,
                        I + numHidden, 
                        v + numHidden, 
                        (t > 0) ? inputSpikesOverTime + (t-1) * numInputs : NULL,
                        filteredSpikes
                    );

                    __syncthreads();

                    if (h == 0) {
                        for (unsigned o = 0; o < numOutputs; o++) {
                            networkError[0] += pow(
                                v[numHidden + o] - 
                                targetsOverTime[t * numOutputs + o], 
                                2
                            );
                        }
                    }

                    fullyConnectedSpikePropagationKernel(
                        numInputs,
                        numHidden,
                        numOutputs,
                        inputSpikesOverTime + t * numInputs,
                        hiddenSpikes,
                        I,
                        inputWeights,
                        hiddenWeights,
                        outputWeights
                    );
                    fullyConnectedLeakyIntegrateAndFireEligibilityGradientKernel(
                        numInputs,
                        numHidden,
                        numOutputs,
                        targetFiringRate,
                        firingRateScallingFactor,
                        readoutDecayFactor,
                        filteredSpikes,
                        firingRates,
                        derivatives,
                        feedbackWeights,
                        v + numHidden,
                        targetsOverTime + t * numOutputs,
                        filteredEligibilityTraces,
                        inputFiringRateGradients,
                        inputFixedBroadcastGradients
                    );
                    fullyConnectedLeakyIntegrateAndFireEligibilityGradientKernel(
                        numHidden,
                        numHidden,
                        numOutputs,
                        targetFiringRate,
                        firingRateScallingFactor,
                        readoutDecayFactor,
                        filteredSpikes + numInputs,
                        firingRates,
                        derivatives,
                        feedbackWeights,
                        v + numHidden,
                        targetsOverTime + t * numOutputs,
                        filteredEligibilityTraces + numInputs * numHidden,
                        hiddenFiringRateGradients,
                        hiddenFixedBroadcastGradients
                    );
                    fullyConnectedLeakyReadoutGradientKernel(
                        numHidden,
                        numOutputs,
                        filteredSpikes + numInputs,
                        v + numHidden,
                        targetsOverTime + t * numOutputs,
                        leakyReadoutGradients
                    );

                    __syncthreads();
                }

                firingRates[h] = numSpikes[h] / (numSimulationTimesteps * timeStepLength);
            }
        }
    }
}
#endif /* __FULLY_CONNECTED_LEAKY_INTEGRATE_AND_FIRE_FIXED_BROADCAST_KERNEL__ */
