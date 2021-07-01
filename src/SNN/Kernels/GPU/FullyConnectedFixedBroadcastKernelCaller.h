#ifndef __FULLY_CONNECTED_FIXED_BROADCAST_KERNEL_CALLER__
#define __FULLY_CONNECTED_FIXED_BROADCAST_KERNEL_CALLER__
#include "BasicGPUArray.h"
#include <vector>
/**
 * Parallel FULLY_CONNECTED Kernel for a fully connected Network of neurons
 * SpikePropagation version
 */
namespace SNN {

    namespace Kernels {

        namespace GPU {

            class FullyConnectedFixedBroadcastKernelCaller {

                private:

                    /* the number of GPU blocks */
                    unsigned numBlocks;

                    /* the number of GPU threads */
                    unsigned numThreads;

                    /* the network errors (one for each batch */
                    FloatType *batchErrors;

                    /* the number of input neurons */
                    BasicGPUArray *numInputs;

                    /* the number of hidden neurons */
                    BasicGPUArray *numHidden;

                    /* the number of output neurons */
                    BasicGPUArray *numOutputs;

                    /* the number of simmulation time steps */
                    BasicGPUArray *numSimulationTimesteps;

                    /* the simulation timestep length */
                    BasicGPUArray *timeStepLength;

                    /* neuron spike threshold */
                    BasicGPUArray *spikeThreshold;

                    /* neuron refactory period */
                    BasicGPUArray *refactoryPeriod;

                    /* the hidden voltage decay factor */
                    BasicGPUArray *hiddenDecayFactor;

                    /* the readout voltage decay factor */
                    BasicGPUArray *readoutDecayFactor;

                    /* the target firing rate */
                    BasicGPUArray *targetFiringRate;

                    /* the firing rate gradient scalling factor */
                    BasicGPUArray *firingRateScallingFactor;

                    /* the derivative dumping factor */
                    BasicGPUArray *derivativeDumpingFactor;

                    /* the input neuron spikes over one simulation run */
                    BasicGPUArray *inputSpikesOverTime;

                    /* the hidden neurons firing rates */
                    BasicGPUArray *firingRates;

                    /* the hidden neurons number of spikes */
                    BasicGPUArray *numSpikes;

                    /* the synaptic input weights */
                    BasicGPUArray *inputWeights;

                    /* the synaptic input weights */
                    BasicGPUArray *hiddenWeights;

                    /* the synaptic input weights */
                    BasicGPUArray *outputWeights;

                    /* the feedback weights */
                    BasicGPUArray *feedbackWeights;

                    /* the network targets fore one simulation run */
                    BasicGPUArray *targetsOverTime;

                    /* the fixed braodcast gradients for input synapses */
                    BasicGPUArray *inputFixedBroadcastGradients;

                    /* the firing rate gradients for input synapses  */
                    BasicGPUArray *inputFiringRateGradients;

                    /* the fixed braodcast gradients for hidden synapses */
                    BasicGPUArray *hiddenFixedBroadcastGradients;

                    /* the firing rate gradients for hidden synapses */
                    BasicGPUArray *hiddenFiringRateGradients;

                    /* the leaky readout gradients */
                    BasicGPUArray *leakyReadoutGradients;

                    /* the networks summed error */
                    BasicGPUArray *networkError;

                    /***** content managed by kernel ******/

                    /* the filtered eligibility traces */
                    BasicGPUArray *filteredEligibilityTraces;

                    /* the filtered hidden spikes */
                    BasicGPUArray *filteredSpikes;

                    /* hidden derivatives */
                    BasicGPUArray *derivatives;

                    /* input current for hidden and output neurons */
                    BasicGPUArray *I;

                    /* hidden and readout voltage */
                    BasicGPUArray *v;

                    /* hidden spikes */
                    BasicGPUArray *hiddenSpikes;

                    /* time since last spike for hidden neurons */
                    BasicGPUArray *timeStepsSinceLastSpike;

                public:

                    /* constructor */
                    FullyConnectedFixedBroadcastKernelCaller(
                        unsigned batchSize,
                        unsigned numInputs,
                        unsigned numHidden,
                        unsigned numOutputs,
                        unsigned numSimulationTimesteps,
                        FloatType timeStepLength,
                        FloatType spikeThreshold,
                        FloatType refactoryPeriod,
                        FloatType hiddenDecayFactor,
                        FloatType readoutDecayFactor,
                        FloatType targetFiringRate,
                        FloatType firingRateScallingFactor,
                        FloatType derivativeDumpingFactor,
                        std::vector<FloatType *> inputSpikesOverTime,
                        std::vector<FloatType *> firingRates,
                        std::vector<FloatType *> numSpikes,
                        FloatType *inputWeights,
                        FloatType *hiddenWeights,
                        FloatType *outputWeights,
                        FloatType *feedbackWeights,
                        std::vector<FloatType *> targetsOverTime,
                        std::vector<FloatType *> inputFixedBroadcastGradients,
                        std::vector<FloatType *> inputFiringRateGradients,
                        std::vector<FloatType *> hiddenFixedBroadcastGradients,
                        std::vector<FloatType *> hiddenFiringRateGradients,
                        std::vector<FloatType *> leakyReadoutGradients
                    );

                    /* destructor */
                    ~FullyConnectedFixedBroadcastKernelCaller();

                    /* runs the kernel of this (blocks untill finished) */
                    void runAndWait();

                    /* returns the networks squared summed error for the last run */
                    FloatType getError(int batchIndex = 0);
            };
        }
    }
}
#endif /* __FULLY_CONNECTED_FIXED_BROADCAST_KERNEL_CALLER__ */
