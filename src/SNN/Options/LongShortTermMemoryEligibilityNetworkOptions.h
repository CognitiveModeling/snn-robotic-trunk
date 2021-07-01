/**
 * global network options
 */
#ifndef __LONG_SHORT_TERM_MEMORY_ELIGIBILITY_NETWORK_OPTIONS_H__
#define __LONG_SHORT_TERM_MEMORY_ELIGIBILITY_NETWORK_OPTIONS_H__
#include "BasicNetworkOptions.h"

#include <assert.h>
#include <inttypes.h>

namespace SNN {

    namespace Options {

        class LongShortTermMemoryEligibilityNetworkOptions: public Interfaces::BasicNetworkOptions {

            public:

            /* number of (leaky integrate and fire) hidden neurons */
            addOption(unsigned, numStandartHiddenNeurons, -1, true)

            /* number of (adaptive leaky integrate and fire) hidden neurons */
            addOption(unsigned, numAdaptiveHiddenNeurons, -1, true)

            /* wether to shuffle the samples within each new processed batch */
            addOption(bool, shuffleBatch, false)

            /* parameter for the neuron firing threshold */
            addOption(FloatType, spikeThreshold, 0.61);

            /* hidden neuron membran time constant */
            addOption(FloatType, hiddenMembranTimeConstant, 20);

            /* readout neuron membran time constant */
            addOption(FloatType, readoutMembranTimeConstant, 20);

            /* refactory period in simulation milliseconds */
            addOption(FloatType, refactoryPeriod, 5);

            /* the optimizer type for the main loss */
            addOption(OptimizerType, optimizer, OptimizerType::Adam);

            /* the learn rate */
            addOption(FloatType, learnRate, 0.003)

            /* the momentum for SignDampedMomentumOptimizer */
            addOption(FloatType, momentum, 0.3);

            /* the learn rate decay */
            addOption(FloatType, learnRateDecay, 0.7)

            /* the learn rate decay intervall (in batches) */
            addOption(unsigned, learnRateDecayIntervall, 100)

            /* the optimizer type for the regularizer loss */
            addOption(OptimizerType, regularizerOptimizer, OptimizerType::StochasticGradientDescent);

            /* the regularizer lern rate */
            addOption(FloatType, regularizerLearnRate, 0.003);

            /* the momentum for SignDampedMomentumOptimizer (regularizer) */
            addOption(FloatType, regularizerMomentum, 0.1);

            /* the learn rate decay for the regularizer */
            addOption(FloatType, regularizerLearnRateDecay, 0.7)

            /* the learn rate decay intervall (in batches) for the regularizer */
            addOption(unsigned, regularizerLearnRateDecayIntervall, 100)

            /* the threashold increase constant for adaptive leaky integrate and fire neurons */
            addOption(FloatType, thresholdIncreaseConstant, 0.03)

            /* decay constant for adaptive threshold back to baseline */
            addOption(FloatType, adaptationTimeConstant, 1200)

            /* optimizer update intervall in simulation timesteps */
            addOption(unsigned, optimizerUpdateInterval, 1000)

            /* the pseudo derivative dumping factor */
            addOption(FloatType, derivativeDumpingFactor, 0.3);

            /* the number of threads for cpu optimization */
            addOption(unsigned, numThreads, 8);

            /* debug propability (modifiable) */
            addOption(FloatType, debugPropability, 0.1, false, true);

            /* the network error mode */
#define ERROR_MODE_REGRESSION 0
#define ERROR_MODE_CLASSIFICATION 1
#define ERROR_MODE_INFINITY 2
#define ERROR_MODE_LEARNSIGNAL 3
            addOption(unsigned, errorMode, ERROR_MODE_REGRESSION);

            /* the training set size */
            addOption(unsigned, trainSetSize, 1);
            
            /* the intervall in seconds to save the moddel during training */
            addOption(uint64_t, saveInterval, 3600);
            
            /* epoch at wich to start saveing every impoved modle */
            addOption(unsigned, saveStartEpoch, 10000000);

            /* wether to use input errors or not */
            addOption(bool, useInputErrors, false);

            /* wether to use izhikevich neurons or not */
            addOption(bool, izhikevich, false);

            /* when to start pruning weight (-1 = no pruning) */
            addOption(int, pruneStart, -1);

            /* the intervall petween pruning events */
            addOption(int, pruneIntervall, 1000);

            /* the prune strength (smaler = more pruning */
            addOption(FloatType, pruneStrength, 2.0);

            /* wether to stop training if we can not further prune */
            addOption(bool, stopOnNoPrune, true);

            /* the evalulation percentage (from the total traioning epochs) for NEAT */
            addOption(FloatType, neatEvaluationPercent, 0.05);

            /* wether to use eligibility back propagation or not */
            addOption(bool, useEligibilityBackPropagation, false);

            /* intervall for e-prop 3 to provide synthetic gradients */
            addOption(unsigned, eprop3Interval, 12);

            /* wether to use symetric eprop1 or not */
            addOption(bool, symetricEprop1, false);

            /* time multiplayer for eprop3 error mpodule */
            addOption(unsigned, eprop3TimeFactor, 1);

            /* learnrate for synapses to izhikevich neurons */
            addOption(FloatType, izhikevichLearnRate, 0.1);

            /* wether to use an adaptive learning rate or not */
            addOption(bool, adaptiveLearningRate, false);
        };
    }
}

#endif /* __LONG_SHORT_TERM_MEMORY_ELIGIBILITY_NETWORK_OPTIONS_H__ */
