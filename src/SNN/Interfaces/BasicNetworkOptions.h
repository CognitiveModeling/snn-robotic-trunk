/**
 * global network options
 */
#ifndef __BASIC_NETWORK_OPTIONS_H__
#define __BASIC_NETWORK_OPTIONS_H__

#include <assert.h>

/**
 * Preprocessor macro for easy creation of getter and setter methods 
 * for a specific option with or wihtout default parameters
 * */
#define addOptionFull(type, name, value, mandatory, modifiable) \
                                                                \
    private :                                                   \
                                                                \
        /* the actual value */                                  \
        type _##name = value;                                   \
                                                                \
        /* if the value was set */                              \
        bool setted_##name = false;                             \
                                                                \
    public  :                                                   \
                                                                \
        /* getter method */                                     \
        type name() const {                                     \
            assert(                                             \
                (!mandatory || setted_##name) &&                \
                "mandatory option not set"                      \
            );                                                  \
            return _##name;                                     \
        }                                                       \
                                                                \
        /* setter method when opts aren't frozen */             \
        void name(type arg) {                                   \
            assert(!this->frozen || modifiable);                \
            if (!this->frozen || modifiable) {                  \
                _##name = arg;                                  \
                setted_##name = true;                           \
            }                                                   \
        }                                                       
#define addOptionShort1(type, name, value) addOptionFull(type, name, value, false, false)
#define addOptionShort2(type, name, value, mandatory) addOptionFull(type, name, value, mandatory, false)
#define addOptionX(X, T1, T2, T3, T4, T5, FUNC, ...) FUNC
#define addOption(...) addOptionX(,     \
    ##__VA_ARGS__,                      \
    addOptionFull(__VA_ARGS__),         \
    addOptionShort2(__VA_ARGS__),       \
    addOptionShort1(__VA_ARGS__)        \
)

                                                                

namespace SNN {

    namespace Options {

        /* the different optimizer types */
        enum class OptimizerType { 
            StochasticGradientDescent,
            Adam,
            Nadam,
            AMSGrad,
            SignDampedMomentum
        };
    }

    namespace Interfaces {

        class BasicNetworkOptions {

            protected:

                /* indicates if this can be edited */
                int frozen = 0;


            public:

                /* standart constructor */
                BasicNetworkOptions() { }
                
                /* returns whether this is frozen */
                bool isFrozen() const { return frozen != 0; }

                /* freezes this so its values can not longer be changed */
                BasicNetworkOptions &freeze() { frozen = 1; return *this; }

                
            /* the simulation timestep length in milliseconds */
            addOption(FloatType, timeStepLength, 1.0)

            /* the target firing rate for the neurons within the network in spikes peer millisecond */
            addOption(FloatType, targetFiringRate, 0.01)

            /* the number of simulation timesteps */
            addOption(unsigned, numSimulationTimesteps, 1000)

            /* the batch size (samples per batch) */
            addOption(unsigned, batchSize, 1)

            /* wether to use back propagation for error calculation */
            addOption(bool, useBackPropagation, false);

            /* wether to use synaptic delays or not */
            addOption(bool, useSynapticDelays, false);

            /* number of input neurons */
            addOption(unsigned, numInputNeurons, -1, true)

            /* total number of hidden neurons */
            addOption(unsigned, numHiddenNeurons, -1, true)

            /* number of output neurons */
            addOption(unsigned, numOutputNeurons, -1, true)
                
            /* wether the network is in training mode or not */
            addOption(bool, training, true, false, true)

            /* the max synaptic delay in milliseconds timesteps */
            addOption(FloatType, maxSynapticDelay, 0.25);

            /* adam beta1, beta2 and epsilon parameters */
            addOption(FloatType, adamBetaOne, 0.9);
            addOption(FloatType, adamBetaTwo, 0.999);
            addOption(FloatType, adamEpsilon, 1e-16);
            addOption(FloatType, adamWeightDecay, 0.001);

            /* synaptic weight scalling factors */
            addOption(FloatType, inputWeightFactor, 1.0);
            addOption(FloatType, hiddenWeightFactor, 1.0);
            addOption(FloatType, outputWeightFactor, 1.0);

        };
    }
}

#endif /* __BASIC_NETWORK_OPTIONS_H__ */
