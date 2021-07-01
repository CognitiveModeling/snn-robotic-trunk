#ifndef __BASIC_ROBOT_ARM_SIMULATION_H__
#define __BASIC_ROBOT_ARM_SIMULATION_H__
#include <memory>
#include <vector>
#include "utils.h"
#include <assert.h>

namespace SNN {
    
    namespace Visualizations {

        /* forward declarations */
        class VTK3DWindow;
        class BasicRobotArmModule;
        class RobotArmPrediction;

        class BasicRobotArmSimulation {
            
            public:

                virtual ~BasicRobotArmSimulation() { }

                /* returns the position of the inference target */
                virtual void getInferenceTarget(
                    FloatType &x, 
                    FloatType &y, 
                    FloatType &z,
                    FloatType &xUp, 
                    FloatType &yUp, 
                    FloatType &zUp,
                    FloatType &xXDirection, 
                    FloatType &yXDirection, 
                    FloatType &zXDirection
                ) = 0;

                /* sets the position of the inference target */
                virtual void setInferenceTarget(
                    FloatType x, 
                    FloatType y, 
                    FloatType z,
                    FloatType xUp = 0, 
                    FloatType yUp = 0, 
                    FloatType zUp = 0,
                    FloatType xXDirection = 0, 
                    FloatType yXDirection = 0, 
                    FloatType zXDirection = 0
                ) = 0;

                /* returns the position of the i'th joint */
                virtual void getPosition(
                    unsigned index, 
                    FloatType &x, 
                    FloatType &y, 
                    FloatType &z,
                    FloatType &xUp, 
                    FloatType &yUp, 
                    FloatType &zUp,
                    FloatType &xXDirection, 
                    FloatType &yXDirection, 
                    FloatType &zXDirection
                ) = 0;

                /* sets the prediction for the i'th joint */
                virtual void setPrediction(
                    unsigned index, 
                    FloatType x, 
                    FloatType y, 
                    FloatType z,
                    FloatType xUp, 
                    FloatType yUp, 
                    FloatType zUp,
                    FloatType xXDirection, 
                    FloatType yXDirection, 
                    FloatType zXDirection
                ) = 0;

                /* returns the prediction for the i'th joint */
                virtual void getPrediction(
                    unsigned index, 
                    FloatType &x, 
                    FloatType &y, 
                    FloatType &z,
                    FloatType &xUp, 
                    FloatType &yUp, 
                    FloatType &zUp,
                    FloatType &xXDirection, 
                    FloatType &yXDirection, 
                    FloatType &zXDirection
                ) = 0;

                /* sets the inference for the i'th joint */
                virtual void setInferredAngles(unsigned index, FloatType xAngle, FloatType yAngle) = 0;

                /* returns the inferece angles for the i'th joint */
                virtual void getInferredAngles(unsigned index, FloatType &xAngle, FloatType &yAngle) = 0;

                /* returns the angles for the i'th joint */
                virtual void getAngles(unsigned index, FloatType &xAngle, FloatType &yAngle) = 0;

                /* sets the angles for the i'th joint */
                virtual void setAngles(unsigned index, FloatType xAngle, FloatType yAngle) = 0;

                /* returns error in euclidean distance for the prediction of this */
                virtual FloatType getPredictionError(unsigned numJoints = 0) = 0;

                /* returns error in euclidean distance for the prediction from the inference of this */
                virtual FloatType getInferencePredictionError() = 0;

                /* returns error in euclidean distance for the inferred endeffector possition of this */
                virtual FloatType getInferenceError() = 0;

                /* returns error in degree for the inferred endeffector orientation of this */
                virtual FloatType getInferenceOrientationError() = 0;

                /* returns the inferred position of the i'th joint */
                virtual void getInferredPosition(
                    unsigned index,
                    FloatType &x, 
                    FloatType &y, 
                    FloatType &z,
                    FloatType &xUp, 
                    FloatType &yUp, 
                    FloatType &zUp,
                    FloatType &xXDirection, 
                    FloatType &yXDirection, 
                    FloatType &zXDirection
                ) = 0;

                /* let the Arm take a random pose */
                virtual bool randomPose(FloatType anglePercent = 1.0, FloatType edjeCaseJointPercent = 0.25) = 0;

                /* returns the number of joints */
                virtual unsigned getNumJoints() = 0;

                /* returns the max magle of this */
                virtual FloatType getMaxAngle() = 0;

                /* deactivate rendering for faster math */
                virtual void deactivateRenderer() = 0;

                /* activate rendering */
                virtual void activateRenderer() = 0;

                /* resets this */
                virtual void reset() { log_err("BasicRobotArmSimulation::reset not implemented", LOG_E); }

                /* normalizes the given coordinates */
                virtual void normalize(FloatType &x, FloatType &y, FloatType &z) = 0;

                /* denormalizes the given coordinates */
                virtual void denormalize(FloatType &x, FloatType &y, FloatType &z) = 0;

                /* saves this to the given file descripto */
                virtual void saveState(int fd) = 0;
   
                /* loads this from the given file descriptor */
                virtual void loadState(int fd) = 0;

                /* returns the last pressed key (if visualized) */
                virtual std::string getKey() = 0;

                /* choose a random reachable target for inference */
                virtual void randomTargetDensityCorrected(FloatType &x, FloatType &y, FloatType &z) {
                    assert(false); (void) x; (void) y; (void) z;
                }

                /* returns the checksum of this */
                virtual FloatType getChecksum() { return 0; }

                /* returns the initial random state of this */
                virtual rand128_t getInitialRandomState() { return rand128_t(); }

                /* sets the initial random state of this */
                virtual void setInitialRandomState(rand128_t initialRandomState)  { assert(false); (void) initialRandomState; }

                /* saves this as a screenshot */
                virtual void saveScreenshot(std::string path, int magnification) = 0;

        };

    }
}

#endif
