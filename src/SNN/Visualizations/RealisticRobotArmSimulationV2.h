#ifndef __REALISTIC_ROBOT_ARM_SIMULATION_V2_H__
#define __REALISTIC_ROBOT_ARM_SIMULATION_V2_H__
#include <memory>
#include <vector>
#include "utils.h"
#include "BasicRobotArmSimulation.h"
#include "RobotArmPrediction.h"

namespace SNN {
    
    namespace Visualizations {

        /* forward declarations */
        class VTK3DWindow;
        class RealisticRobotArmModuleV2;
        class RealisticRobotArmGripperV2;
        class SimpleRobotArmPrediction;
        class STLObject;

        class RealisticRobotArmSimulationV2: public BasicRobotArmSimulation {
            
            protected:

                /* the window of this */
                std::shared_ptr<VTK3DWindow> window;

                /* the robot stand */
                std::shared_ptr<STLObject> stand;
                
                /* the first unmovable module */
                std::shared_ptr<RealisticRobotArmModuleV2> firstModule;

                /* the joints of the robot arm */
                std::vector<std::shared_ptr<RealisticRobotArmModuleV2>> joints;

                /* the predictions */
                std::vector<std::shared_ptr<SimpleRobotArmPrediction>> predictions;

                /* the inference of this */
                std::vector<std::shared_ptr<RealisticRobotArmModuleV2>> inference;

                /* the gripper for the standart arm, inference, and prediction */
                std::shared_ptr<RealisticRobotArmGripperV2> gripper, inferenceGripper, predictionGripper;

                /* the inference target */
                std::shared_ptr<RobotArmPrediction> inferenceTarget;

                /* random number generator */
                rand128_t *rand;

                /* mean values for normalization */
                FloatType meanX, meanY, meanZ;

                /* the distance between two joints */
                FloatType jointDistance;

                /* indecates that the renderer is deactivated */
                bool rendererDeactivated;

            public:

                /* constructor */
                RealisticRobotArmSimulationV2(
                    unsigned length,
                    std::string baseSTL,
                    std::string motorSTL,
                    std::string linearGearSTL,
                    std::string ballJointSTL,
                    std::string gearSTL,
                    std::string standSTL,
                    std::string gripperBaseSTL, 
                    std::string gripperMotorSTL,
                    std::string linearActuatorSTL,
                    std::string gripperSTL,
                    std::string gripperGearSTL,
                    bool inference = false,
                    bool visualize = true,
                    bool visualizeGrid = true
                );

                /* returns the position of the inference target */
                void getInferenceTarget(
                    FloatType &x, 
                    FloatType &y, 
                    FloatType &z,
                    FloatType &xUp, 
                    FloatType &yUp, 
                    FloatType &zUp,
                    FloatType &xXDirection, 
                    FloatType &yXDirection, 
                    FloatType &zXDirection
                );

                /* sets the position of the inference target */
                void setInferenceTarget(
                    FloatType x, 
                    FloatType y, 
                    FloatType z,
                    FloatType xUp = 0, 
                    FloatType yUp = 0, 
                    FloatType zUp = 0,
                    FloatType xXDirection = 0, 
                    FloatType yXDirection = 0, 
                    FloatType zXDirection = 0
                );

                /* returns the position of the i'th joint */
                void getPosition(
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
                );

                /* sets the prediction for the i'th joint */
                void setPrediction(
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
                );

                /* returns the prediction for the i'th joint */
                void getPrediction(
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
                );

                /* sets the inference for the i'th joint */
                void setInferredAngles(
                    unsigned index, 
                    FloatType xAngle, 
                    FloatType yAngle, 
                    FloatType height
                );

                /* returns the inferece angles for the i'th joint */
                void getInferredAngles(
                    unsigned index, 
                    FloatType &xAngle, 
                    FloatType &yAngle, 
                    FloatType &height
                );

                /* returns the inferece orientation for the i'th joint */
                void getInferredOrientation(
                    unsigned index, 
                    FloatType &w,
                    FloatType &x,
                    FloatType &y,
                    FloatType &z
                );

                /* returns the angles for the i'th joint */
                void getAngles(
                    unsigned index, 
                    FloatType &xAngle, 
                    FloatType &yAngle, 
                    FloatType &height
                );

                /* sets the angles for the i'th joint */
                void setAngles(
                    unsigned index, 
                    FloatType xAngle, 
                    FloatType yAngle, 
                    FloatType height
                );

                /* returns error in euclidean distance for the prediction of this */
                FloatType getPredictionError(unsigned numJoints = 0);

                /* returns error in euclidean distance for the prediction from the inference of this */
                FloatType getInferencePredictionError();

                /* returns error in euclidean distance for the inferred endeffector possition of this */
                FloatType getInferenceError();

                /* returns error in degree for the inferred endeffector orientation of this */
                FloatType getInferenceOrientationError();

                /* returns the inferred position of the i'th joint */
                void getInferredPosition(
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
                );

                /* let the Arm take a random pose */
                bool randomPose(FloatType anglePercent = 1.0, FloatType edjeCaseJointPercent = 0.25);

                /* returns the number of joints */
                unsigned getNumJoints() { return joints.size() + 1; }

                /* deactivate rendering for faster math */
                void deactivateRenderer();

                /* activate rendering */
                void activateRenderer();

                /* normalizes the given coordinates */
                void normalize(FloatType &x, FloatType &y, FloatType &z);

                /* denormalizes the given coordinates */
                void denormalize(FloatType &x, FloatType &y, FloatType &z);

                /* saves this to the given file descripto */
                void saveState(int fd);
   
                /* loads this from the given file descriptor */
                void skipState(int fd);
                void loadState(int fd);

                /* returns the last pressed key (if visualized) */
                std::string getKey();

                /* saves this as a screenshot */
                void saveScreenshot(std::string path, int magnification);

                /* return the inference target inversly transformed by teh given joint */
                void targetFromJoint(unsigned index, FloatType &x, FloatType &y, FloatType &z);

                /* returns the max magle of this */
                virtual FloatType getMaxAngle() { return 1; }

                /* sets the inference for the i'th joint */
                void setInferredAngles(unsigned index, FloatType xAngle, FloatType yAngle) {
                    setInferredAngles(index, xAngle, yAngle, 0);
                }

                /* returns the inferece angles for the i'th joint */
                void getInferredAngles(unsigned index, FloatType &xAngle, FloatType &yAngle) {
                    FloatType unused;
                    getInferredAngles(index, xAngle, yAngle, unused);
                }

                /* returns the angles for the i'th joint */
                void getAngles(unsigned index, FloatType &xAngle, FloatType &yAngle) {
                    FloatType unused;
                    getAngles(index, xAngle, yAngle, unused);
                }

                /* sets the angles for the i'th joint */
                void setAngles(unsigned index, FloatType xAngle, FloatType yAngle) {
                    setAngles(index, xAngle, yAngle, 0);
                }

            public:

                /* returns a vector of all STLObjects of this */
                std::vector<std::shared_ptr<STLObject>> getSTLObjects();

                /* updates the the possition and orientation of this */
                void update();

                /* returns the camera possition */
                void getCamera(
                    double &x,
                    double &y,
                    double &z,
                    double &fx,
                    double &fy,
                    double &fz,
                    double &ux,
                    double &uy,
                    double &uz
                );

                /* sets the camera possition */
                void setCamera(
                    double x,
                    double y,
                    double z,
                    double fx,
                    double fy,
                    double fz,
                    double ux,
                    double uy,
                    double uz
                );
        };

    }
}

#endif
