#ifndef __REALISTIC_ROBOT_ARM_GRIPPER_V2_H__
#define __REALISTIC_ROBOT_ARM_GRIPPER_V2_H__
#include <vtkSmartPointer.h>
#include <vtkTransform.h>
#include <vtkTransformPolyDataFilter.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include "VTK3DObject.h"
#include "VTK3DWindow.h"
#include "VTKInteractor.h"
#include "Threads.h"
#include "RealisticRobotArmModuleV2.h"
#include "STLObject.h"

namespace SNN {

    namespace Visualizations {

        class RealisticRobotArmGripperV2 {

            private:

                /* the main transformation filter of this */
                vtkSmartPointer<vtkTransform> transform;

                /* the base of this */
                std::shared_ptr<STLObject> base;

                /* the motor of this */
                std::shared_ptr<STLObject> motor;

                /* the linear actuator  */
                std::shared_ptr<STLObject> linearActuator;

                /* the actuator gear of this */
                std::shared_ptr<STLObject> gear;

                /* the ball joints */
                std::vector<std::shared_ptr<STLObject>> ballJoints;

                /* the grippers */
                std::vector<std::shared_ptr<STLObject>> grippers;

                /* the current angles and heigh of this */
                FloatType angleX, angleY, height;

                /* transformation to end of the gripper */
                vtkSmartPointer<vtkTransform> gripperTransform;

                /* indicates that this is beeing visualized */
                bool visualized;

            public:

                /* returns wether this has ben rendered */
                bool rendered() {
                    if (!visualized)
                        return true;

                    if (!base->isInitialized() || base->changed() ||
                        !motor->isInitialized() || motor->changed() ||
                        !gear->isInitialized() || gear->changed() ||
                        !linearActuator->isInitialized() || linearActuator->changed()) {
                        return false;
                    }

                    for (int i = 0; i < 4; i++) {
                        if (!ballJoints[i]->isInitialized() || ballJoints[i]->changed())
                            return false;
                    }

                    for (int i = 0; i < 8; i++) {
                        if (!grippers[i]->isInitialized() || grippers[i]->changed())
                            return false;
                    }
                    return true;
                }

                /* returns a vector of all STLObjects of this */
                std::vector<std::shared_ptr<STLObject>> getSTLObjects() {
                    std::vector<std::shared_ptr<STLObject>> objects;
                    if (base != nullptr) {
                        objects.push_back(base);
                        objects.push_back(motor);
                        objects.push_back(linearActuator);
                        objects.push_back(gear);
                        objects.insert(objects.end(), ballJoints.begin(), ballJoints.end());
                        objects.insert(objects.end(), grippers.begin(), grippers.end());
                    }
                    return objects;
                }

                /* constructor */
                RealisticRobotArmGripperV2(
                    std::string baseSTL, 
                    std::string motorSTL,
                    std::string linearActuatorSTL,
                    std::string ballJointSTL,
                    std::string gripperSTL,
                    std::string gearSTL,
                    std::shared_ptr<VTK3DWindow> window = nullptr,
                    FloatType opacity = 255) : 
                    angleX(0), angleY(0), height(0), visualized(window != nullptr) { 
                    
                    if (visualized) {
                        base  = std::make_shared<STLObject>(baseSTL);   
                        motor = std::make_shared<STLObject>(motorSTL);   
                        gear = std::make_shared<STLObject>(gearSTL);   
                        linearActuator = std::make_shared<STLObject>(linearActuatorSTL);

                        window->add(base);
                        window->add(motor);
                        window->add(gear);
                        window->add(linearActuator);

                        for (int i = 0; i < 4; i++) {
                            ballJoints.push_back(std::make_shared<STLObject>(ballJointSTL));
                            window->add(ballJoints[i]);
                        }
                        for (int i = 0; i < 8; i++) {
                            grippers.push_back(std::make_shared<STLObject>(gripperSTL));
                            window->add(grippers[i]);
                        }

                        while (!this->rendered())
                            Thread::yield();

                        base->setColor(153,153,51,opacity);
                        motor->setColor(35,55,59,opacity);
                        gear->setColor(221,204,119,opacity);
                        linearActuator->setColor(221,204,119,opacity);

                        for (int i = 0; i < 4; i++) {
                            ballJoints[i]->setColor(221,204,119,opacity);
                            ballJoints[i]->addTranslation(0, -33.4, 0);
                            ballJoints[i]->addRotation(-90, 1, 0, 0);
                            ballJoints[i]->addRotation(i * 90 - 45, 0, 0, 1);
                            ballJoints[i]->transform();
                        }
                        for (int i = 0; i < 8; i++) {
                            grippers[i]->setColor(68,170,153,opacity);
                            grippers[i]->addRotation(i  * 45, 0, 1, 0);
                            grippers[i]->addRotation(-90, 1, 0, 0);
                            grippers[i]->transform();
                        }
                        base->addRotation(-90, 1, 0, 0);
                        motor->addRotation(-90, 1, 0, 0);
                        gear->addRotation(-90, 1, 0, 0);
                        linearActuator->addRotation(-90, 1, 0, 0);

                        base->transform();
                        motor->transform();
                        gear->transform();
                        linearActuator->transform();
                    }
                    transform = vtkSmartPointer<vtkTransform>::New();
                    gripperTransform = vtkSmartPointer<vtkTransform>::New();
                }

                /* returns the (relative) height and angles of this */
                FloatType getHeight() { return height; }
                FloatType getXAngle() { return angleX; }
                FloatType getZAngle() { return angleY; }

                void move(
                    FloatType height, 
                    FloatType angleX, 
                    FloatType angleY, 
                    std::shared_ptr<RealisticRobotArmModuleV2> prevModule
                ) {
                    this->move(height, angleX, angleY, prevModule->getTransform());

                    if (prevModule->visualized) {
                        for (int i = 0; i < 4; i++) {
                            prevModule->linearGears[i]->reset();
                            prevModule->gears[i]->reset();
                            prevModule->gears[i]->addRotation(-90, 1, 0, 0);
                            prevModule->gears[i]->addRotation(i * 90, 0, 0, 1);
                            prevModule->gears[i]->addTranslation(-10.3, 0, 12.8);
                        }

                        FloatType xPercent = angleX / 2.0 + 0.5;
                        FloatType yPercent = angleY / 2.0 + 0.5;
                        FloatType hPercent = std::min(xPercent, std::min(yPercent, std::min(1.0-xPercent, 1.0-yPercent))) * height;
                        prevModule->linearGears[0]->addTranslation(0,21.46 * ((1.0 - xPercent) + hPercent), 0);
                        prevModule->linearGears[1]->addTranslation(0,21.46 * (yPercent + hPercent), 0);
                        prevModule->linearGears[2]->addTranslation(0,21.46 * (xPercent + hPercent), 0);
                        prevModule->linearGears[3]->addTranslation(0,21.46 * ((1.0 - yPercent) + hPercent), 0);

                        prevModule->gears[0]->addRotation(-169 * (1.0 - xPercent + hPercent), 0, 1, 0);
                        prevModule->gears[1]->addRotation(-169 * (yPercent + hPercent),         0, 1, 0);
                        prevModule->gears[2]->addRotation(-169 * (xPercent + hPercent),         0, 1, 0);
                        prevModule->gears[3]->addRotation(-169 * (1.0 - yPercent + hPercent), 0, 1, 0);

                        for (int i = 0; i < 4; i++) {
                            prevModule->linearGears[i]->addRotation(-90, 1, 0, 0);
                            prevModule->linearGears[i]->addRotation(i * 90, 0, 0, 1);
                            prevModule->linearGears[i]->transform(prevModule->getTransform());
                            prevModule->gears[i]->addTranslation(10.3, 0, -12.8);
                            prevModule->gears[i]->transform(prevModule->getTransform());
                        }
                    }
                }

                void move(
                    FloatType height, 
                    FloatType angleX, 
                    FloatType angleY, 
                    vtkSmartPointer<vtkTransform> base = NULL
                ) {
                    this->height = height;
                    this->angleX = angleX;
                    this->angleY = angleY;
                    transform->Identity();

                    FloatType xPercent = angleX;
                    FloatType yPercent = angleY;
                    FloatType hPercent = std::max(fabs(yPercent), fabs(xPercent));

                    FloatType xPer = angleX / 2.0 + 0.5;
                    FloatType yPer = angleY / 2.0 + 0.5;
                    FloatType hPer = std::min(xPer, std::min(yPer, std::min(1.0-xPer, 1.0-yPer))) * height;

                    if (base != NULL)
                        transform->Concatenate(base);
                    transform->Translate(-4.5 * yPercent * 0.5, 44.2 + hPer * 21.46 - 0.4 * hPercent, 4.5 * xPercent * 0.5);
                    transform->RotateWXYZ(angleY*16.5, 0, 0, 1);
                    transform->RotateWXYZ(angleX*16.5, 1, 0, 0);
                    transform->RotateWXYZ(45, 0, 1, 0);

                    if (visualized) {
                        FloatType pi = 3.141592653589793;
                        xPercent = (cos(xPercent * pi / 9) - cos(pi / 9)) / (1 - cos(pi/9));
                        xPercent = 1.0 * sgn(xPercent) - xPercent;

                        yPercent = (cos(yPercent * pi / 9) - cos(pi / 9)) / (1 - cos(pi/9));
                        yPercent = 1.0 * sgn(yPercent) - yPercent;

                        FloatType yDerivation1 = 1.1 * yPercent;
                        FloatType yDerivation3 =  -1.1 * yPercent;

                        FloatType yDerivation0 = -0.55 * yPercent;
                        FloatType yDerivation2 = -0.55 * yPercent;

                        FloatType xDerivation1 = -0.55 * xPercent;
                        FloatType xDerivation3 = -0.55 * xPercent;
                        if (angleY * angleX < 0) {
                            yDerivation0 = 3*0.55 * yPercent;
                            yDerivation2 = 3*0.55 * yPercent;

                            xDerivation1 = 0.55 * xPercent;
                            xDerivation3 = 0.55 * xPercent;
                        }

                        FloatType xDerivation0 =  -1.1 * xPercent;
                        FloatType xDerivation2 = 1.1 * xPercent;

                        FloatType xyDerivation = ((angleX * angleY > 0) ? -3.5 : 3.5) * fabs(angleX ) * fabs(angleY);

                        for (int i = 0; i < 4; i++) {
                            ballJoints[i]->reset();
                        }

                        this->ballJoints[0]->addTranslation(xDerivation0 - yDerivation0 + xyDerivation, -33.4, -xDerivation0 - yDerivation0 + xyDerivation);
                        this->ballJoints[1]->addTranslation(yDerivation1 - xDerivation1, -33.4, yDerivation1 + xDerivation1);
                        this->ballJoints[2]->addTranslation(xDerivation2 + yDerivation2 - xyDerivation, -33.4, -xDerivation2 + yDerivation2 - xyDerivation);
                        this->ballJoints[3]->addTranslation(yDerivation3 + xDerivation3, -33.4, yDerivation3 - xDerivation3);

                        for (int i = 0; i < 4; i++) {
                            ballJoints[i]->addRotation(-90, 1, 0, 0);
                            ballJoints[i]->addRotation(i * 90 - 45, 0, 0, 1);
                        }

                        this->base->transform(transform);
                        this->motor->transform(transform);
                        this->gear->transform(transform);
                        this->linearActuator->transform(transform);
                        for (int i = 0; i < 4; i++) {
                            this->ballJoints[i]->transform(transform);
                        }
                        for (int i = 0; i < 8; i++) {
                            this->grippers[i]->transform(transform);
                        }
                    }
                }

                vtkSmartPointer<vtkTransform> getTransform() { return transform; }

                /* returns the gripper tip */
                void getGripperTip(FloatType &x, FloatType &y, FloatType &z) {
                    double center[3];
                    gripperTransform->Identity();
                    gripperTransform->Concatenate(transform);
                    gripperTransform->Translate(0, 92.5, 0);
                    gripperTransform->GetPosition(center);

                    x = center[0];
                    y = center[1];
                    z = center[2];
                }

                /* returns the center of this */
                void getCenter(FloatType &x, FloatType &y, FloatType &z) {
                    double center[3];
                    transform->GetPosition(center);

                    x = center[0];
                    y = center[1];
                    z = center[2];
                }

                /* returns the orientation of this */
                void getOrientation(FloatType &angle, FloatType &x, FloatType &y, FloatType &z) {
                    double wxyz[4];
                    transform->GetOrientationWXYZ(wxyz);

                    angle = wxyz[0];
                    x = wxyz[1];
                    y = wxyz[2];
                    z = wxyz[3];
                }

                /* returns the up vector of this */
                void getUpVector(FloatType &x, FloatType &y, FloatType &z) {
                    double up[3] = { 0, 1, 0 };
                    transform->TransformVector(up, up);

                    x = up[0];
                    y = up[1];
                    z = up[2];
                }

                /* returns the x direction vector of this */
                void getXDirection(FloatType &x, FloatType &y, FloatType &z) {
                    double xDirection[3] = { 1, 0, 0 };
                    transform->TransformVector(xDirection, xDirection);

                    x = xDirection[0];
                    y = xDirection[1];
                    z = xDirection[2];
                }

                /* deactivate rendering for faster math */
                void deactivateRenderer() {
                    while (!this->rendered())
                        Thread::yield();
                        
                    visualized = false;
                }


                /* activate rendering */
                void activateRenderer() {
                    if (base != nullptr)
                        visualized = true;
                }


                /* request rendering */
                void render() {
                    if (base != nullptr) {
                        this->base->transform(transform);
                        this->motor->transform(transform);
                        this->gear->transform(transform);
                        this->linearActuator->transform(transform);
                        for (int i = 0; i < 4; i++) {
                            this->ballJoints[i]->transform(transform);
                        }
                        for (int i = 0; i < 8; i++) {
                            this->grippers[i]->transform(transform);
                        }
                    }
                }
        };
    }
}
#endif /* __REALISTIC_ROBOT_ARM_GRIPPER_V2_H__ */
