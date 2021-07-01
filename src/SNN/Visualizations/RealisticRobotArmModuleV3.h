#ifndef __REALISTIC_ROBOT_ARM_MODULE_V3_H__
#define __REALISTIC_ROBOT_ARM_MODULE_V3_H__
#include <vtkSmartPointer.h>
#include <vtkTransform.h>
#include <vtkTransformPolyDataFilter.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include "VTK3DObject.h"
#include "VTK3DWindow.h"
#include "VTKInteractor.h"
#include "Threads.h"
#include "STLObject.h"

#define constrain(x, a, b) do { \
    if (x < a) x = a;           \
    if (x > b) x = b;           \
} while(0)

namespace SNN {

    namespace Visualizations {

        class RealisticRobotArmGripperV3;
        class RealisticRobotArmModuleV3 {

            friend class RealisticRobotArmGripperV3;

            private:

                /* the main transformation filter of this */
                vtkSmartPointer<vtkTransform> transform;

                /* the base of this */
                std::shared_ptr<STLObject> base;

                /* the motors of this */
                std::shared_ptr<STLObject> motor;

                /* the linear gears */
                std::vector<std::shared_ptr<STLObject>> linearGears;

                /* the ball joints */
                std::vector<std::shared_ptr<STLObject>> ballJoints;

                /* normal gears */
                std::vector<std::shared_ptr<STLObject>> gears;

                /* the current angles and heigh of this */
                FloatType gearState1, gearState2, gearState3;

                /* indicates that this is beeing visualized */
                bool visualized;

                /* the ball joint states */
                std::vector<FloatType> ballJointStates;

                /* the base transformations: translation rage (0, 53.75), 
                 * rotation angle range(-50, 50) */
                FloatType baseTranslationX, baseTranslationY, baseTranslationZ;
                FloatType baseRotationAngleY, baseRotationAngleX;

            public:

                /* returns wether this has ben rendered */
                bool rendered() {
                    if (!visualized)
                        return true;

                    if (!base->isInitialized() || base->changed() ||
                        !motor->isInitialized() || motor->changed()) {
                        return false;
                    }

                    for (int i = 0; i < 3; i++) {
                        if (!linearGears[i]->isInitialized() || linearGears[i]->changed() ||
                            !ballJoints[i]->isInitialized() || ballJoints[i]->changed() ||
                            !gears[i]->isInitialized() || gears[i]->changed()) {
                            return false;
                        }
                    }
                    return true;
                }

                /* returns a vector of all STLObjects of this */
                std::vector<std::shared_ptr<STLObject>> getSTLObjects() {
                    std::vector<std::shared_ptr<STLObject>> objects;
                    if (base != nullptr) {
                        objects.push_back(base);
                        objects.push_back(motor);
                        objects.insert(objects.end(), linearGears.begin(), linearGears.end());
                        objects.insert(objects.end(), ballJoints.begin(), ballJoints.end());
                        objects.insert(objects.end(), gears.begin(), gears.end());
                    }
                    return objects;
                }

                /* constructor */
                RealisticRobotArmModuleV3(
                    std::string baseSTL, 
                    std::string motorSTL,
                    std::string linearGearSTL,
                    std::string ballJointSTL,
                    std::string gearSTL,
                    std::shared_ptr<VTK3DWindow> window = nullptr,
                    FloatType opacity = 255) : 
                    gearState1(0), gearState2(0), gearState3(0), visualized(window != nullptr),
                    baseTranslationX(0), baseTranslationY(0), baseTranslationZ(0),
                    baseRotationAngleY(0), baseRotationAngleX(0) { 

                    base  = std::make_shared<STLObject>(baseSTL);   
                    motor = std::make_shared<STLObject>(motorSTL);   

                    if (visualized) {
                        window->add(base);
                        window->add(motor);
                    }

                    for (int i = 0; i < 3; i++) {
                        linearGears.push_back(std::make_shared<STLObject>(linearGearSTL));
                        ballJoints.push_back(std::make_shared<STLObject>(ballJointSTL));
                        ballJointStates.push_back(0); 
                        gears.push_back(std::make_shared<STLObject>(gearSTL));

                        if (visualized) {
                            window->add(linearGears[i]);
                            window->add(ballJoints[i]);
                            window->add(gears[i]);
                        }
                    }

                    if (visualized) 
                        while (!this->rendered())
                            Thread::yield();

                    if (visualized) {
                        base->setColor(102,153,204,opacity);
                        motor->setColor(35,55,59,opacity);
                    }

                    base->addRotation(-90, 1, 0, 0);
                    motor->addRotation(-90, 1, 0, 0);

                    base->transform();
                    motor->transform();
                        
                    if (visualized) {
                        for (int i = 0; i < 3; i++) {
                            linearGears[i]->setColor(221,204,119,opacity);
                            ballJoints[i]->setColor(221,204,119,opacity);
                            gears[i]->setColor(221,204,119,opacity);
                        }
                    }
                    transform = vtkSmartPointer<vtkTransform>::New();
                }

                /* returns the (relative) gearState3 and angles of this */
                FloatType getGearState1() { return gearState1; }
                FloatType getGearState2() { return gearState2; }
                FloatType getGearState3() { return gearState3; }

                void getTransfromation(
                    FloatType &baseTranslationX, 
                    FloatType &baseTranslationY, 
                    FloatType &baseTranslationZ,
                    FloatType &baseRotationAngleY, 
                    FloatType &baseRotationAngleX
                ) {
                    baseTranslationX   = this->baseTranslationX;
                    baseTranslationY   = this->baseTranslationY;
                    baseTranslationZ   = this->baseTranslationZ;
                    baseRotationAngleY = this->baseRotationAngleY;
                    baseRotationAngleX = this->baseRotationAngleX;
                }

                void setTransfromation(
                    FloatType baseTranslationX, 
                    FloatType baseTranslationY, 
                    FloatType baseTranslationZ,
                    FloatType baseRotationAngleY, 
                    FloatType baseRotationAngleX,
                    std::shared_ptr<RealisticRobotArmModuleV3> prevModule
                ) {
                    this->baseTranslationX   = baseTranslationX;
                    this->baseTranslationY   = baseTranslationY;
                    this->baseTranslationZ   = baseTranslationZ;
                    this->baseRotationAngleY = baseRotationAngleY;
                    this->baseRotationAngleX = baseRotationAngleX;

                    this->moveSelf(prevModule->getTransform());
                    this->movePrev(prevModule);
                }

                bool move(
                    FloatType gearState3, 
                    FloatType gearState1, 
                    FloatType gearState2, 
                    std::shared_ptr<RealisticRobotArmModuleV3> prevModule,
                    unsigned maxSteps = 10000
                ) {
                    VTKInteractor::lock();
                    this->gearState3 = gearState3;
                    this->gearState1 = gearState1;
                    this->gearState2 = gearState2;

                    FloatType values[20][8];
                    FloatType accuracies[20];
                    FloatType bestAccuracy = accuracy(NULL, prevModule);
                    FloatType worstAccuracy = 0;
                    int bestIndex       = 0;
                    int N               = 10;

                    FloatType F = 0.9, CR = 0.4, CM = 0.0;

                    for (int i = 0; i < N; i++) {
                        if (i < N / 2) {
                            values[i][0] = ballJointStates[0] + rand_range_d128(-7, 26) * std::min(1, bestAccuracy / 100) * std::min(1, i); 
                            values[i][1] = ballJointStates[1] + rand_range_d128(-7, 26) * std::min(1, bestAccuracy / 100) * std::min(1, i);
                            values[i][2] = ballJointStates[2] + rand_range_d128(-7, 26) * std::min(1, bestAccuracy / 100) * std::min(1, i);
                            values[i][3] = baseTranslationX   + rand_range_d128(-60, 60) * std::min(1, bestAccuracy / 100) * std::min(1, i);
                            values[i][4] = baseTranslationY   + rand_range_d128(-1, 55) * std::min(1, bestAccuracy / 100) * std::min(1, i);
                            values[i][5] = baseTranslationZ   + rand_range_d128(-60, 60) * std::min(1, bestAccuracy / 100) * std::min(1, i);
                            values[i][6] = baseRotationAngleY + rand_range_d128(-180, 180) * std::min(1, bestAccuracy / 100) * std::min(1, i);
                            values[i][7] = baseRotationAngleX + rand_range_d128(-50, 50) * std::min(1, bestAccuracy / 100) * std::min(1, i);
                        } else {
                            values[i][0] = rand_range_d128(-7, 26);
                            values[i][1] = rand_range_d128(-7, 26);
                            values[i][2] = rand_range_d128(-7, 26);
                            values[i][3] = rand_range_d128(-60, 60);
                            values[i][4] = rand_range_d128(-1, 55);
                            values[i][5] = rand_range_d128(-60, 60);
                            values[i][6] = rand_range_d128(-180, 180);
                            values[i][7] = rand_range_d128(-50, 50);
                        }

                        constrain(values[i][0], -7, 26);
                        constrain(values[i][1], -7, 26);
                        constrain(values[i][2], -7, 26);
                        constrain(values[i][3], -60, 60);
                        constrain(values[i][4], -1, 55);
                        constrain(values[i][5], -60, 60);
                        constrain(values[i][7], -50, 50);

                        if (values[i][6] < 180) values[N + i][6] += 360;
                        if (values[i][6] > 180) values[N + i][6] -= 360;

                        accuracies[i] = accuracy(values[i], prevModule);

                        if (accuracies[i] < bestAccuracy) {
                            bestAccuracy = accuracies[i];
                            bestIndex = i;
                        }
                        if (worstAccuracy < accuracies[i])
                            worstAccuracy = accuracies[i];
                    }

                    unsigned n = 0;
                    for (; n < maxSteps && bestAccuracy > 0.1; n++) {

                        for (int i = 0; i < N; i++) {
                            int otherIndex1 = rand128() % N;
                            int otherIndex2 = rand128() % N;
                            int otherIndex3 = rand128() % N;
                         
                            while (bestIndex == otherIndex1) // || i == otherIndex1)
                                otherIndex1 = rand128() % N;
                         
                            while (bestIndex   == otherIndex2 || //i == otherIndex2 ||
                                   otherIndex1 == otherIndex2)
                                otherIndex2 = rand128() % N;
                         
                            while (bestIndex   == otherIndex3 || //i == otherIndex3 ||
                                   otherIndex1 == otherIndex3 || 
                                   otherIndex2 == otherIndex3)
                                otherIndex3 = rand128() % N;

                            for (unsigned k = 0; k < 8; k++) {
                                if (rand_range_d128(0.0, 1.0) < CR) {
                                    values[N + i][k] = 
                                        values[i][k] + 
                                        F * (values[bestIndex][k] - values[otherIndex1][k]) +
                                        F * (values[otherIndex2][k] - values[otherIndex3][k]);
                                } else if (rand_range_d128(0.0, 1.0) < CM) {
                                    values[N + i][k] = values[i][k] + rand128n() * pow(bestAccuracy / worstAccuracy, 2);
                                } else {
                                    values[N + i][k] = values[i][k];
                                }
                            }
                         
                            constrain(values[N + i][0], -7, 26);
                            constrain(values[N + i][1], -7, 26);
                            constrain(values[N + i][2], -7, 26);
                            constrain(values[N + i][3], -60, 60);
                            constrain(values[N + i][4], -1, 55);
                            constrain(values[N + i][5], -60, 60);
                            constrain(values[N + i][7], -50, 50);

                            if (values[N + i][6] < 180) values[N + i][6] += 360;
                            if (values[N + i][6] > 180) values[N + i][6] -= 360;
                        }

                        for (int i = 0; i < N; i++) {
                            FloatType acc = accuracy(values[N + i], prevModule);
                            if (acc < accuracies[i]) { 
                                accuracies[i] = acc;
                                for (unsigned k = 0; k < 8; k++) 
                                    values[i][k] = values[N + i][k];

                                if (accuracies[i] < bestAccuracy) {
                                    bestAccuracy = accuracies[i];
                                    bestIndex = i;
                                }
                            }
                            if (worstAccuracy < accuracies[i])
                                worstAccuracy = accuracies[i];
                        }
                    }
                    FloatType finalAccuracy = accuracy(values[bestIndex], prevModule);
                    //printf("Final accuracy: %f | %u\n", finalAccuracy, n);
                    VTKInteractor::unlock();
                    return (finalAccuracy < 0.1);
                }

                FloatType accuracy(
                    FloatType *values,
                    std::shared_ptr<RealisticRobotArmModuleV3> prevModule
                ) {
                    if (values != NULL) {
                        ballJointStates[0] = values[0];
                        ballJointStates[1] = values[1];
                        ballJointStates[2] = values[2];
                        baseTranslationX = values[3];
                        baseTranslationY = values[4];
                        baseTranslationZ = values[5];
                        baseRotationAngleY = values[6];
                        baseRotationAngleX = values[7];
                    }
                    this->moveSelf(prevModule->getTransform());
                    this->movePrev(prevModule);

                    FloatType acc = 0;
                    for (unsigned i = 0; i < 3; i++) {
                        double pos1[3], pos2[3];
                        ballJoints[i]->getCenter(pos1[0], pos1[1], pos1[2]);  
                        prevModule->linearGears[(i + 2) % 3]->getCenter(pos2[0], pos2[1], pos2[2]);  
                        acc += sqrt(
                            pow(pos1[0] - pos2[0], 2) + 
                            pow(pos1[1] - pos2[1], 2) + 
                            pow(pos1[2] - pos2[2], 2)
                        );
                    }

                    return acc;
                }

                void movePrev(std::shared_ptr<RealisticRobotArmModuleV3> prevModule) {
                    for (int i = 0; i < 3; i++) {
                        prevModule->linearGears[i]->reset();
                        prevModule->gears[i]->reset();
                        prevModule->gears[i]->addRotation(-90, 1, 0, 0);
                        prevModule->gears[i]->addRotation(i * 120, 0, 0, 1);
                        prevModule->gears[i]->addTranslation(38.025, -23.5, 0);
                        prevModule->gears[i]->addRotation(89.5, 1, 0, 0);
                    }
                    prevModule->gears[0]->addRotation(180 * gearState1, 1, 0, 0);
                    prevModule->gears[1]->addRotation(180 * gearState2, 1, 0, 0);
                    prevModule->gears[2]->addRotation(180 * gearState3, 1, 0, 0);

                    for (int i = 0; i < 3; i++) {
                        prevModule->gears[i]->addRotation(90, 0, 1, 0);
                    }

                    prevModule->linearGears[0]->addTranslation(0,53.75 * gearState1, 0);
                    prevModule->linearGears[1]->addTranslation(0,53.75 * gearState2, 0);
                    prevModule->linearGears[2]->addTranslation(0,53.75 * gearState3, 0);


                    for (int i = 0; i < 3; i++) {
                        prevModule->linearGears[i]->addRotation(-90, 1, 0, 0);
                        prevModule->linearGears[i]->addRotation(i * 120, 0, 0, 1);
                        prevModule->linearGears[i]->addTranslation(40.5, 0, 44);
                        prevModule->linearGears[i]->transform(prevModule->getTransform());
                        prevModule->gears[i]->transform(prevModule->getTransform());
                    }
                }

                void moveSelf(vtkSmartPointer<vtkTransform> base = NULL) {
                    transform->Identity();

                    if (base != NULL)
                        transform->Concatenate(base);
                    transform->Translate(baseTranslationX, 83.02 + baseTranslationY, baseTranslationZ);
                    transform->RotateWXYZ(baseRotationAngleY, 0, 1, 0);
                    transform->RotateWXYZ(baseRotationAngleX, 1, 0, 0);
                    transform->RotateWXYZ(baseRotationAngleY, 0, -1, 0);
                    transform->RotateWXYZ(180, 0, 1, 0);

                   for (int i = 0; i < 3; i++) {
                        ballJoints[i]->reset();
                    }

                   for (int i = 0; i < 3; i++) {
                        ballJoints[i]->addRotation(-90, 1, 0, 0);
                        ballJoints[i]->addRotation(i * 120 + 60, 0, 0, 1);
                        ballJoints[i]->addTranslation(40.525 + ballJointStates[i], 0, -38.875);
                    }

                    this->base->transform(transform);
                    this->motor->transform(transform);
                    for (int i = 0; i < 3; i++) {
                        this->linearGears[i]->transform(transform);
                        this->ballJoints[i]->transform(transform);
                        this->gears[i]->transform(transform);
                    }
                }

                vtkSmartPointer<vtkTransform> getTransform() { return transform; }

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
                        for (int i = 0; i < 3; i++) {
                            this->linearGears[i]->transform(transform);
                            this->ballJoints[i]->transform(transform);
                            this->gears[i]->transform(transform);
                        }
                    }
                }
        };
    }
}
#endif /* __REALISTIC_ROBOT_ARM_MODULE_V3_H__ */
