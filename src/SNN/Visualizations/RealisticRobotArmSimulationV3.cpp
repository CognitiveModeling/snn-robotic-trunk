#include "RealisticRobotArmSimulationV3.h"
#include "RealisticRobotArmModuleV3.h"
#include "STLRobotArmModule.h"
#include "RealisticRobotArmModuleV3.h"
#include "VTKSphere.h"
#include "VTK3DWindow.h"
#include "Simple3DInteractorStyle.h"
#include "Image.h"
#include "STLRobotArmModule.h"
#include "RealisticRobotArmModuleV3.h"
#include "RealisticRobotArmGripperV3.h"
#include "SimpleRobotArmPrediction.h"
#include "VTKGrid.h"
#include "VTKSphere.h"

using namespace SNN;
using namespace Visualizations;

using std::vector;
using std::string;
using std::shared_ptr;
using std::static_pointer_cast;
using std::make_shared;


/* constructor */
RealisticRobotArmSimulationV3::RealisticRobotArmSimulationV3(
    unsigned length,
    string baseSTL,
    string firstModuleSTL,
    string motorSTL,
    string linearGearSTL,
    string ballJointSTL,
    string gearSTL,
    string standSTL,
    string gripperBaseSTL, 
    string gripperMotorSTL,
    string linearActuatorSTL,
    string gripperSTL,
    string gripperGearSTL,
    bool inference,
    bool visualize,
    bool visualizeGrid,
    bool inferreOrientation
) : window(nullptr) {
    this->jointDistance = 93.75 + 72;
    meanX = this->jointDistance;
    meanY = this->jointDistance;
    meanZ = this->jointDistance;

    printf("jointDistance: %f\n", jointDistance);

    stand = make_shared<STLObject>(standSTL);

    if (visualize) {
        rendererDeactivated = false;
        window = make_shared<VTK3DWindow>(make_shared<Simple3DInteractorStyleGenerator>());
        window->start();

        if (visualizeGrid) {
            shared_ptr<VTK3DObject> grid = static_pointer_cast<VTK3DObject>(
                make_shared<VTKGrid>(length * 10,-270, 10)
            );
            window->add(grid);

            while (!grid->isInitialized())
                Thread::yield();

            grid->setColor(50,50,50,255);
        }
        window->add(stand);
        stand->setColor(102,153,204,255);
        stand->addRotation(-90, 1, 0, 0);
        stand->transform();
    } else {
        rendererDeactivated = true;
        stand->update();
    }

    firstModule = make_shared<RealisticRobotArmModuleV3>(
        firstModuleSTL, 
        motorSTL, 
        linearGearSTL, 
        "", 
        gearSTL, 
        window
    );

    for (unsigned i = 0; i < length - 1; i++) {
        this->joints.push_back(make_shared<RealisticRobotArmModuleV3>(
            baseSTL, 
            motorSTL, 
            linearGearSTL, 
            ballJointSTL, 
            gearSTL, 
            (inference ? nullptr : window)
        ));
        if (inference) {
            this->inference.push_back(make_shared<RealisticRobotArmModuleV3>(
                baseSTL, 
                motorSTL, 
                linearGearSTL, 
                ballJointSTL, 
                gearSTL, 
                window
            ));
        }
    }

    gripper = make_shared<RealisticRobotArmGripperV3>(
        gripperBaseSTL, 
        gripperMotorSTL,
        linearActuatorSTL,
        ballJointSTL,
        gripperSTL,
        gripperGearSTL,
        (inference ? nullptr : window)
    );

    if (inference) {
        inferenceGripper = make_shared<RealisticRobotArmGripperV3>(
            gripperBaseSTL, 
            gripperMotorSTL,
            linearActuatorSTL,
            ballJointSTL,
            gripperSTL,
            gripperGearSTL,
            window 
        );
    }

    inferenceTarget = make_shared<RobotArmPrediction>(jointDistance / 2, inferreOrientation);
    inferenceTarget->setColor(180, 0, 0, 128);
    if (window != nullptr && inference)
        window->add(static_pointer_cast<VTK3DObject>(inferenceTarget));
    else
        inferenceTarget->update();

    for (unsigned i = 0; i < length + 2; i++) 
        this->predictions.push_back(make_shared<SimpleRobotArmPrediction>());

}

/* updates the the possition and orientation of this */
void RealisticRobotArmSimulationV3::update() {
    stand->transform();
    firstModule->render();
    for (unsigned i = 0; i < joints.size(); i++)
        joints[i]->render();
    for (unsigned i = 0; i < inference.size(); i++)
        inference[i]->render();
    gripper->render();
    if (inferenceGripper != nullptr)
        inferenceGripper->render();
}

/* let the Ram take a random pose */
bool RealisticRobotArmSimulationV3::randomPose(FloatType anglePercent, FloatType edjeCasePercent) {
    
    bool setted = true;
    FloatType gearState1 = 0;
    FloatType gearState2 = 0;
    FloatType gearState3 = 0;
    FloatType edjeCaseJointPercent = 1.0 - pow(1.0 - edjeCasePercent, 1.0/this->getNumJoints());

    if (anglePercent > 1.0)
        anglePercent = 1.0;

    unsigned i = 0;

    if (rand_range_d128(0.0, 1.0) < 0.2) {
        for (; i < this->getNumJoints(); i++) {
            FloatType strength = rand_range_d128(0,1);
            FloatType g1Off = round(rand_range_d128(0, 1)) * anglePercent * (1 - strength);
            FloatType g2Off = round(rand_range_d128(0, 1)) * anglePercent * (1 - strength);
            FloatType g3Off = round(rand_range_d128(0, 1)) * anglePercent * (1 - strength);
            gearState1 = rand_range_d128(0, anglePercent * strength) + g1Off;
            gearState2 = rand_range_d128(0, anglePercent * strength) + g2Off;
            gearState3 = rand_range_d128(0, anglePercent * strength) + g3Off;

            setted &= this->setGearStates(i, gearState1, gearState2, gearState3);
        }
    }

    if (rand_range_d128(0.0, 1.0) < 0.2) {
        FloatType strength = rand_range_d128(0,1);
        FloatType g1Off = round(rand_range_d128(0, 1)) * anglePercent * (1 - strength);
        FloatType g2Off = round(rand_range_d128(0, 1)) * anglePercent * (1 - strength);
        FloatType g3Off = round(rand_range_d128(0, 1)) * anglePercent * (1 - strength);
        for (; i < this->getNumJoints(); i++) {
            gearState1 = rand_range_d128(0, anglePercent * strength) + g1Off;
            gearState2 = rand_range_d128(0, anglePercent * strength) + g2Off;
            gearState3 = rand_range_d128(0, anglePercent * strength) + g3Off;

            setted &= this->setGearStates(i, gearState1, gearState2, gearState3);
        }
    }

    if (rand_range_d128(0.0, 1.0) < 0.2) {
        FloatType base1 = rand_range_d128(0.0, 1.0);
        FloatType base2 = rand_range_d128(0.0, 1.0);
        FloatType base3 = rand_range_d128(0.0, 1.0);
        for (; i < this->getNumJoints(); i++) {
            gearState1 = base1 + rand128n(0.0, 0.1);
            gearState2 = base2 + rand128n(0.0, 0.1);
            gearState3 = base3 + rand128n(0.0, 0.1);
            if (gearState1 < 0) gearState1 = 0;
            if (gearState1 > 1) gearState1 = 1;
            if (gearState2 < 0) gearState2 = 0;
            if (gearState2 > 1) gearState2 = 1;
            if (gearState3 < 0) gearState3 = 0;
            if (gearState3 > 1) gearState3 = 1;
            setted &= this->setGearStates(i, gearState1, gearState2, gearState3);
        }
    }

    if (rand_range_d128(0.0, 1.0) < 0.025) {
        for (; i < this->getNumJoints(); i++) {
            gearState1 = round(rand_range_d128(0.0, 1.0));
            gearState2 = round(rand_range_d128(0.0, 1.0));
            gearState3 = round(rand_range_d128(0.0, 1.0));

            setted &= this->setGearStates(i, gearState1, gearState2, gearState3);
        }
    }

    for (; i < this->getNumJoints(); i++) {
        gearState1 = rand_range_d128(0, anglePercent);
        gearState2 = rand_range_d128(0, anglePercent);
        gearState3 = rand_range_d128(0, anglePercent);

        if (rand_range_d128(0.0, 1.0) < edjeCaseJointPercent) {
            gearState1 = round(gearState1);
            gearState2 = round(gearState2);
            gearState3 = round(gearState3);
        }

        setted &= this->setGearStates(i, gearState1, gearState2, gearState3);
    }

    return setted;
}

/* returns the position of the inference target */
void RealisticRobotArmSimulationV3::getInferenceTarget(
    FloatType &x, 
    FloatType &y, 
    FloatType &z,
    FloatType &xUp, 
    FloatType &yUp, 
    FloatType &zUp,
    FloatType &xXDirection, 
    FloatType &yXDirection, 
    FloatType &zXDirection
) {
    inferenceTarget->getPosition(x, y, z);
    inferenceTarget->getUpVector(xUp, yUp, zUp);
    inferenceTarget->getXDirection(xXDirection, yXDirection, zXDirection);
    normalize(x, y, z);
}

/* sets the position of the inference target */
void RealisticRobotArmSimulationV3::setInferenceTarget(
    FloatType x, 
    FloatType y, 
    FloatType z,
    FloatType xUp, 
    FloatType yUp, 
    FloatType zUp,
    FloatType xXDirection, 
    FloatType yXDirection, 
    FloatType zXDirection
) {
    denormalize(x, y, z);
    inferenceTarget->setPosition(x, y, z);
    if (fabs(xUp) + fabs(yUp) + fabs(zUp) + fabs(xXDirection) + fabs(yXDirection) + fabs(zXDirection) != 0)
        inferenceTarget->setDirections(x, y, z, xUp, yUp, zUp, xXDirection, yXDirection, zXDirection);
}

/* returns the position of the i'th joint */
void RealisticRobotArmSimulationV3::getPosition(
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
) {
    assert(index < joints.size() + 1);
    if (index == joints.size())
        gripper->getGripperTip(x, y, z);
    else if (index == joints.size() - 1)
        gripper->getCenter(x, y, z);
    else
        joints[index + 1]->getCenter(x, y, z);

    if (index == joints.size()) {
        gripper->getUpVector(xUp, yUp, zUp);
        gripper->getXDirection(xXDirection, yXDirection, zXDirection);
    } else {
        joints[index]->getUpVector(xUp, yUp, zUp);
        joints[index]->getXDirection(xXDirection, yXDirection, zXDirection);
    }

    normalize(x, y, z);
}

/* sets the prediction for the i'th joint */
void RealisticRobotArmSimulationV3::setPrediction(
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
) {
    assert(index < predictions.size() - 2);
    denormalize(x, y, z);

    if (index == 0) {
        predictions[1]->setPosition(0, jointDistance, 0);
        predictions[0]->setDirections(0, 1, 0, 1, 0, 0);
    }

    predictions[index + 2]->setPosition(x, y, z);
    predictions[index + 1]->getPosition(x, y, z);
    predictions[index + 1]->setDirections(xUp, yUp, zUp, xXDirection, yXDirection, zXDirection);
}

/* returns the prediction for the i'th joint */
void RealisticRobotArmSimulationV3::getPrediction(
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
) {
    assert(index < predictions.size() - 2);

    predictions[index + 2]->getPosition(x, y, z);
    predictions[index + 1]->getUpVector(xUp, yUp, zUp);
    predictions[index + 1]->getXDirection(xXDirection, yXDirection, zXDirection);
    normalize(x, y, z);
}

/* normalizes the given coordinates */
void RealisticRobotArmSimulationV3::normalize(FloatType &x, FloatType &y, FloatType &z) {
    x = x / meanX;
    y = (y - jointDistance) / meanY;
    z = z / meanZ;
}

/* denormalizes the given coordinates */
void RealisticRobotArmSimulationV3::denormalize(FloatType &x, FloatType &y, FloatType &z) {
    x = x * meanX;
    y = y * meanY + jointDistance;
    z = z * meanZ;
}

void RealisticRobotArmSimulationV3::getTransfromation(
    unsigned index,
    FloatType &baseTranslationX, 
    FloatType &baseTranslationY, 
    FloatType &baseTranslationZ,
    FloatType &baseRotationAngleY, 
    FloatType &baseRotationAngleX
) {
    assert(index < joints.size() + 1);
    if (index == joints.size()) {
        gripper->getTransfromation(
            baseTranslationX, 
            baseTranslationY, 
            baseTranslationZ,
            baseRotationAngleY, 
            baseRotationAngleX
        );
    } else {
        joints[index]->getTransfromation(
            baseTranslationX, 
            baseTranslationY, 
            baseTranslationZ,
            baseRotationAngleY, 
            baseRotationAngleX
        );
    }
}

void RealisticRobotArmSimulationV3::setTransfromation(
    unsigned index,
    FloatType baseTranslationX, 
    FloatType baseTranslationY, 
    FloatType baseTranslationZ,
    FloatType baseRotationAngleY, 
    FloatType baseRotationAngleX
) {
    assert(index < joints.size() + 1);
    
    if (index == joints.size()) {
        if (joints.size() == 0) {
            gripper->setTransfromation(
                baseTranslationX, 
                baseTranslationY, 
                baseTranslationZ,
                baseRotationAngleY, 
                baseRotationAngleX,
                firstModule
            );
        } else {
            gripper->setTransfromation(
                baseTranslationX, 
                baseTranslationY, 
                baseTranslationZ,
                baseRotationAngleY, 
                baseRotationAngleX,
                joints.back()
            );
        }

        FloatType x, y, z, xUp, yUp, zUp, xXDirection, yXDirection, zXDirection;
        this->getPosition(index, x, y, z, xUp, yUp, zUp, xXDirection, yXDirection, zXDirection);
        this->setInferenceTarget(x, y, z, xUp, yUp, zUp, xXDirection, yXDirection, zXDirection);
    } else if (index == 0) {
        joints[index]->setTransfromation(
            baseTranslationX, 
            baseTranslationY, 
            baseTranslationZ,
            baseRotationAngleY, 
            baseRotationAngleX,
            firstModule
        );
    } else {
        joints[index]->setTransfromation(
            baseTranslationX, 
            baseTranslationY, 
            baseTranslationZ,
            baseRotationAngleY, 
            baseRotationAngleX,
            joints[index - 1]
        );
    }
}

/* returns the gear states for the i'th joint */
void RealisticRobotArmSimulationV3::getGearStates(
    unsigned index, 
    FloatType &gearState1, 
    FloatType &gearState2,
    FloatType &gearState3
) {
    assert(index < joints.size() + 1);
    if (index == joints.size()) {
        gearState1 = gripper->getGearState1();
        gearState2 = gripper->getGearState2();
        gearState3 = gripper->getGearState3();
    } else {
        gearState1 = joints[index]->getGearState1();
        gearState2 = joints[index]->getGearState2();
        gearState3 = joints[index]->getGearState3();
    }

    if (index % 2 == 0) {
        FloatType gMax = std::max(gearState1, std::max(gearState2, gearState3));
        FloatType gMin = std::min(gearState1, std::min(gearState2, gearState3));
        gearState1 = gMin + (gMax - gearState1);
        gearState2 = gMin + (gMax - gearState2);
        gearState3 = gMin + (gMax - gearState3);
    }
}

/* sets the gear states for the i'th joint */
bool RealisticRobotArmSimulationV3::setGearStates(
    unsigned index, 
    FloatType gearState1, 
    FloatType gearState2,
    FloatType gearState3,
    unsigned maxSteps
) {
    assert(index < joints.size() + 1);
    assert(gearState1 >= 0 && gearState1 <= 1);
    assert(gearState2 >= 0 && gearState2 <= 1);
    assert(gearState3 >= 0 && gearState3 <= 1);
    
    if (index % 2 == 0) {
        FloatType gMax = std::max(gearState1, std::max(gearState2, gearState3));
        FloatType gMin = std::min(gearState1, std::min(gearState2, gearState3));
        gearState1 = gMin + (gMax - gearState1);
        gearState2 = gMin + (gMax - gearState2);
        gearState3 = gMin + (gMax - gearState3);
    }

    bool setted = false;
    
    if (index == joints.size()) {
        if (joints.size() == 0)
            setted = gripper->move(gearState3, gearState1, gearState2, firstModule, maxSteps);
        else
            setted = gripper->move(gearState3, gearState1, gearState2, joints.back(), maxSteps);

        FloatType x, y, z, xUp, yUp, zUp, xXDirection, yXDirection, zXDirection;
        this->getPosition(index, x, y, z, xUp, yUp, zUp, xXDirection, yXDirection, zXDirection);
        this->setInferenceTarget(x, y, z, xUp, yUp, zUp, xXDirection, yXDirection, zXDirection);
    } else if (index == 0) {
        setted = joints[index]->move(gearState3, gearState1, gearState2, firstModule, maxSteps);
    } else {
        setted = joints[index]->move(gearState3, gearState1, gearState2, joints[index - 1], maxSteps);
    }
    return setted;
}

/* sets the inference for the i'th joint */
bool RealisticRobotArmSimulationV3::setInferredGearStates(
    unsigned index, 
    FloatType gearState1, 
    FloatType gearState2,
    FloatType gearState3,
    unsigned maxSteps
) {
    assert(index < inference.size() + 1);
    assert(gearState1 >= 0 && gearState1 <= 1);
    assert(gearState2 >= 0 && gearState2 <= 1);
    assert(gearState3 >= 0 && gearState3 <= 1);

    if (index % 2 == 0) {
        FloatType gMax = std::max(gearState1, std::max(gearState2, gearState3));
        FloatType gMin = std::min(gearState1, std::min(gearState2, gearState3));
        gearState1 = gMin + (gMax - gearState1);
        gearState2 = gMin + (gMax - gearState2);
        gearState3 = gMin + (gMax - gearState3);
    }

    bool setted = false;

    if (index == inference.size()) {
        if (inference.size() == 0)
            setted = inferenceGripper->move(gearState3, gearState1, gearState2, firstModule, maxSteps);
        else
            setted = inferenceGripper->move(gearState3, gearState1, gearState2, inference.back(), maxSteps);

    } else if (index == 0) {
        setted = inference[index]->move(gearState3, gearState1, gearState2, firstModule, maxSteps);
    } else {
        setted = inference[index]->move(gearState3, gearState1, gearState2, inference[index - 1], maxSteps);
    }
    return setted;
}

/* returns the inferece gear states for the i'th joint */
void RealisticRobotArmSimulationV3::getInferredGearStates(
    unsigned index, 
    FloatType &gearState1, 
    FloatType &gearState2,
    FloatType &gearState3
) {
    assert(index < inference.size() + 1);
    if (index == inference.size()) {
        gearState1 = inferenceGripper->getGearState1();
        gearState2 = inferenceGripper->getGearState2();
        gearState3 = inferenceGripper->getGearState3();
    } else {
        gearState1 = inference[index]->getGearState1();
        gearState2 = inference[index]->getGearState2();
        gearState3 = inference[index]->getGearState3();
    }

    if (index % 2 == 0) {
        FloatType gMax = std::max(gearState1, std::max(gearState2, gearState3));
        FloatType gMin = std::min(gearState1, std::min(gearState2, gearState3));
        gearState1 = gMin + (gMax - gearState1);
        gearState2 = gMin + (gMax - gearState2);
        gearState3 = gMin + (gMax - gearState3);
    }
}

/* returns the inferece orientation for the i'th joint */
void RealisticRobotArmSimulationV3::getInferredOrientation(
    unsigned index, 
    FloatType &w,
    FloatType &x,
    FloatType &y,
    FloatType &z
) {
    assert(index < inference.size() + 1);
    if (index == inference.size()) {
        inferenceGripper->getOrientation(w, x, y, z);
    } else {
        inference[index]->getOrientation(w, x, y, z);
    }
}

/* deactivate rendering for faster math */
void RealisticRobotArmSimulationV3::deactivateRenderer() {
/*
    if (window != nullptr && !rendererDeactivated) {
        while (!stand->isInitialized() || stand->changed())
            Thread::yield();

        for (auto &j: joints)
            j->deactivateRenderer();

        for (auto &i: inference)
            i->deactivateRenderer();

        firstModule->deactivateRenderer();
        gripper->deactivateRenderer();
        if (inferenceGripper != nullptr)
            inferenceGripper->deactivateRenderer();
        VTKInteractor::lock();
        rendererDeactivated = true;
    }*/
}

/* activate rendering */
void RealisticRobotArmSimulationV3::activateRenderer() {
/*
    if (window != nullptr && rendererDeactivated) {
        for (auto &j: joints)
            j->activateRenderer();

        for (auto &i: inference)
            i->activateRenderer();

        firstModule->activateRenderer();
        gripper->activateRenderer();
        if (inferenceGripper != nullptr)
            inferenceGripper->activateRenderer();
        VTKInteractor::unlock();
        rendererDeactivated = false;
    }*/
}

/* returns error in euclidean distance for the prediction of this */
FloatType RealisticRobotArmSimulationV3::getPredictionError(unsigned numJoints) {

    FloatType error = 0;
    if (numJoints == 0) numJoints = joints.size() + 1;

    for (unsigned i = 0; i < numJoints + 1; i++) {
        FloatType targetX, targetY, targetZ;
        FloatType predictedX, predictedY, predictedZ;

        if (i < joints.size())
            joints[i]->getCenter(targetX, targetY, targetZ);
        else if (i == joints.size())
            gripper->getCenter(targetX, targetY, targetZ);
        else 
            gripper->getGripperTip(targetX, targetY, targetZ);

        predictions[i + 1]->getPosition(predictedX, predictedY, predictedZ);

        error += sqrt(
            pow(targetX - predictedX, 2) + 
            pow(targetY - predictedY, 2) + 
            pow(targetZ - predictedZ, 2)
        );
    }

    return error / (numJoints + 2);
}

/* returns error in euclidean distance for the prediction from the inference of this */
FloatType RealisticRobotArmSimulationV3::getInferencePredictionError() {

    FloatType error = 0;
    for (unsigned i = 0; i < inference.size() + 2; i++) {
        FloatType targetX, targetY, targetZ;
        FloatType predictedX, predictedY, predictedZ;

        if (i < inference.size())
            inference[i]->getCenter(targetX, targetY, targetZ);
        else if (i == inference.size())
            inferenceGripper->getCenter(targetX, targetY, targetZ);
        else 
            inferenceGripper->getGripperTip(targetX, targetY, targetZ);

        predictions[i + 1]->getPosition(predictedX, predictedY, predictedZ);

        error += sqrt(
            pow(targetX - predictedX, 2) + 
            pow(targetY - predictedY, 2) + 
            pow(targetZ - predictedZ, 2)
        );
    }

    return error / (inference.size() + 2);
}

/* returns error in euclidean distance for the inferred endeffector possition of this */
FloatType RealisticRobotArmSimulationV3::getInferenceError() {

    FloatType xInference = 0, yInference = 0, zInference = 0;
    FloatType x = 0, y = 0, z = 0;

    inferenceGripper->getGripperTip(xInference, yInference, zInference);
    inferenceTarget->getPosition(x, y, z);

    FloatType error = sqrt(
        pow(xInference - x, 2) + 
        pow(yInference - y, 2) + 
        pow(zInference - z, 2)
    );
    
    return error;
}

#define vectorAngle(x1, y1, z1, x2, y2, z2) \
    ((180.0 / 3.141592653589793) * acos(((x1) * (x2) + (y1) * (y2) + (z1) * (z2)) / \
    (sqrt(pow(x1, 2) + pow(y1, 2) + pow(z1, 2)) * sqrt(pow(x2, 2) + pow(y2, 2) + pow(z2, 2)))))

/* returns error in degree for the inferred endeffector orientation of this */
FloatType RealisticRobotArmSimulationV3::getInferenceOrientationError() {

    FloatType xUpInference = 0, yUpInference = 0, zUpInference = 0;
    FloatType xXDirectionInference = 0, yXDirectionInference = 0, zXDirectionInference = 0;
    FloatType xUp = 0, yUp = 0, zUp = 0;
    FloatType xXDirection = 0, yXDirection = 0, zXDirection = 0;

    gripper->getUpVector(xUp, yUp, zUp);
    gripper->getXDirection(xXDirection, yXDirection, zXDirection);

    inferenceGripper->getUpVector(xUpInference, yUpInference, zUpInference);
    inferenceGripper->getXDirection(xXDirectionInference, yXDirectionInference, zXDirectionInference);

    FloatType qw, qx, qy, qz, qwInf, qxInf, qyInf, qzInf;
    toQuaternion(
        xUp, 
        yUp, 
        zUp,
        xXDirection, 
        yXDirection, 
        zXDirection,
        qw, qx, qy, qz
    );
    toQuaternion(
        xUpInference, 
        yUpInference, 
        zUpInference,
        xXDirectionInference, 
        yXDirectionInference, 
        zXDirectionInference,
        qwInf, qxInf, qyInf, qzInf
    );

    const FloatType PI = 3.141592653589793;
    return 360 / PI * acos(fabs(
       qw * qwInf + 
       qx * qxInf + 
       qy * qyInf+ 
       qz * qzInf
    ));
}

/* returns the inferred position of the i'th joint */
void RealisticRobotArmSimulationV3::getInferredPosition(
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
) {
    assert(index < inference.size() + 1);
    if (index == inference.size())
        inferenceGripper->getGripperTip(x, y, z);
    else if (index == inference.size() - 1)
        inferenceGripper->getCenter(x, y, z);
    else
        inference[index + 1]->getCenter(x, y, z);

    if (index == inference.size()) {
        inferenceGripper->getUpVector(xUp, yUp, zUp);
        inferenceGripper->getXDirection(xXDirection, yXDirection, zXDirection);
    } else {
        inference[index]->getUpVector(xUp, yUp, zUp);
        inference[index]->getXDirection(xXDirection, yXDirection, zXDirection);
    }

    normalize(x, y, z);
}

/* saves this to the given file descriptor */
void RealisticRobotArmSimulationV3::saveState(int fd) {

    FloatType x; 
    FloatType y; 
    FloatType z;
    FloatType xUp; 
    FloatType yUp; 
    FloatType zUp;
    FloatType xXDirection; 
    FloatType yXDirection; 
    FloatType zXDirection;
    for (unsigned i = 0; i < this->getNumJoints(); i++) {
        FloatType gearState1, gearState2, gearState3;

        this->getGearStates(i, gearState1, gearState2, gearState3);
        assert(write(fd, &gearState1, sizeof(FloatType)) == sizeof(FloatType));
        assert(write(fd, &gearState2, sizeof(FloatType)) == sizeof(FloatType));
        assert(write(fd, &gearState3, sizeof(FloatType)) == sizeof(FloatType));

        if (inference.size() > 0) {
            this->getInferredGearStates(i, gearState1, gearState2, gearState3);
            assert(write(fd, &gearState1, sizeof(FloatType)) == sizeof(FloatType));
            assert(write(fd, &gearState2, sizeof(FloatType)) == sizeof(FloatType));
            assert(write(fd, &gearState3, sizeof(FloatType)) == sizeof(FloatType));
        }
    
        this->getPrediction(i, x, y, z, xUp, yUp, zUp, xXDirection, yXDirection, zXDirection);
        assert(write(fd, &x, sizeof(FloatType)) == sizeof(FloatType));
        assert(write(fd, &y, sizeof(FloatType)) == sizeof(FloatType));
        assert(write(fd, &z, sizeof(FloatType)) == sizeof(FloatType));
        assert(write(fd, &xUp, sizeof(FloatType)) == sizeof(FloatType));
        assert(write(fd, &yUp, sizeof(FloatType)) == sizeof(FloatType));
        assert(write(fd, &zUp, sizeof(FloatType)) == sizeof(FloatType));
        assert(write(fd, &xXDirection, sizeof(FloatType)) == sizeof(FloatType));
        assert(write(fd, &yXDirection, sizeof(FloatType)) == sizeof(FloatType));
        assert(write(fd, &zXDirection, sizeof(FloatType)) == sizeof(FloatType));

    }
    if (inference.size() > 0) {
        this->getInferenceTarget(x, y, z, xUp, yUp, zUp, xXDirection, yXDirection, zXDirection);
        assert(write(fd, &x, sizeof(FloatType)) == sizeof(FloatType));
        assert(write(fd, &y, sizeof(FloatType)) == sizeof(FloatType));
        assert(write(fd, &z, sizeof(FloatType)) == sizeof(FloatType));
        assert(write(fd, &xUp, sizeof(FloatType)) == sizeof(FloatType));
        assert(write(fd, &yUp, sizeof(FloatType)) == sizeof(FloatType));
        assert(write(fd, &zUp, sizeof(FloatType)) == sizeof(FloatType));
        assert(write(fd, &xXDirection, sizeof(FloatType)) == sizeof(FloatType));
        assert(write(fd, &yXDirection, sizeof(FloatType)) == sizeof(FloatType));
        assert(write(fd, &zXDirection, sizeof(FloatType)) == sizeof(FloatType));
    }
}

void RealisticRobotArmSimulationV3::skipState(int fd) {

    FloatType x; 
    FloatType y; 
    FloatType z;
    FloatType xUp; 
    FloatType yUp; 
    FloatType zUp;
    FloatType xXDirection; 
    FloatType yXDirection; 
    FloatType zXDirection;
    for (unsigned i = 0; i < this->getNumJoints(); i++) {
        FloatType gearState1, gearState2, gearState3;

        assert(read(fd, &gearState1, sizeof(FloatType)) == sizeof(FloatType));
        assert(read(fd, &gearState2, sizeof(FloatType)) == sizeof(FloatType));
        assert(read(fd, &gearState3, sizeof(FloatType)) == sizeof(FloatType));

        if (inference.size() > 0) {
            assert(read(fd, &gearState1, sizeof(FloatType)) == sizeof(FloatType));
            assert(read(fd, &gearState2, sizeof(FloatType)) == sizeof(FloatType));
            assert(read(fd, &gearState3, sizeof(FloatType)) == sizeof(FloatType));
        }
    
        assert(read(fd, &x, sizeof(FloatType)) == sizeof(FloatType));
        assert(read(fd, &y, sizeof(FloatType)) == sizeof(FloatType));
        assert(read(fd, &z, sizeof(FloatType)) == sizeof(FloatType));
        assert(read(fd, &xUp, sizeof(FloatType)) == sizeof(FloatType));
        assert(read(fd, &yUp, sizeof(FloatType)) == sizeof(FloatType));
        assert(read(fd, &zUp, sizeof(FloatType)) == sizeof(FloatType));
        assert(read(fd, &xXDirection, sizeof(FloatType)) == sizeof(FloatType));
        assert(read(fd, &yXDirection, sizeof(FloatType)) == sizeof(FloatType));
        assert(read(fd, &zXDirection, sizeof(FloatType)) == sizeof(FloatType));
    }
    if (inference.size() > 0) {
        assert(read(fd, &x, sizeof(FloatType)) == sizeof(FloatType));
        assert(read(fd, &y, sizeof(FloatType)) == sizeof(FloatType));
        assert(read(fd, &z, sizeof(FloatType)) == sizeof(FloatType));
        assert(read(fd, &xUp, sizeof(FloatType)) == sizeof(FloatType));
        assert(read(fd, &yUp, sizeof(FloatType)) == sizeof(FloatType));
        assert(read(fd, &zUp, sizeof(FloatType)) == sizeof(FloatType));
        assert(read(fd, &xXDirection, sizeof(FloatType)) == sizeof(FloatType));
        assert(read(fd, &yXDirection, sizeof(FloatType)) == sizeof(FloatType));
        assert(read(fd, &zXDirection, sizeof(FloatType)) == sizeof(FloatType));
    }
}

/* loads this from the given file descriptor */
bool RealisticRobotArmSimulationV3::loadState(int fd, unsigned maxSteps) {

    FloatType x; 
    FloatType y; 
    FloatType z;
    FloatType xUp; 
    FloatType yUp; 
    FloatType zUp;
    FloatType xXDirection; 
    FloatType yXDirection; 
    FloatType zXDirection;
    bool setted = true;
    for (unsigned i = 0; i < this->getNumJoints(); i++) {
        FloatType gearState1, gearState2, gearState3;

        assert(read(fd, &gearState1, sizeof(FloatType)) == sizeof(FloatType));
        assert(read(fd, &gearState2, sizeof(FloatType)) == sizeof(FloatType));
        assert(read(fd, &gearState3, sizeof(FloatType)) == sizeof(FloatType));
        setted &= this->setGearStates(i, gearState1, gearState2, gearState3, maxSteps);

        if (inference.size() > 0) {
            assert(read(fd, &gearState1, sizeof(FloatType)) == sizeof(FloatType));
            assert(read(fd, &gearState2, sizeof(FloatType)) == sizeof(FloatType));
            assert(read(fd, &gearState3, sizeof(FloatType)) == sizeof(FloatType));
            setted &= this->setInferredGearStates(i, gearState1, gearState2, gearState3, maxSteps);
        }
    
        assert(read(fd, &x, sizeof(FloatType)) == sizeof(FloatType));
        assert(read(fd, &y, sizeof(FloatType)) == sizeof(FloatType));
        assert(read(fd, &z, sizeof(FloatType)) == sizeof(FloatType));
        assert(read(fd, &xUp, sizeof(FloatType)) == sizeof(FloatType));
        assert(read(fd, &yUp, sizeof(FloatType)) == sizeof(FloatType));
        assert(read(fd, &zUp, sizeof(FloatType)) == sizeof(FloatType));
        assert(read(fd, &xXDirection, sizeof(FloatType)) == sizeof(FloatType));
        assert(read(fd, &yXDirection, sizeof(FloatType)) == sizeof(FloatType));
        assert(read(fd, &zXDirection, sizeof(FloatType)) == sizeof(FloatType));
        this->setPrediction(i, x, y, z, xUp, yUp, zUp, xXDirection, yXDirection, zXDirection);
    }
    if (inference.size() > 0) {
        assert(read(fd, &x, sizeof(FloatType)) == sizeof(FloatType));
        assert(read(fd, &y, sizeof(FloatType)) == sizeof(FloatType));
        assert(read(fd, &z, sizeof(FloatType)) == sizeof(FloatType));
        assert(read(fd, &xUp, sizeof(FloatType)) == sizeof(FloatType));
        assert(read(fd, &yUp, sizeof(FloatType)) == sizeof(FloatType));
        assert(read(fd, &zUp, sizeof(FloatType)) == sizeof(FloatType));
        assert(read(fd, &xXDirection, sizeof(FloatType)) == sizeof(FloatType));
        assert(read(fd, &yXDirection, sizeof(FloatType)) == sizeof(FloatType));
        assert(read(fd, &zXDirection, sizeof(FloatType)) == sizeof(FloatType));
        this->setInferenceTarget(x, y, z, xUp, yUp, zUp, xXDirection, yXDirection, zXDirection);
    }

    for (auto &j: joints)
        j->render();

    for (auto &i: inference)
        i->render();
    
    firstModule->render();
    gripper->render();
    if (inferenceGripper != nullptr)
        inferenceGripper->render();

    return setted;
}

/* returns the last pressed key (if visualized) */
std::string RealisticRobotArmSimulationV3::getKey() {
    if (window == nullptr)
        return "";

    return Simple3DInteractorStyle::SafeDownCast(window->getInteractorStyle())->getLastKey();
}

/* return the inference target inversly transformed by teh given joint */
void RealisticRobotArmSimulationV3::targetFromJoint(unsigned index, FloatType &x, FloatType &y, FloatType &z) {
    
    assert(index < joints.size());
    vtkSmartPointer<vtkTransform> jointInverse = vtkSmartPointer<vtkTransform>::New();
    jointInverse->Identity();

    FloatType xPos, yPos, zPos, xUp, yUp, zUp, xXDirection, yXDirection, zXDirection;
    this->getInferredPosition(index, xPos, yPos, zPos, xUp, yUp, zUp, xXDirection, yXDirection, zXDirection);
    
    double jointTransform[16] = {
        xXDirection, xUp, yXDirection * zUp - zXDirection * yUp, xPos, 
        yXDirection, yUp, zXDirection * xUp - xXDirection * zUp, yPos,
        zXDirection, zUp, xXDirection * yUp - yXDirection * xUp, zPos,
        0, 0, 0, 1
    };

    jointInverse->Concatenate(jointTransform);
    jointInverse->Inverse();

    double in[3], out[3];
    inferenceTarget->getPosition(in[0], in[1], in[2]);
    normalize(in[0], in[1], in[2]);
    jointInverse->TransformPoint(in, out);

    x = out[0];
    y = out[1];
    z = out[2];
}

/* saves this as a screenshot */
void RealisticRobotArmSimulationV3::saveScreenshot(std::string path, int magnification) {

    for (auto &j: joints)
        while (!j->rendered())
            Thread::yield();

    for (auto &i: inference)
        while (!i->rendered())
            Thread::yield();
    
    while (!firstModule->rendered())
        Thread::yield();
    while (!gripper->rendered())
        Thread::yield();
    if (inferenceGripper != nullptr)
        while (!inferenceGripper->rendered())
            Thread::yield();

    window->saveScreenShot(path, magnification);
}

/* returns a vector of all STLObjects of this */
std::vector<std::shared_ptr<STLObject>> RealisticRobotArmSimulationV3::getSTLObjects() {
    std::vector<std::shared_ptr<STLObject>> objects;
    objects.push_back(stand);
    if (firstModule != nullptr) {
        std::vector<std::shared_ptr<STLObject>> o = firstModule->getSTLObjects();
        objects.insert(objects.end(), o.begin(), o.end());
    }
    for (auto &j: joints) {
        std::vector<std::shared_ptr<STLObject>> o = j->getSTLObjects();
        objects.insert(objects.end(), o.begin(), o.end());
    }
    for (auto &i: inference) {
        std::vector<std::shared_ptr<STLObject>> o = i->getSTLObjects();
        objects.insert(objects.end(), o.begin(), o.end());
    }
    if (gripper != nullptr) {
        std::vector<std::shared_ptr<STLObject>> o = gripper->getSTLObjects();
        objects.insert(objects.end(), o.begin(), o.end());
    }
    if (inferenceGripper != nullptr) {
        std::vector<std::shared_ptr<STLObject>> o = inferenceGripper->getSTLObjects();
        objects.insert(objects.end(), o.begin(), o.end());
    }
    return objects;
}

/* returns the angles for the i'th joint */
void RealisticRobotArmSimulationV3::getAngles(
    unsigned index, 
    FloatType &gearState1, 
    FloatType &gearState2, 
    FloatType &gearState3
) {
    getGearStates(index, gearState1, gearState2, gearState3);
    gearState1 = gearState1 * 2 - 1;
    gearState2 = gearState2 * 2 - 1;
    gearState3 = gearState3 * 2 - 1;
}

/* sets the angles for the i'th joint */
void RealisticRobotArmSimulationV3::setAngles(
    unsigned index, 
    FloatType gearState1, 
    FloatType gearState2, 
    FloatType gearState3
) {
    setGearStates(index, (gearState1 + 1) / 2, (gearState2 + 1) / 2, (gearState3 + 1) / 2); 
}

/* sets the inference for the i'th joint */
void RealisticRobotArmSimulationV3::setInferredAngles(
    unsigned index, 
    FloatType gearState1, 
    FloatType gearState2, 
    FloatType gearState3
) {
    setInferredGearStates(index, (gearState1 + 1) / 2, (gearState2 + 1) / 2, (gearState3 + 1) / 2); 
}

/* returns the inferece angles for the i'th joint */
void RealisticRobotArmSimulationV3::getInferredAngles(
    unsigned index, 
    FloatType &gearState1, 
    FloatType &gearState2, 
    FloatType &gearState3
) {
    getInferredGearStates(index, gearState1, gearState2, gearState3);
    gearState1 = gearState1 * 2 - 1;
    gearState2 = gearState2 * 2 - 1;
    gearState3 = gearState3 * 2 - 1;
}

/* returns the camera possition */
void RealisticRobotArmSimulationV3::getCamera(
    double &x,
    double &y,
    double &z,
    double &fx,
    double &fy,
    double &fz,
    double &ux,
    double &uy,
    double &uz
) {
    window->getCamera(x, y, z, fx, fy, fz, ux, uy, uz);
}

/* sets the camera possition */
void RealisticRobotArmSimulationV3::setCamera(
    double x,
    double y,
    double z,
    double fx,
    double fy,
    double fz,
    double ux,
    double uy,
    double uz
) {
    window->setCamera(x, y, z, fx, fy, fz, ux, uy, uz);
}
