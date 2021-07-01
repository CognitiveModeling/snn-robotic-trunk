#ifndef __BASIC_ROBOT_ARM_MODULE_H__
#define __BASIC_ROBOT_ARM_MODULE_H__
#include <vtkSmartPointer.h>
#include <vtkTransform.h>
#include <vtkTransformPolyDataFilter.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include "VTK3DObject.h"
#include "VTKInteractor.h"
#include "Threads.h"

namespace SNN {

    namespace Visualizations {

        class BasicRobotArmModule: public VTK3DObject {

            private:

                /* the main transformation filter of this */
                vtkSmartPointer<vtkTransform> transform;
                vtkSmartPointer<vtkTransformPolyDataFilter> transformFilter;

                /* should return the poly data element of this module */
                virtual vtkSmartPointer<vtkPolyData> getPolyData() = 0;

                /* the current angles and heigh of this */
                FloatType angleX, angleY, height;

                /* stops this from beeing rendered */
                bool stopRenering;

                /* indicates that this is beeing visualized */
                bool visualized;

                /* dan do additinal processing steps on move */
                virtual void doMove(
                    FloatType height, 
                    FloatType angleX, 
                    FloatType angleY, 
                    vtkSmartPointer<vtkTransform> base) { 
                    
                    (void) height;
                    (void) angleX;
                    (void) angleY;
                    (void) base;
                }

            public:

                /* constructor */
                BasicRobotArmModule() : 
                VTK3DObject(VTK3DType::POLYS), angleX(0), angleY(0), height(0), stopRenering(false), visualized(true) { }

                /* returns the (relative) height and angles of this */
                FloatType getHeight() { return height; }
                FloatType getXAngle() { return angleX; }
                FloatType getZAngle() { return angleY; }

                /* should initalize vtk related stuff within the vtk Thread */
                virtual void initVTK(vtkSmartPointer<vtkPolyDataMapper> mapper) {

                    transform = vtkSmartPointer<vtkTransform>::New();
                    transformFilter = vtkSmartPointer<vtkTransformPolyDataFilter>::New();

                    transformFilter->SetInputData(this->getPolyData());
                    transformFilter->SetTransform(transform);
                    transformFilter->Update();

                    mapper->SetInputConnection(transformFilter->GetOutputPort());
                }

                void move(
                    FloatType height, 
                    FloatType angleX, 
                    FloatType angleY, 
                    vtkSmartPointer<vtkTransform> base = NULL
                ) {
                    while (!this->isInitialized() && visualized)
                        Thread::yield();
                    
                    if (!stopRenering && visualized)
                        VTKInteractor::lock();

                    this->height = height;
                    this->angleX = angleX;
                    this->angleY = angleY;
                    transform->Identity();

                    this->doMove(height, angleX, angleY, base);

                    if (base != NULL)
                        transform->Concatenate(base);
                    transform->Translate(0, height, 0);
                    transform->RotateWXYZ(angleY, 0, 0, 1);
                    transform->RotateWXYZ(angleX, 1, 0, 0);

                    if (!stopRenering && visualized) {
                        VTKInteractor::unlock();
                        this->modified();
                    }
                }

                vtkSmartPointer<vtkTransform> getTransform() { return transform; }

                /* returns the center of this */
                void getCenter(FloatType &x, FloatType &y, FloatType &z) {
                    while (!this->isInitialized() && visualized)
                        Thread::yield();

                    double center[3];
                    if (!stopRenering && visualized)
                        VTKInteractor::lock();

                    transform->GetPosition(center);

                    if (!stopRenering && visualized)
                        VTKInteractor::unlock();

                    x = center[0];
                    y = center[1];
                    z = center[2];
                }

                /* returns the up vector of this */
                void getUpVector(FloatType &x, FloatType &y, FloatType &z) {
                    while (!this->isInitialized() && visualized)
                        Thread::yield();

                    double up[3] = { 0, 1, 0 };
                    if (!stopRenering && visualized)
                        VTKInteractor::lock();

                    transform->TransformVector(up, up);

                    if (!stopRenering && visualized)
                        VTKInteractor::unlock();

                    x = up[0];
                    y = up[1];
                    z = up[2];
                }

                /* returns the x direction vector of this */
                void getXDirection(FloatType &x, FloatType &y, FloatType &z) {
                    while (!this->isInitialized() && visualized)
                        Thread::yield();

                    double xDirection[3] = { 1, 0, 0 };
                    if (!stopRenering && visualized)
                        VTKInteractor::lock();

                    transform->TransformVector(xDirection, xDirection);

                    if (!stopRenering && visualized)
                        VTKInteractor::unlock();

                    x = xDirection[0];
                    y = xDirection[1];
                    z = xDirection[2];
                }

                /* deactivate rendering for faster math */
                void deactivateRenderer() {
                    
                    if (visualized) {
                        stopRenering = true;

                        while (!this->isInitialized() || this->changed())
                            Thread::yield();
                    }
                }


                /* activate rendering */
                void activateRenderer() {
                    if (visualized) {
                        stopRenering = false;
                        this->modified();
                    }
                }

                /* sets the visualized status of this */
                void setVisualized(bool visualized) {
                    this->visualized = visualized;
                }
        };
    }
}
#endif /* __BASIC_ROBOT_ARM_MODULE_H__ */
