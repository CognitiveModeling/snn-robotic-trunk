#ifndef __FULL_ROBOT_ARM_PREDICTION_H__ 
#define __FULL_ROBOT_ARM_PREDICTION_H__
#include <vtkCubeSource.h>
#include <vtkSphereSource.h>
#include <vtkTransformPolyDataFilter.h>
#include <vtkPolyDataMapper.h>
#include <vtkPolyData.h>
#include <vtkArrowSource.h>
#include <vtkAppendPolyData.h>
#include <vtkTransform.h>
#include "VTK3DObject.h"
#include "Threads.h"
#include "VTKInteractor.h"

namespace SNN {

    namespace Visualizations {

        class RobotArmPrediction: public VTK3DObject {

            private:

                /* the radius position and directions of this */
                FloatType r, x, y, z, xUp, yUp, zUp, xXDirection, yXDirection, zXDirection;

                /* the cube / sphere source of this */
                vtkSmartPointer<vtkCubeSource> cubeSource;
                vtkSmartPointer<vtkSphereSource> sphereSource;

                /* transformations for the cube of this */
                vtkSmartPointer<vtkTransform> cubeTransform;
                vtkSmartPointer<vtkTransformPolyDataFilter> cubeTransformFilter;

                /* the arrows of this for up and x direction */
                vtkSmartPointer<vtkArrowSource> upVector;
                vtkSmartPointer<vtkArrowSource> xDirection;

                /* transformations for the arrows of this */
                vtkSmartPointer<vtkTransform> xDirectionTransform;
                vtkSmartPointer<vtkTransformPolyDataFilter> xDirectionTransformFilter;

                /* transformation filter for the arrows of this */
                vtkSmartPointer<vtkTransform> upVectorTransform;
                vtkSmartPointer<vtkTransformPolyDataFilter> upVectorTransformFilter;

                /* append filter to combine cube and arrows */
                vtkSmartPointer<vtkAppendPolyData> appendFilter;

                /* inditates that the renderer is blocked */
                bool rendererLocked;

                /* wether to visualize directions */
                bool visualizeDirections; 

                /* pi constant */
                const FloatType PI;

                /* should initalize vtk related stuff within the vtk Thread */
                virtual void initVTK(vtkSmartPointer<vtkPolyDataMapper> mapper) {

                    cubeSource = vtkSmartPointer<vtkCubeSource>::New();
                    cubeSource->SetCenter(x, y, z);
                    cubeSource->SetXLength(r*2);
                    cubeSource->SetYLength(1);
                    cubeSource->SetZLength(r*2);
                    cubeSource->Update();

                    sphereSource = vtkSmartPointer<vtkSphereSource>::New();
                    sphereSource->SetCenter(x, y, z);
                    sphereSource->SetRadius(r/2.5);
                    sphereSource->Update();

                    cubeTransform = vtkSmartPointer<vtkTransform>::New();
                    cubeTransformFilter = vtkSmartPointer<vtkTransformPolyDataFilter>::New();

                    upVector = vtkSmartPointer<vtkArrowSource>::New();
                    upVector->Update();

                    xDirection = vtkSmartPointer<vtkArrowSource>::New();
                    xDirection->Update();

                    xDirectionTransform = vtkSmartPointer<vtkTransform>::New();
                    xDirectionTransformFilter = vtkSmartPointer<vtkTransformPolyDataFilter>::New();

                    upVectorTransform = vtkSmartPointer<vtkTransform>::New();
                    upVectorTransformFilter = vtkSmartPointer<vtkTransformPolyDataFilter>::New();

                    xDirectionTransform->Scale(2*r,2*r,2*r);
                    upVectorTransform->Scale(2*r,2*r,2*r);
                    upVectorTransform->RotateZ(90);

                    xDirectionTransform->Update();
                    upVectorTransform->Update();
                    cubeTransform->Update();

                    xDirectionTransformFilter->SetInputData(xDirection->GetOutput());
                    xDirectionTransformFilter->SetTransform(xDirectionTransform);
                    xDirectionTransformFilter->Update();

                    upVectorTransformFilter->SetInputData(upVector->GetOutput());
                    upVectorTransformFilter->SetTransform(upVectorTransform);
                    upVectorTransformFilter->Update();

                    cubeTransformFilter->SetInputData(cubeSource->GetOutput());
                    cubeTransformFilter->SetTransform(cubeTransform);
                    cubeTransformFilter->Update();

                    appendFilter = vtkSmartPointer<vtkAppendPolyData>::New();
                    appendFilter->AddInputConnection(xDirectionTransformFilter->GetOutputPort());
                    appendFilter->AddInputConnection(upVectorTransformFilter->GetOutputPort());
                    appendFilter->AddInputConnection(cubeTransformFilter->GetOutputPort());
                    appendFilter->Update();

                    if (visualizeDirections)
                        mapper->SetInputConnection(appendFilter->GetOutputPort());
                    else
                        mapper->SetInputConnection(sphereSource->GetOutputPort());
                }

            public:

                /* constructor */
                RobotArmPrediction(FloatType r, bool visualizeDirections):
                    VTK3DObject(VTK3DType::POLYS), r(r), x(0), y(0), z(0), xUp(0), yUp(1), zUp(0),
                    xXDirection(1), yXDirection(0), zXDirection(0), rendererLocked(false), 
                    visualizeDirections(visualizeDirections), PI(3.141592653589793) { }


                /* sets the position of this of this */
                void setPosition(
                    FloatType x, 
                    FloatType y, 
                    FloatType z
                ) {
                    while (!this->isInitialized())
                        Thread::yield();
                    
                    if (!rendererLocked) 
                        VTKInteractor::lock();
                    this->x = x;
                    this->y = y;
                    this->z = z;

                    sphereSource->SetCenter(x, y, z);
                    cubeTransform->Identity();
                    cubeTransform->Translate(x, y, z);

                    if (!rendererLocked) 
                        VTKInteractor::unlock();
                    this->modified();
                }

                /* sets the directions of this of this */
                void setDirections(
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
                    while (!this->isInitialized())
                        Thread::yield();

                    FloatType lengthUp = sqrt(pow(xUp, 2) + pow(yUp, 2) + pow(zUp, 2)); 
                    xUp /= lengthUp;
                    yUp /= lengthUp; 
                    zUp /= lengthUp; 
                    
                    FloatType lengthXDirection = sqrt(pow(xXDirection, 2) + pow(yXDirection, 2) + pow(zXDirection, 2));
                    xXDirection /= lengthXDirection;
                    yXDirection /= lengthXDirection;
                    zXDirection /= lengthXDirection;
                    
                    if (!rendererLocked) 
                        VTKInteractor::lock();
                    this->xUp = xUp; 
                    this->yUp = yUp; 
                    this->zUp = zUp;
                    this->xXDirection = xXDirection; 
                    this->yXDirection = yXDirection; 
                    this->zXDirection = zXDirection;
                    
                    double rotation[16] = {
                        xXDirection, xUp, yXDirection * zUp - zXDirection * yUp, 0, 
                        yXDirection, yUp, zXDirection * xUp - xXDirection * zUp, 0,
                        zXDirection, zUp, xXDirection * yUp - yXDirection * xUp, 0,
                        0, 0, 0, 1
                    };
                    cubeTransform->Concatenate(rotation);

                    xDirectionTransform->Identity();
                    xDirectionTransform->Translate(x, y, z);
                    xDirectionTransform->Scale(2*r,2*r,2*r);
                    xDirectionTransform->Concatenate(rotation);

                    upVectorTransform->Identity();
                    upVectorTransform->Translate(x, y, z);
                    upVectorTransform->Scale(2*r,2*r,2*r);
                    upVectorTransform->Concatenate(rotation);
                    upVectorTransform->RotateZ(90);

                    if (!rendererLocked) 
                        VTKInteractor::unlock();
                    this->modified();
                }

                /* returns the position and orientation of this of this */
                void getPosition(
                    FloatType &x, 
                    FloatType &y, 
                    FloatType &z
                ) {
                    x = this->x;
                    y = this->y;
                    z = this->z;
                }
                void getUpVector(
                    FloatType &xUp, 
                    FloatType &yUp, 
                    FloatType &zUp
                ) {
                    xUp = this->xUp;               
                    yUp = this->yUp; 
                    zUp = this->zUp;
                }
                void getXDirection(
                    FloatType &xXDirection, 
                    FloatType &yXDirection, 
                    FloatType &zXDirection
                ) {
                    xXDirection = this->xXDirection; 
                    yXDirection = this->yXDirection; 
                    zXDirection = this->zXDirection;
                }

                /* updates this */
                void doUpdate() {
                    cubeSource->Update();
                }

                /* sets wether the renderer is allready locked */
                void setRendererLocked(bool locked) {
                    rendererLocked = locked;
                }

        };

    }
}
#endif
