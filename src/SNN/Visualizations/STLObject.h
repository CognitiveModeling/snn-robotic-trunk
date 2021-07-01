#ifndef __STL_OBJECT_H__
#define __STL_OBJECT_H__
#include <vtkSTLReader.h>
#include <vtkAppendPolyData.h>
#include <vtkCleanPolyData.h>
#include "VTK3DObject.h"

namespace SNN {

    namespace Visualizations {

        class STLObject: public VTK3DObject {
          
            private:

                /* path to the stl file */
                std::string stlFile;

                /* wether to translate / rotate befor or after global transformation */
                bool translateBefor, rotateBefor;

                /* stl reader */
                vtkSmartPointer<vtkSTLReader> stlReader;

                /* translation adn rotations */
                vtkSmartPointer<vtkTransform> globalTransform;
                vtkSmartPointer<vtkTransform> localTransform;
                vtkSmartPointer<vtkTransform> globalTransformStart;
                vtkSmartPointer<vtkTransformPolyDataFilter> globalTransformFilter;

            public:

                /* constructor */
                STLObject(
                    std::string stlFile
                ) : 
                    VTK3DObject(VTK3DType::POLYS), 
                    stlFile(stlFile) { 

                    localTransform = vtkSmartPointer<vtkTransform>::New();
                    globalTransform = vtkSmartPointer<vtkTransform>::New();
                    globalTransformStart = vtkSmartPointer<vtkTransform>::New();
                }

                /* should initalize vtk related stuff within the vtk Thread */
                virtual void initVTK(vtkSmartPointer<vtkPolyDataMapper> mapper) { 
                        
                    if (stlFile != "") {
                        stlReader = vtkSmartPointer<vtkSTLReader>::New();
                        stlReader->SetFileName(stlFile.c_str());
                        stlReader->Update();

                        globalTransformFilter = vtkSmartPointer<vtkTransformPolyDataFilter>::New();
                        globalTransformFilter->SetInputData(stlReader->GetOutput());
                        globalTransformFilter->SetTransform(globalTransform);
                        globalTransformFilter->Update();
                        mapper->SetInputData(globalTransformFilter->GetOutput());
                    }
                }

                /* resets the global transformation */
                void globalReset() {
                    VTKInteractor::lock();
                    globalTransformStart->Identity();
                    VTKInteractor::unlock();
                }

                /* adds a rotation to this */
                void addGlobalRotation(FloatType angle, FloatType x, FloatType y, FloatType z) {
                    VTKInteractor::lock();
                    globalTransformStart->RotateWXYZ(angle, x, y, z);
                    VTKInteractor::unlock();
                }

                /* adds a translation to this */
                void addGlobalTranslation(FloatType x, FloatType y, FloatType z) {
                    VTKInteractor::lock();
                    globalTransformStart->Translate(x, y, z);
                    VTKInteractor::unlock();
                }

                /* resets the local transformation */
                void reset() {
                    VTKInteractor::lock();
                    localTransform->Identity();
                    VTKInteractor::unlock();
                }

                /* adds a rotation to this */
                void addRotation(FloatType angle, FloatType x, FloatType y, FloatType z) {
                    VTKInteractor::lock();
                    localTransform->RotateWXYZ(angle, x, y, z);
                    VTKInteractor::unlock();
                }

                /* adds a translation to this */
                void addTranslation(FloatType x, FloatType y, FloatType z) {
                    VTKInteractor::lock();
                    localTransform->Translate(x, y, z);
                    VTKInteractor::unlock();
                }

                /* shoud update this */
                virtual void doUpdate() { 
                    if (stlFile != "") {
                        VTKInteractor::lock();
                        globalTransformFilter->Update();
                        VTKInteractor::unlock();
                    }
                }

                /* adds a transformation to this */
                void transform(vtkSmartPointer<vtkTransform> transform = NULL) {
                    //while (this->changed())
                    //    Thread::yield();

                    VTKInteractor::lock();
                    globalTransform->Identity();
                    globalTransform->Concatenate(globalTransformStart);
                    if (transform != NULL)
                        globalTransform->Concatenate(transform);
                    globalTransform->Concatenate(localTransform);
                    VTKInteractor::unlock();

                    this->modified();
                }

                /* returns the center of this */
                void getCenter(FloatType &x, FloatType &y, FloatType &z) {
                    double center[3];
                    VTKInteractor::lock();
                    globalTransform->GetPosition(center);
                    VTKInteractor::unlock();

                    x = center[0];
                    y = center[1];
                    z = center[2];
                }

                /* returns the orientation of this */
                void getOrientation(FloatType &angle, FloatType &x, FloatType &y, FloatType &z) {
                    double wxyz[4];
                    VTKInteractor::lock();
                    globalTransform->GetOrientationWXYZ(wxyz);
                    VTKInteractor::unlock();

                    angle = wxyz[0];
                    x = wxyz[1];
                    y = wxyz[2];
                    z = wxyz[3];
                }
                    
                    
        };
    }
}
#endif /* __STL_OBJECT_H__ */
