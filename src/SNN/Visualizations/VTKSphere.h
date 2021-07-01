#ifndef __VTK_SPHERE_H__ 
#define __VTK_SPHERE_H__
#include <vtkSphereSource.h>
#include "VTKInteractor.h"
#include "VTK3DObject.h"

namespace SNN {

    namespace Visualizations {

        class VTKSphere: public VTK3DObject {

            private:

                /* the radius and position of this */
                FloatType r, x, y, z;

                /* the resolution of this */
                FloatType resolution;

                /* the sphere source of this */
                vtkSmartPointer<vtkSphereSource> sphereSource;

                /* inditates that the renderer is blocked */
                bool rendererLocked;

                /* should initalize vtk related stuff within the vtk Thread */
                virtual void initVTK(vtkSmartPointer<vtkPolyDataMapper> mapper) {

                    sphereSource = vtkSmartPointer<vtkSphereSource>::New();
                    sphereSource->SetCenter(x, y, z);
                    sphereSource->SetPhiResolution(resolution);
                    sphereSource->SetThetaResolution(resolution);
                    sphereSource->SetRadius(r);
                    sphereSource->Update();

                    mapper->SetInputConnection(sphereSource->GetOutputPort());
                }

            public:

                /* constructor */
                VTKSphere(
                    FloatType r, 
                    FloatType x, 
                    FloatType y, 
                    FloatType z,
                    FloatType resolution = 32
                ): 
                    VTK3DObject(VTK3DType::POLYS), 
                    r(r), x(x), y(y), z(z), 
                    resolution(resolution),
                    rendererLocked(false) { }


                /* sets the center of this */
                void setCenter(FloatType x, FloatType y , FloatType z) {
                    while (!this->isInitialized())
                        Thread::yield();
                    
                    if (!rendererLocked) 
                        VTKInteractor::lock();
                    this->x = x;
                    this->y = y;
                    this->z = z;
                    sphereSource->SetCenter(x, y, z);
                    if (!rendererLocked) 
                        VTKInteractor::unlock();
                    this->modified();
                }

                /* returns the center of this */
                void getCenter(FloatType &x, FloatType &y , FloatType &z) {
                    x = this->x;
                    y = this->y;
                    z = this->z;
                }

                /* returns the radius of this */
                FloatType getRadius() { return r; }

                /* updates this */
                void doUpdate() {
                    sphereSource->Update();
                }

                /* sets wether the renderer is allready locked */
                void setRendererLocked(bool locked) {
                    rendererLocked = locked;
                }

        };
    }
}
#endif
