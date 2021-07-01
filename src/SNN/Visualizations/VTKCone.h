#ifndef __VTK_CONE_H__ 
#define __VTK_CONE_H__
#include <vtkConeSource.h>
#include "VTKInteractor.h"
#include "VTK3DObject.h"

namespace SNN {

    namespace Visualizations {

        class VTKCone: public VTK3DObject {

            private:

                /* the radius and position of this */
                FloatType r, x, y, z;

                /* the direction of this */
                FloatType dx, dy, dz;

                /* the height of this */
                FloatType height;

                /* the resolution of this */
                FloatType resolution;

                /* the cone source of this */
                vtkSmartPointer<vtkConeSource> coneSource;

                /* inditates that the renderer is blocked */
                bool rendererLocked;

                /* should initalize vtk related stuff within the vtk Thread */
                virtual void initVTK(vtkSmartPointer<vtkPolyDataMapper> mapper) {

                    coneSource = vtkSmartPointer<vtkConeSource>::New();
                    coneSource->SetCenter(x, y, z);
                    coneSource->SetRadius(r);
                    coneSource->SetResolution(resolution);
                    coneSource->SetDirection(dx, dy, dz);
                    coneSource->SetHeight(height);
                    coneSource->Update();

                    mapper->SetInputConnection(coneSource->GetOutputPort());
                }

            public:

                /* constructor */
                VTKCone(
                    FloatType r, 
                    FloatType x, 
                    FloatType y, 
                    FloatType z,
                    FloatType dx, 
                    FloatType dy, 
                    FloatType dz,
                    FloatType height,
                    FloatType resolution
                ): 
                    VTK3DObject(VTK3DType::POLYS), 
                    r(r), x(x), y(y), z(z), 
                    dx(dx), dy(dy), dz(dz), 
                    height(height),
                    resolution(resolution),
                    rendererLocked(false) { }


                /* sets the center of this */
                void setCenter(FloatType x, FloatType y, FloatType z) {
                    while (!this->isInitialized())
                        Thread::yield();
                    
                    if (!rendererLocked) 
                        VTKInteractor::lock();
                    this->x = x;
                    this->y = y;
                    this->z = z;
                    coneSource->SetCenter(x, y, z);
                    if (!rendererLocked) 
                        VTKInteractor::unlock();
                    this->modified();
                }

                /* returns the center of this */
                void getCenter(FloatType &x, FloatType &y, FloatType &z) {
                    x = this->x;
                    y = this->y;
                    z = this->z;
                }

                /* sets the direction of this */
                void setDirection(FloatType dx, FloatType dy, FloatType dz) {
                    while (!this->isInitialized())
                        Thread::yield();
                    
                    if (!rendererLocked) 
                        VTKInteractor::lock();
                    this->dx = dx;
                    this->dy = dy;
                    this->dz = dz;
                    coneSource->SetDirection(dx, dy, dz);
                    if (!rendererLocked) 
                        VTKInteractor::unlock();
                    this->modified();
                }

                /* returns the direction of this */
                void getDirection(FloatType &dx, FloatType &dy , FloatType &dz) {
                    dx = this->dx;
                    dy = this->dy;
                    dz = this->dz;
                }

                /* sets the height of this */
                void setHeight(FloatType height) {
                    while (!this->isInitialized())
                        Thread::yield();
                    
                    if (!rendererLocked) 
                        VTKInteractor::lock();
                    this->height = height;
                    coneSource->SetHeight(height);
                    if (!rendererLocked) 
                        VTKInteractor::unlock();
                    this->modified();
                }

                /* returns the height of this */
                void getHeight(FloatType &height) {
                    height = this->height;
                }

                /* updates this */
                void doUpdate() {
                    coneSource->Update();
                }

                /* sets wether the renderer is allready locked */
                void setRendererLocked(bool locked) {
                    rendererLocked = locked;
                }

        };
    }
}
#endif
