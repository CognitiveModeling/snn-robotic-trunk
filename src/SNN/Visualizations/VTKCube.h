#ifndef __VTK_CUBE_H__ 
#define __VTK_CUBE_H__
#include <vtkCubeSource.h>
#include "VTKInteractor.h"
#include "VTK3DObject.h"

namespace SNN {

    namespace Visualizations {

        class VTKCube: public VTK3DObject {

            private:

                /* the length and position of this */
                FloatType lX, lY, lZ, x, y, z;

                /* the sphere source of this */
                vtkSmartPointer<vtkCubeSource> cubeSource;

                /* inditates that the renderer is blocked */
                bool rendererLocked;

                /* should initalize vtk related stuff within the vtk Thread */
                virtual void initVTK(vtkSmartPointer<vtkPolyDataMapper> mapper) {

                    cubeSource = vtkSmartPointer<vtkCubeSource>::New();
                    cubeSource->SetCenter(x, y, z);
                    cubeSource->SetXLength(lX);
                    cubeSource->SetYLength(lY);
                    cubeSource->SetZLength(lZ);
                    cubeSource->Update();

                    mapper->SetInputConnection(cubeSource->GetOutputPort());
                }

            public:

                /* constructor */
                VTKCube(
                    FloatType lX, 
                    FloatType lY, 
                    FloatType lZ, 
                    FloatType x, 
                    FloatType y, 
                    FloatType z
                ): 
                    VTK3DObject(VTK3DType::POLYS), 
                    lX(lX), lY(lY), lZ(lZ), x(x), y(y), z(z), 
                    rendererLocked(false) { }


                /* sets the center of this */
                void setCenter(FloatType x, FloatType y , FloatType z, FloatType lX, FloatType lY, FloatType lZ) {
                    while (!this->isInitialized())
                        Thread::yield();
                    
                    if (!rendererLocked) 
                        VTKInteractor::lock();
                    this->x = x;
                    this->y = y;
                    this->z = z;
                    cubeSource->SetCenter(x, y, z);
                    cubeSource->SetXLength(lX);
                    cubeSource->SetYLength(lY);
                    cubeSource->SetZLength(lZ);
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

                /* returns wether the given sphere is inside this */
                bool intersectSphere(FloatType sx, FloatType sy, FloatType sz, FloatType r) {
                    FloatType minX = x - lX/2, minY = y - lY/2, minZ = z - lZ/2;
                    FloatType maxX = x + lX/2, maxY = y + lY/2, maxZ = z + lZ/2;

                    if ((minX <= sx && sx <= maxX) && 
                        (minY <= sy && sy <= maxY) &&
                        (minZ <= sz && sz <= maxZ))
                        return true;

                    FloatType tx = vmax(minX, vmin(sx, maxX));
                    FloatType ty = vmax(minY, vmin(sy, maxY));
                    FloatType tz = vmax(minZ, vmin(sz, maxZ));

                    FloatType distance = sqrt(
                        pow(tx - sx, 2) +
                        pow(ty - sy, 2) +
                        pow(tz - sz, 2)
                    );

                    return distance <= r;
                }

                /* returns the boundling bo of this */
                void getAABB(
                    FloatType &minX, 
                    FloatType &minY, 
                    FloatType &minZ, 
                    FloatType &maxX, 
                    FloatType &maxY, 
                    FloatType &maxZ
                ) {
                    minX = x - lX/2, minY = y - lY/2, minZ = z - lZ/2;
                    maxX = x + lX/2, maxY = y + lY/2, maxZ = z + lZ/2;
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
