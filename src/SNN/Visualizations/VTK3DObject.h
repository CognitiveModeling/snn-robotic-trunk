#ifndef __VTK_3D_OBJECT_H__
#define __VTK_3D_OBJECT_H__
#include <vtkSmartPointer.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkContextInteractorStyle.h>
#include <vtkActor.h>
#include <vector>
#include <string>
namespace SNN {

    namespace Visualizations {

        /* the different visualization types */
        enum class VTK3DType { POINTS, LINES, POLYS };
        
        class VTK3DObject {

            private:

                /* the object mapper */
                vtkSmartPointer<vtkPolyDataMapper> mapper;

                /* the actor of this */
                vtkSmartPointer<vtkActor> actor;

                /* indicates that the data of this has changed */
                bool dataChanged;

                /* indicates that this was initialized */
                bool initialized;

                /* the type of this */
                VTK3DType type;

                /* shoud update this */
                virtual void doUpdate() { }

                /* should initalize vtk related stuff within the vtk Thread */
                virtual void initVTK(vtkSmartPointer<vtkPolyDataMapper>) = 0;

            public:

                /* sets dataChanged to true */
                void modified();

                /* returns wether this was initialized */
                bool isInitialized() { return initialized; }

                /* constructor */
                VTK3DObject(VTK3DType type);

                /* indicates wether tha data of this has changed */
                bool changed();

                /* updates this */
                void update();

                /* sets the color of this */
                void setColor(
                    unsigned char r,
                    unsigned char g,
                    unsigned char b,
                    unsigned char a
                );

                /* returns the actor of this */
                vtkSmartPointer<vtkActor> getActor() { return actor; }
        };

    }

}
#endif /* __VTK_3D_OBJECT_H__ */
