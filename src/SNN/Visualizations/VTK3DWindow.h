#ifndef __VTK_3D_WINDOW_H__
#define __VTK_3D_WINDOW_H__ 
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <memory>
#include <vector>
#include "VTKWindow.h"
namespace SNN {

    namespace Visualizations {

        /* forward declaration */
        class VTK3DObject;

        /* generator class for a vtkContextInteractorStyle */
        class VTK3DInteractorStyleGenerator {
            
            public:

                /* virtual constructor for correct polimorphism */
                virtual ~VTK3DInteractorStyleGenerator() { }

                /* should create the vtkContextInteractorStyle */
                virtual vtkSmartPointer<vtkInteractorStyleTrackballCamera> generate() {
                    return vtkSmartPointer<vtkInteractorStyleTrackballCamera>::New();
                }

        };

        class VTK3DWindow: public VTKWindow {

            private: 

                /* the visualized plot objects of this */
                std::vector<std::shared_ptr<VTK3DObject>> objects;

                /* the main Renderer of this */
                vtkSmartPointer<vtkRenderer> renderer;

                /* the main render window of this */
                vtkSmartPointer<vtkRenderWindow> renderWindow;

                /* the interactor of this */
                vtkSmartPointer<vtkRenderWindowInteractor> interactor;

                /* the interactor stlye of this */
                vtkSmartPointer<vtkInteractorStyleTrackballCamera> interactorStyle;

                /* the interactor style generator */
                std::shared_ptr<VTK3DInteractorStyleGenerator> styleGenerator;

                /* indicates that this has changed */
                bool changed;

                /* indicate that an object has ben added */
                bool newObject;
                
                /* screenshot variables */
                std::string screenShotPath;
                int screenShotMagification;

                /* camera variables */
                double x, y, z, fx, fy, fz, ux, uy, uz;
                bool setcamera;

                /* should initalize vtk related stuff within the vtk Thread */
                virtual vtkSmartPointer<vtkRenderWindowInteractor> initVTK();

                /* should update this */
                virtual void doUpdate();

                /* shoudl close this */
                virtual void doClose();
            
            public: 

                /* constructor */
                VTK3DWindow(
                    std::shared_ptr<VTK3DInteractorStyleGenerator> styleGenerator = 
                        std::make_shared<VTK3DInteractorStyleGenerator>()
                );

                /* virtual destructor for correct polimorphism */
                virtual ~VTK3DWindow();

                /* updates this */
                void update();

                /* adds the given VTK3DObject */
                void add(std::shared_ptr<VTK3DObject>);

                /* removes the given VTK3DObject */
                void remove(std::shared_ptr<VTK3DObject>);

                /* returns the interactor style of this */
                vtkSmartPointer<vtkInteractorStyleTrackballCamera> getInteractorStyle() {
                    return interactorStyle;
                }

                /* saves a screnshot in the gifen magnification */
                void saveScreenShot(std::string path, int magnification = 1);

                /* returns the camera possition */
                void getCamera(
                    double &x,
                    double &y,
                    double &z,
                    double &fx,
                    double &fy,
                    double &fz,
                    double &ux,
                    double &uy,
                    double &uz
                );

                /* sets the camera possition */
                void setCamera(
                    double x,
                    double y,
                    double z,
                    double fx,
                    double fy,
                    double fz,
                    double ux,
                    double uy,
                    double uz
                );

        };

    }

}


#endif /* __VTK_3D_WINDOW_H__ */
