#ifndef __VTK_WINDOW_H__
#define __VTK_WINDOW_H__
#include <vtkSmartPointer.h>
#include <vtkRenderWindowInteractor.h>
#include "Threads.h"
/**
 * base class for a vtk window 
 */
namespace SNN {

    namespace Visualizations {

        /* forward declaration */
        class VTKInteractor;
        class VTKWindow;
        
        class VTKWindow: protected Thread {

            protected:
            
                /* the VTKInteractor of this */
                static vtkSmartPointer<VTKInteractor> interactor;

            private:

                /* indicates wether the vtk thread has ben started or not */
                static volatile bool vtkThreadStarted;

                /* indicates that this was ininitialized */
                volatile bool initialized;

                /* indicates that this was started */
                bool started;

                /* indicates that this was closed */
                bool closed;

                /* should initalize vtk related stuff within the vtk Thread */
                virtual vtkSmartPointer<vtkRenderWindowInteractor> initVTK() = 0;

                /* should update this */
                virtual void doUpdate() = 0;

                /* shoudl close this */
                virtual void doClose() = 0;

                /* starts the vtk event loop */
                virtual void threadFunction(void *);

            public: 

                /* constructor */
                VTKWindow();

                /* virtual destructor for correct polimorphism */
                virtual ~VTKWindow();

                /* starts the window */
                void start();

                /* updates this */
                void update();

                /* closes this */
                void close();

        };

    }

}
#endif /* __VTK_WINDOW_H__ */
