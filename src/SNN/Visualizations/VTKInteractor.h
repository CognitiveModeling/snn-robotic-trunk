#ifndef __VTK_INTERAKTOR_H__
#define __VTK_INTERAKTOR_H__
#ifdef _WIN32
#include <vtkWin32RenderWindowInteractor.h>
#include <vtkWin32OpenGLRenderWindow.h>
#else
#include <vtkXRenderWindowInteractor.h>
#endif
#include <VTKWindow.h>
#include <vector>
#include "Threads.h"
/**
 * Interactor to hande vtk events 
 */
namespace SNN {
    
    namespace Visualizations {
        
#ifdef _WIN32
        class VTKInteractor : public vtkWin32RenderWindowInteractor {
#else
        class VTKInteractor : public vtkXRenderWindowInteractor {
#endif

            private:

                /* the vtk windows of this */
                std::vector<VTKWindow *> windows;

                /* syncronization mutex */
                static Mutex mutex;

                /* the thread id of the vtk thread */
                static uint64_t VTKThreadId;

                /* indicates that this should render objects */
                static bool render;

                /* thread if of the thread that holds the lock */
                static uint64_t blockingThreadID;

                /* lock counter */
                static unsigned lockCounter;

            public:
                
                /* creats a new interactor */
                static VTKInteractor *New();

                /* vtk event loop */
                virtual void StartEventLoop();

                /* registers the given window */
                void addWindow(VTKWindow *);

                /* unregisters the given window */
                void removeWindow(VTKWindow *);

                /* locks the vtk tread */
                static void lock();

                /* unlocks the vtk thread */
                static void unlock();

                /* enables rendering */
                static void enableRendering();

                /* disables the rendering */
                static void disableRendering();

        };
    }
}
#endif /* __VTK_INTERAKTOR_H__ */
