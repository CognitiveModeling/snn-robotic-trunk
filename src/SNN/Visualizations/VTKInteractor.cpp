#include "VTKInteractor.h"
#include "VTKWindow.h"
#include <algorithm>
using namespace SNN;
using namespace Visualizations;

/* syncronization mutex */
Mutex VTKInteractor::mutex;

/* the thread id of the vtk thread */
uint64_t VTKInteractor::VTKThreadId = 0;

/* indicates that this should render objects */
bool VTKInteractor::render = true;

/* thread if of the thread that holds the lock */
uint64_t VTKInteractor::blockingThreadID = -1;

/* lock counter */
unsigned VTKInteractor::lockCounter = 0;

/* creats a new VTKInteractor */
VTKInteractor *VTKInteractor::New() {
    return new VTKInteractor();
}


/* the main event loop which processes events send tho this thread */
void VTKInteractor::StartEventLoop() {

    VTKThreadId = Thread::id();

#ifdef _WIN32
    this->StartedMessageLoop = 1;

    do {
        
        /* process windows messages */
        mutex.lock();
        if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE|PM_NOYIELD)) {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        } 
        mutex.unlock();
#else
    this->BreakLoopFlag = 0;
    do {
        /* process XLib messages */
        mutex.lock();
        if (XtAppPending(vtkXRenderWindowInteractor::App) & XtIMXEvent) {
            XEvent event;
            XtAppNextEvent(vtkXRenderWindowInteractor::App, &event);
            XtDispatchEvent(&event);
        } 
        mutex.unlock();
#endif


        /* update windows */
        if (render) {
            mutex.lock();
            for (auto &w: windows) {
                w->update();
            }
            mutex.unlock();
        }

#ifdef _WIN32
    } while (msg.message != WM_QUIT);
#else
    } while (this->BreakLoopFlag == 0);
#endif
}


/* registers the given window */
void VTKInteractor::addWindow(VTKWindow *window) {
    VTKInteractor::lock();
    if (std::find(this->windows.begin(), this->windows.end(), window) == this->windows.end())
        windows.push_back(window);
    VTKInteractor::unlock();
}

/* unregisters the given window */
void VTKInteractor::removeWindow(VTKWindow *window) {
    if (std::find(windows.begin(), windows.end(), window) != windows.end())
        windows.erase(std::find(windows.begin(), windows.end(), window));

    if (windows.size() == 0)
        exit(EXIT_SUCCESS);
}

/* locks the vtk tread */
void VTKInteractor::lock() {
    if (Thread::id() != VTKThreadId && blockingThreadID != Thread::id()) {
        blockingThreadID = Thread::id();
        mutex.lock();
    }
    if (blockingThreadID == Thread::id())
        lockCounter++;
}

/* unlocks the vtk thread */
void VTKInteractor::unlock() {
    if (Thread::id() != VTKThreadId) {
        if (blockingThreadID == Thread::id()) {
            lockCounter--;
        }
        if (blockingThreadID == Thread::id() && lockCounter == 0) {
            mutex.unlock();
            blockingThreadID = -1;
        }
    }
}

/* enables rendering */
void VTKInteractor::enableRendering() {
    render = true;
}

/* disables the rendering */
void VTKInteractor::disableRendering() {
    render = false;
}
