#include <vtkCallbackCommand.h>
#include "VTKWindow.h"
#include "VTKInteractor.h"
#include <assert.h>
using namespace SNN;
using namespace Visualizations;

/* indicates wether the vtk thread has ben started or not */
volatile bool VTKWindow::vtkThreadStarted = false;

/* the VTKInteractor of this */
vtkSmartPointer<VTKInteractor> VTKWindow::interactor = NULL;

/* constructor */
VTKWindow::VTKWindow(): initialized(false), started(false), closed(false) { }

/* starts the window */
void VTKWindow::start() {

    assert(!started);

    if (!started) {
        if (!vtkThreadStarted)
            this->run();

        /* bussy wating loop */
        while (!vtkThreadStarted)
            Thread::yield();

        VTKWindow::interactor->addWindow(this);

        /* bussy wating loop */
        while (!initialized)
            Thread::yield();

        started = true;
    }
}

/* closes this */
void VTKWindow::close() {
    closed = true;
}

/* virtual destructor for correct polimorphism */
VTKWindow::~VTKWindow() {
}

/* function to delet a window upon user exit */
void remove_window(vtkObject*, unsigned long,  void *window, void *) {
    static_cast<VTKWindow *>(window)->close();
}

/* register remove callback */
void register_callback(vtkSmartPointer<vtkRenderWindowInteractor> interactor, VTKWindow *window) {

    /* create remove callback */
    vtkSmartPointer<vtkCallbackCommand> removeCallback = vtkSmartPointer<vtkCallbackCommand>::New();
    removeCallback->SetCallback(remove_window);
    removeCallback->SetClientData(window);

	interactor->AddObserver(vtkCommand::ExitEvent, removeCallback);
}

/* starts the vtk event loop */
void VTKWindow::threadFunction(void *) {
    VTKWindow::interactor = dynamic_cast<VTKInteractor *>(this->initVTK().Get());
    VTKWindow::interactor->Initialize();
    register_callback(VTKWindow::interactor, this);
    initialized = true;
    vtkThreadStarted = true;
    VTKWindow::interactor->Start();
}

/* updates this */
void VTKWindow::update() {
    if (!this->initialized) {
        vtkSmartPointer<vtkRenderWindowInteractor> interactor = this->initVTK();
        register_callback(interactor, this);
        this->initialized = true;
    }
    this->doUpdate();

    if (closed) {
        VTKWindow::interactor->removeWindow(this);
        this->doClose();
    }
}
