#include <vtkContextScene.h>
#include <vtkAxis.h>
#include "VTK3DWindow.h"
#include "VTK3DObject.h"
#include "VTKInteractor.h"
#include <algorithm>
#include <vtkPNGWriter.h>
#include <vtkWindowToImageFilter.h>
#include <vtkCamera.h>
using namespace SNN;
using namespace Visualizations;

using std::string;

/* constructor */
VTK3DWindow::VTK3DWindow(std::shared_ptr<VTK3DInteractorStyleGenerator> styleGenerator) : 
    styleGenerator(styleGenerator), changed(false), newObject(false), setcamera(false) { }

/* destructor */
VTK3DWindow::~VTK3DWindow() { }


/* shoudl close this */
void VTK3DWindow::doClose() {

    objects.clear();

    interactorStyle->SetEnabled(false);
    interactor->Disable();
    interactor->EnableRenderOff();
    interactor->SetRenderWindow(NULL);
    interactor->SetInteractorStyle(NULL);

    renderWindow->RemoveRenderer(renderer);
    renderWindow->SetInteractor(NULL);
    renderWindow    = NULL; 
    renderer        = NULL;
    interactor      = NULL;
    interactorStyle = NULL;
}

/* should initalize vtk related stuff within the vtk Thread */
vtkSmartPointer<vtkRenderWindowInteractor> VTK3DWindow::initVTK() {

    if (VTKWindow::interactor == NULL) 
        interactor  = vtkSmartPointer<VTKInteractor>::New();
    else 
        interactor  = VTKWindow::interactor;
    interactorStyle = styleGenerator->generate();
    renderer        = vtkSmartPointer<vtkRenderer>::New();
    renderWindow    = vtkSmartPointer<vtkRenderWindow>::New();

  	renderWindow->AddRenderer(renderer);
  	interactor->SetRenderWindow(renderWindow);

  	//renderer->SetBackground(0.9803921568627451,0.9803921568627451,0.9803921568627451); 
  	renderer->SetBackground(1,1,1);
  	//renderer->SetBackground(.1,.2,.3); 

  	renderWindow->SetSize(900,900);
  	renderWindow->SetWindowName("Window Name");
  	renderWindow->Render();
  
  	interactor->SetInteractorStyle(interactorStyle);
  
    return interactor;
}

/* should update this */
void VTK3DWindow::doUpdate() {

    bool render = false;
    for (auto &o: objects) {
        if (o->changed()) {
            o->update();
            render = true;
        }
    }
    
    if (render || changed) {
        if (newObject)
            renderer->ResetCamera();

        if (setcamera) {
            vtkSmartPointer<vtkCamera> camera = vtkSmartPointer<vtkCamera>::New();
            camera->SetPosition(x, y, z);
            camera->SetFocalPoint(fx, fy, fz);
            camera->SetViewUp(ux, uy, uz);
            renderer->SetActiveCamera(camera);
            renderer->ResetCameraClippingRange();
            setcamera = false;
        }

        renderWindow->Render();

        if (screenShotPath != "") {
            vtkSmartPointer<vtkWindowToImageFilter> filter = vtkSmartPointer<vtkWindowToImageFilter>::New();
            filter->SetInput(this->renderWindow);
            filter->SetMagnification(screenShotMagification);
            filter->SetInputBufferTypeToRGBA();
            filter->ShouldRerenderOn();
            filter->Update();

            vtkSmartPointer<vtkPNGWriter> writer = vtkSmartPointer<vtkPNGWriter>::New();
            writer->SetFileName(screenShotPath.c_str());
            writer->SetInputConnection(filter->GetOutputPort());
            writer->Write();
            screenShotPath = "";
        }
        changed = false;
        newObject = false;
    }
}

/* adds the given VTK3DObject */
void VTK3DWindow::add(std::shared_ptr<VTK3DObject> object) {
    VTKInteractor::lock();
    if (std::find(objects.begin(), objects.end(), object) == objects.end()) {
        objects.push_back(object);
        renderer->AddActor(object->getActor());
        changed = true;
        newObject = true;
	}
    VTKInteractor::unlock();

    while (!object->isInitialized())
        Thread::yield();
}

/* removes the given VTK3DObject */
void VTK3DWindow::remove(std::shared_ptr<VTK3DObject> object) {
    VTKInteractor::lock();
    if (std::find(objects.begin(), objects.end(), object) != objects.end()) {
        objects.erase(std::find(objects.begin(), objects.end(), object));
        renderer->RemoveActor(object->getActor());
        changed = true;
    }
    VTKInteractor::unlock();
}

/* saves a screnshot in the gifen magnification */
void VTK3DWindow::saveScreenShot(string path, int magnification) {
 
    VTKInteractor::lock();
    screenShotPath = path;
    screenShotMagification = magnification;
    changed = true;
    VTKInteractor::unlock();

    while (screenShotPath != "")
        Thread::yield();
}

/* returns the camera possition */
void VTK3DWindow::getCamera(
    double &x,
    double &y,
    double &z,
    double &fx,
    double &fy,
    double &fz,
    double &ux,
    double &uy,
    double &uz
) {

    VTKInteractor::lock();
    renderer->GetActiveCamera()->GetPosition(x, y, z);
    renderer->GetActiveCamera()->GetFocalPoint(fx, fy, fz);
    renderer->GetActiveCamera()->GetViewUp(ux, uy, uz);
    VTKInteractor::unlock();
}

/* sets the camera possition */
void VTK3DWindow::setCamera(
    double x,
    double y,
    double z,
    double fx,
    double fy,
    double fz,
    double ux,
    double uy,
    double uz
) {

    VTKInteractor::lock();
    this->x = x;
    this->y = y;
    this->z = z;
    this->fx = fx;
    this->fy = fy;
    this->fz = fz;
    this->ux = ux;
    this->uy = uy;
    this->uz = uz;
    
    setcamera = true;
    changed   = true;
    VTKInteractor::unlock();
}
