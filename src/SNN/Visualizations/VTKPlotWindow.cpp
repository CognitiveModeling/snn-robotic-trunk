#include <vtkContextScene.h>
#include <vtkAxis.h>
#include <vtkTextProperty.h>
#include "VTKPlotWindow.h"
#include "VTKPlotObject.h"
#include "VTKInteractor.h"
#include <algorithm>
using namespace SNN;
using namespace Visualizations;

/* constructor */
VTKPlotWindow::VTKPlotWindow(std::shared_ptr<VTKPlotInteractorStyleGenerator> styleGenerator) : 
    styleGenerator(styleGenerator), changed(false) { }

/* destructor */
VTKPlotWindow::~VTKPlotWindow() { }


/* shoudl close this */
void VTKPlotWindow::doClose() {

    objects.clear();
    view->GetScene()->RemoveItem(chart);
    chart->ClearPlots();
    chart = NULL;
    view  = NULL;

    interactorStyle->SetEnabled(false);
    interactorStyle->SetScene(NULL);
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
vtkSmartPointer<vtkRenderWindowInteractor> VTKPlotWindow::initVTK() {

    view            = vtkSmartPointer<vtkContextView>::New();
    chart           = vtkSmartPointer<vtkChartXY>::New();
    interactorStyle = styleGenerator->generate();
    renderer        = view->GetRenderer();
    renderWindow    = view->GetRenderWindow();

    if (VTKWindow::interactor == NULL) 
        interactor  = vtkSmartPointer<VTKInteractor>::New();
    else 
        interactor  = view->GetInteractor();
        
    view->SetInteractor(interactor);
    interactor->SetInteractorStyle(interactorStyle);
    interactorStyle->SetScene(view->GetScene());
    renderer->ResetCamera();
    view->GetScene()->AddItem(chart);
    chart->SetShowLegend(true);

    textActor = vtkSmartPointer<vtkTextActor>::New();
    textActor->SetInput("");
    textActor->GetTextProperty()->SetFontSize(14);
    textActor->GetTextProperty()->SetColor(0, 0, 0);
    textActor->GetTextProperty()->SetOpacity(0.75);
    textActor->GetTextProperty()->SetBackgroundColor(1,1,1);
    textActor->GetTextProperty()->SetBackgroundOpacity(0.75);
    textActor->GetTextProperty()->SetBold(true);

    renderer->AddActor2D(textActor);

    return interactor;
}

/* should update this */
void VTKPlotWindow::doUpdate() {
    double bbox[4];
    textActor->GetBoundingBox(renderer, bbox);
    textActor->SetPosition(
        renderWindow->GetSize()[0] - (bbox[1] - bbox[0]),
        renderWindow->GetSize()[1] - (bbox[3] - bbox[2])
    );

    bool render = false;
    for (auto &o: objects) {
        if (o->changed()) {
            o->update(chart);
            render = true;
        }
    }
    
    if (render || changed) {
        view->Render();
        changed = false;
    }
}

/* adds the given VTKPlotObject */
void VTKPlotWindow::add(std::shared_ptr<VTKPlotObject> object) {
    VTKInteractor::lock();
    if (std::find(objects.begin(), objects.end(), object) == objects.end())
        objects.push_back(object);
    VTKInteractor::unlock();
}

/* removes the given VTKPlotObject */
void VTKPlotWindow::remove(std::shared_ptr<VTKPlotObject> object) {
    VTKInteractor::lock();
    if (std::find(objects.begin(), objects.end(), object) != objects.end()) {
        object->removeFromChart(chart);
        objects.erase(std::find(objects.begin(), objects.end(), object));
    }
    VTKInteractor::unlock();
}

/* removes all VTKPlotObjects */
void VTKPlotWindow::clear() {
    VTKInteractor::lock();
    for (auto &object: objects) 
        object->removeFromChart(chart);

    objects.clear();
    VTKInteractor::unlock();
}

/* sets the x and y axis name */
void VTKPlotWindow::setAxisNames(std::string xAxis, std::string yAxis) {
    VTKInteractor::lock();

    chart->GetAxis(vtkAxis::BOTTOM)->SetTitle(xAxis);
    chart->GetAxis(vtkAxis::LEFT)->SetTitle(yAxis);
    changed = true;

    VTKInteractor::unlock();
}

/* sets wether the legend should be shown or not */
void VTKPlotWindow::showLegend(bool show) {
    VTKInteractor::lock();

    chart->SetShowLegend(show);
    changed = true;

    VTKInteractor::unlock();
}

/* displays the given string */
void VTKPlotWindow::showText(std::string text) {
    VTKInteractor::lock();
    textActor->SetInput(text.c_str());
    changed = true;
    VTKInteractor::unlock();
}
