#include <vtkFloatArray.h>
#include <vtkProperty.h>
#include "VTK3DObject.h"
#include "VTKInteractor.h"
using namespace SNN;
using namespace Visualizations;

/* constructor */
VTK3DObject::VTK3DObject(VTK3DType type): 
    dataChanged(false), initialized(false), type(type) {

    mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    actor  = vtkSmartPointer<vtkActor>::New();
    actor->GetProperty()->SetLineWidth(3);
}

/* indicates wether tha data of this ahs changed */
bool VTK3DObject::changed() {
    return dataChanged || !initialized;
}

/* updates this */
void VTK3DObject::update() {

    if (!initialized) {
        this->initVTK(mapper);
        actor->SetMapper(mapper);

        initialized = true;
    }
    this->doUpdate();
    mapper->Update();
    dataChanged = false;
}

/* sets dataChanged to true */
void VTK3DObject::modified() {
    dataChanged = true;
}

void VTK3DObject::setColor(
    unsigned char r,
    unsigned char g,
    unsigned char b,
    unsigned char a
) {
    unsigned char color[4] = {r, g, b, a};
    VTKInteractor::lock();
    actor->GetProperty()->SetColor(color[0] / 255.0, color[1] / 255.0, color[2] / 255.0);
    actor->GetProperty()->SetOpacity(color[3] / 255.0);
    dataChanged = true;
    VTKInteractor::unlock();
}
