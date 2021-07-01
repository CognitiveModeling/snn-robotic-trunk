#include <vtkFloatArray.h>
#include "VTKPlotObject.h"
#include "VTKInteractor.h"
using namespace SNN;
using namespace Visualizations;

/* constructor */
VTKPlotObject::VTKPlotObject(VTKPlotType type, std::string name): 
    dataChanged(false), type(type), color{0,255,0,255} {

    table = vtkSmartPointer<vtkTable>::New();

    vtkSmartPointer<vtkFloatArray> xAxis= vtkSmartPointer<vtkFloatArray>::New();
    vtkSmartPointer<vtkFloatArray> yAxis= vtkSmartPointer<vtkFloatArray>::New();
    xAxis->SetName("X Axis");
    yAxis->SetName(name.c_str());

    table->AddColumn(xAxis);
    table->AddColumn(yAxis);

}

/* indicates wether tha data of this ahs changed */
bool VTKPlotObject::changed() {
    return dataChanged;
}

/* updates this */
void VTKPlotObject::update(vtkSmartPointer<vtkChartXY> chart) {

    if (dataChanged) {
        chart->RemovePlotInstance(plot);
        
        if (table->GetNumberOfRows() > 0) {
            if (type == VTKPlotType::POINTS)
                plot = chart->AddPlot(vtkChart::POINTS);
            else if (type == VTKPlotType::LINE)
                plot = chart->AddPlot(vtkChart::LINE);

            plot->SetInputData(table, 0, 1);
            plot->SetColor(color[0], color[1], color[2], color[3]);
        }

        dataChanged = false;
    }
}

/* removes this from the given chart */
void VTKPlotObject::removeFromChart(vtkSmartPointer<vtkChartXY> chart) {
    chart->RemovePlotInstance(plot);
    dataChanged = true;
}

/* changes the data of this */
void VTKPlotObject::setData(std::vector<FloatType> dataX, std::vector<FloatType> dataY) {
    VTKInteractor::lock();

    assert(dataX.size() == dataY.size());
    table->SetNumberOfRows(dataX.size());

    for (unsigned i = 0; i < dataX.size(); i++) {
        table->SetValue(i, 0, dataX[i]);
        table->SetValue(i, 1, dataY[i]);
    }

    dataChanged = true;
    VTKInteractor::unlock();
}

/* sets the color of this */
void VTKPlotObject::setColor(std::initializer_list<unsigned char> color) {
    assert(color.size() == 4);
    VTKInteractor::lock();
    std::copy(color.begin(), color.end(), this->color);
    dataChanged = true;
    VTKInteractor::unlock();
}

void VTKPlotObject::setColor(unsigned char color[4]) {
    VTKInteractor::lock();
    this->color[0] = color[0];
    this->color[1] = color[1];
    this->color[2] = color[2];
    this->color[3] = color[3];
    dataChanged = true;
    VTKInteractor::unlock();
}
