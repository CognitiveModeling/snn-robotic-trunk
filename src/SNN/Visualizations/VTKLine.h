#ifndef __VTK_LINE_H__ 
#define __VTK_LINE_H__
#include <vtkPoints.h>
#include <vtkCellArray.h>
#include "VTKInteractor.h"
#include "VTK3DObject.h"
#include "utils.h"


namespace SNN {

    namespace Visualizations {

        class VTKLine: public VTK3DObject {

            private:


                /* the mapper of this */
                vtkSmartPointer<vtkPolyDataMapper> mapper;

                /* the points of this line */
                vtkSmartPointer<vtkPoints> points;

                /* inditates that the renderer is blocked */
                bool rendererLocked;

                /* should initalize vtk related stuff within the vtk Thread */
                virtual void initVTK(vtkSmartPointer<vtkPolyDataMapper> mapper) {
                    points = vtkSmartPointer<vtkPoints>::New();
                    points->SetNumberOfPoints(0);
                    this->mapper = mapper;
                }

            public:

                /* constructor */
                VTKLine(): 
                    VTK3DObject(VTK3DType::POLYS), 
                    rendererLocked(false) { }


                /* sets the center of this */
                void setPoints(std::vector<FloatType> x, std::vector<FloatType> y, std::vector<FloatType> z) {
                    while (!this->isInitialized())
                        Thread::yield();
                    
                    if (!rendererLocked) 
                        VTKInteractor::lock();

                    unsigned length = vmin(x.size(), y.size(), z.size());
                    points->SetNumberOfPoints(length);

                    vtkSmartPointer<vtkCellArray> cells = vtkSmartPointer<vtkCellArray>::New();
                    cells->InsertNextCell(length);

                    for (unsigned i = 0; i < length; i++) {
                        points->SetPoint(i, x[i], y[i], z[i]);
                        cells->InsertCellPoint(i);
                    }

                    vtkSmartPointer<vtkPolyData> polyData = vtkSmartPointer<vtkPolyData>::New();
                    polyData->SetPoints(points);
                    polyData->SetLines(cells);
                    mapper->SetInputData(polyData);

                    if (!rendererLocked) 
                        VTKInteractor::unlock();
                    this->modified();
                }

                /* returns the number of points */
                size_t size() { return points->GetNumberOfPoints(); }

                /* returns the point with the given index */
                void getPoint(unsigned i, FloatType &x, FloatType &y, FloatType &z) {
                    
                    double point[3];
                    points->GetPoint(i, point);

                    x = point[0];
                    y = point[1];
                    z = point[2];
                }

                /* updates this */
                void doUpdate() { }

                /* sets wether the renderer is allready locked */
                void setRendererLocked(bool locked) {
                    rendererLocked = locked;
                }

        };
    }
}
#endif
