#ifndef __VTK_GRID_H__ 
#define __VTK_GRID_H__
#include <vtkPoints.h>
#include <vtkCellArray.h>
#include <vtkPolyData.h>

namespace SNN {

    namespace Visualizations {

        class VTKGrid: public VTK3DObject {

            private:

                /* the radius and position of this */
                FloatType r, y;
                
                /* the number of grid elements */
                unsigned n;

                /* should initalize vtk related stuff within the vtk Thread */
                virtual void initVTK(vtkSmartPointer<vtkPolyDataMapper> mapper) {

                    vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
                    vtkSmartPointer<vtkCellArray> cells = vtkSmartPointer<vtkCellArray>::New();

                    int i = 0;
                    for (FloatType x = -r - n*2*r; x <= r + n*2*r; x += 2*r) {
                        cells->InsertNextCell(2*(n+1));
                        for (FloatType z = -r - n*2*r; z <= r + n*2*r; z += 2*r, i++) {
                            points->InsertNextPoint(x, y, z);
                            cells->InsertCellPoint(i);
                        }
                    }
                    for (FloatType z = -r - n*2*r; z <= r + n*2*r; z += 2*r) {
                        cells->InsertNextCell(2*(n+1));
                        for (FloatType x = -r - n*2*r; x <= r + n*2*r; x += 2*r, i++) {
                            points->InsertNextPoint(x, y, z);
                            cells->InsertCellPoint(i);
                        }
                    }

                    vtkSmartPointer<vtkPolyData> grid = vtkSmartPointer<vtkPolyData>::New();
                    grid->SetPoints(points); 
                    grid->SetLines(cells);

                    mapper->SetInputData(grid);
                }

            public:

                /* constructor */
                VTKGrid(FloatType r, FloatType y, unsigned n): 
                    VTK3DObject(VTK3DType::POLYS), r(r), y(y), n(n) { }

        };
    }
}
#endif
