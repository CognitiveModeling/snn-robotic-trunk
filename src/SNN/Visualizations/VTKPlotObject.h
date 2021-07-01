#ifndef __VTK_PLOT_OBJECT_H__
#define __VTK_PLOT_OBJECT_H__
#include <vtkSmartPointer.h>
#include <vtkTable.h>
#include <vtkPlot.h>
#include <vtkChartXY.h>
#include <vector>
#include <string>
namespace SNN {

    namespace Visualizations {

        /* the different plot types */
        enum class VTKPlotType { POINTS, LINE };
        
        class VTKPlotObject {

            private:

                /* the data table of this */
                vtkSmartPointer<vtkTable> table;

                /* the plot of this */
                vtkSmartPointer<vtkPlot> plot;

                /* indicates that the data of this has changed */
                bool dataChanged;

                /* the type of this */
                VTKPlotType type;

                /* the color of this in RGBA */
                unsigned char color[4];

            public:

                /* constructor */
                VTKPlotObject(VTKPlotType type, std::string name);

                /* indicates wether tha data of this has changed */
                bool changed();

                /* updates this */
                void update(vtkSmartPointer<vtkChartXY>);

                /* changes the data of this */
                void setData(std::vector<FloatType> dataX, std::vector<FloatType> dataY);

                /* sets the color of this */
                void setColor(std::initializer_list<unsigned char> color);
                void setColor(unsigned char color[4]);

                /* removes this from the given chart */
                void removeFromChart(vtkSmartPointer<vtkChartXY>);

        };

    }

}
#endif /* __VTK_PLOT_OBJECT_H__ */
