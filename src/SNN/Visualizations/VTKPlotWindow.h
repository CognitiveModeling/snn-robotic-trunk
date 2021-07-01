#ifndef __VTK_PLOT_WINDOW_H__
#define __VTK_PLOT_WINDOW_H__ 
#include <vtkContextInteractorStyle.h>
#include <vtkContextView.h>
#include <vtkChartXY.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkTextActor.h>
#include <memory>
#include <vector>
#include <string>
#include "VTKWindow.h"
namespace SNN {

    namespace Visualizations {

        /* forward declaration */
        class VTKPlotObject;

        /* generator class for a vtkContextInteractorStyle */
        class VTKPlotInteractorStyleGenerator {
            
            public:

                /* virtual constructor for correct polimorphism */
                virtual ~VTKPlotInteractorStyleGenerator() { }

                /* should create the vtkContextInteractorStyle */
                virtual vtkSmartPointer<vtkContextInteractorStyle> generate() {
                    return vtkSmartPointer<vtkContextInteractorStyle>::New();
                }

        };

        class VTKPlotWindow: public VTKWindow {

            private: 

                /* the visualized plot objects of this */
                std::vector<std::shared_ptr<VTKPlotObject>> objects;

                /* the vtk Context View of this */
                vtkSmartPointer<vtkContextView> view;

                /* the chart of this */
                vtkSmartPointer<vtkChartXY> chart;

                /* the main Renderer of this */
                vtkSmartPointer<vtkRenderer> renderer;

                /* the main render window of this */
                vtkSmartPointer<vtkRenderWindow> renderWindow;

                /* the interactor of this */
                vtkSmartPointer<vtkRenderWindowInteractor> interactor;

                /* the interactor stlye of this */
                vtkSmartPointer<vtkContextInteractorStyle> interactorStyle;

                /* the interactor style generator */
                std::shared_ptr<VTKPlotInteractorStyleGenerator> styleGenerator;

                /* text actor to show text on screen */
                vtkSmartPointer<vtkTextActor> textActor;

                /* indicates that this has changed */
                bool changed;


                /* should initalize vtk related stuff within the vtk Thread */
                virtual vtkSmartPointer<vtkRenderWindowInteractor> initVTK();

                /* should update this */
                virtual void doUpdate();

                /* shoudl close this */
                virtual void doClose();
            
            public: 

                /* constructor */
                VTKPlotWindow(
                    std::shared_ptr<VTKPlotInteractorStyleGenerator> styleGenerator = 
                        std::make_shared<VTKPlotInteractorStyleGenerator>()
                );

                /* virtual destructor for correct polimorphism */
                virtual ~VTKPlotWindow();

                /* updates this */
                void update();

                /* adds the given VTKPlotObject */
                void add(std::shared_ptr<VTKPlotObject>);

                /* removes the given VTKPlotObject */
                void remove(std::shared_ptr<VTKPlotObject>);

                /* removes all VTKPlotObjects */
                void clear();

                /* sets the x and y axis name */
                void setAxisNames(std::string xAxis, std::string yAxis);

                /* sets wether the legend should be shown or not */
                void showLegend(bool);
                
                /* displays the given string */
                void showText(std::string text);

        };

    }

}


#endif /* __VTK_PLOT_WINDOW_H__ */
