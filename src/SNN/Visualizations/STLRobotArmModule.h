#ifndef __STL_ROBOT_ARM_MODULE_H__
#define __STL_ROBOT_ARM_MODULE_H__
#include "BasicRobotArmModule.h"
#include <vtkSTLReader.h>
#include <vtkAppendPolyData.h>
#include <vtkCleanPolyData.h>

namespace SNN {

    namespace Visualizations {

        class STLRobotArmModule: public BasicRobotArmModule {
          
            private:

                /* the base file (without motors and gears) */
                std::string baseFile;

                /* the tree linear gears files */
                std::string linearGearFile1;
                std::string linearGearFile2;
                std::string linearGearFile3;

                /* stl reader for bas and linear gears */
                vtkSmartPointer<vtkSTLReader> baseReader, 
                    linearGearReader1, linearGearReader2, linearGearReader3;

                /* linear transations for gears */
                vtkSmartPointer<vtkTransform> linearGearTransform1;
                vtkSmartPointer<vtkTransform> linearGearTransform2;
                vtkSmartPointer<vtkTransform> linearGearTransform3;
                vtkSmartPointer<vtkTransformPolyDataFilter> linearGearTransformFilter1;
                vtkSmartPointer<vtkTransformPolyDataFilter> linearGearTransformFilter2;
                vtkSmartPointer<vtkTransformPolyDataFilter> linearGearTransformFilter3;

                /* append and clean filter to combine all parts */
                vtkSmartPointer<vtkAppendPolyData> appendFilter;
                vtkSmartPointer<vtkCleanPolyData> cleanFilter;

            public:

                /* constructor */
                STLRobotArmModule(
                    std::string baseFile,
                    std::string linearGearFile1,
                    std::string linearGearFile2,
                    std::string linearGearFile3
                ) : 
                    baseFile(baseFile), 
                    linearGearFile1(linearGearFile1),
                    linearGearFile2(linearGearFile2),
                    linearGearFile3(linearGearFile3) { 
                }


                /* should initalize vtk related stuff within the vtk Thread */
                virtual vtkSmartPointer<vtkPolyData> getPolyData() {
                        
                    baseReader = vtkSmartPointer<vtkSTLReader>::New();
                    baseReader->SetFileName(baseFile.c_str());
                    baseReader->Update();

                    linearGearReader1 = vtkSmartPointer<vtkSTLReader>::New();
                    linearGearReader1->SetFileName(linearGearFile1.c_str());
                    linearGearReader1->Update();

                    linearGearReader2 = vtkSmartPointer<vtkSTLReader>::New();
                    linearGearReader2->SetFileName(linearGearFile2.c_str());
                    linearGearReader2->Update();

                    linearGearReader3 = vtkSmartPointer<vtkSTLReader>::New();
                    linearGearReader3->SetFileName(linearGearFile3.c_str());
                    linearGearReader3->Update();

                    linearGearTransform1 = vtkSmartPointer<vtkTransform>::New();
                    linearGearTransform2 = vtkSmartPointer<vtkTransform>::New();
                    linearGearTransform3 = vtkSmartPointer<vtkTransform>::New();

                    linearGearTransformFilter1 = vtkSmartPointer<vtkTransformPolyDataFilter>::New();
                    linearGearTransformFilter2 = vtkSmartPointer<vtkTransformPolyDataFilter>::New();
                    linearGearTransformFilter3 = vtkSmartPointer<vtkTransformPolyDataFilter>::New();

                    linearGearTransformFilter1->SetInputData(linearGearReader1->GetOutput());
                    linearGearTransformFilter2->SetInputData(linearGearReader2->GetOutput());
                    linearGearTransformFilter3->SetInputData(linearGearReader3->GetOutput());

                    linearGearTransformFilter1->SetTransform(linearGearTransform1);
                    linearGearTransformFilter2->SetTransform(linearGearTransform2);
                    linearGearTransformFilter3->SetTransform(linearGearTransform3);

                    linearGearTransformFilter1->Update();
                    linearGearTransformFilter2->Update();
                    linearGearTransformFilter3->Update();

                    appendFilter = vtkSmartPointer<vtkAppendPolyData>::New();
                    appendFilter->AddInputData(baseReader->GetOutput());
                    appendFilter->AddInputData(linearGearTransformFilter1->GetOutput());
                    appendFilter->AddInputData(linearGearTransformFilter2->GetOutput());
                    appendFilter->AddInputData(linearGearTransformFilter3->GetOutput());
                    appendFilter->Update();

                    cleanFilter = vtkSmartPointer<vtkCleanPolyData>::New();
                    cleanFilter->SetInputConnection(appendFilter->GetOutputPort());
                    cleanFilter->Update();

                    return cleanFilter->GetOutput();
                }

                void doMove(
                    FloatType, 
                    FloatType, 
                    FloatType, 
                    vtkSmartPointer<vtkTransform> 
                ) {
                    //linearGearTransform1->Translate(0, 0, gear1);
                    //linearGearTransform2->Translate(0, 0, gear2);
                    //linearGearTransform3->Translate(0, 0, gear3);
                }

                virtual void doUpdate() {

                    linearGearTransformFilter1->Update();
                    linearGearTransformFilter2->Update();
                    linearGearTransformFilter3->Update();

                    appendFilter->Update();
                    cleanFilter->Update();
                }
                    
        };
    }
}
#endif /* __STL_ROBOT_ARM_MODULE_H__ */
