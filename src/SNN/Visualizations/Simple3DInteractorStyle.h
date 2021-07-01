#ifndef __SIMPLE_MONITOR_3D_INTERACTOR_STYLE__
#define __SIMPLE_MONITOR_3D_INTERACTOR_STYLE__
#include <vtkContextInteractorStyle.h>
#include <vtkRenderWindowInteractor.h>

namespace SNN {

    namespace Visualizations {
    
        class Simple3DInteractorStyle: public vtkInteractorStyleTrackballCamera {

            private:

                /* the last pressed key */
                std::string lastPressedKey;

            protected:

                /* constructor */
                Simple3DInteractorStyle(): lastPressedKey("") { }

            public:

                vtkTypeMacro(Simple3DInteractorStyle, vtkInteractorStyleTrackballCamera);

                static Simple3DInteractorStyle* New() { return new Simple3DInteractorStyle(); }

                void OnKeyPress() {
                    this->lastPressedKey = this->Interactor->GetKeySym();
                }

                /* returns and resets the last pressed key */
                std::string getLastKey() {
                    std::string key = lastPressedKey;
                    lastPressedKey  = "";
                    return key;
                }
        };

        /* generator class for a vtkContextInteractorStyle */
        class Simple3DInteractorStyleGenerator: public VTK3DInteractorStyleGenerator {

            public:

                /* should create the vtkContextInteractorStyle */
                virtual vtkSmartPointer<vtkInteractorStyleTrackballCamera> generate() {
                    return vtkSmartPointer<Simple3DInteractorStyle>::New();
                }

        };

    }

}

#endif /* __SIMPLE_MONITOR_3D_INTERACTOR_STYLE__ */
