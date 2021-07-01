#ifndef __VTK_WOLD_BORDERS_H__ 
#define __VTK_WOLD_BORDERS_H__
#include "VTKGrid.h"
#include "reactphysics3d.h"

namespace SNN {

    namespace Visualizations {

        class VTKWorldBorders: public VTKGrid {

            public:

                /* constructor */
                VTKWorldBorders(rp3d::DynamicsWorld &world, FloatType size, FloatType gridSize): 
                    VTKGrid(gridSize / 2, 0, size / gridSize) {

                    rp3d::RigidBody *ground = world.createRigidBody(rp3d::Transform::identity());
                    ground->setType(rp3d::BodyType::STATIC);
                    
                    ground->addCollisionShape(
                        new rp3d::BoxShape(rp3d::Vector3(size, 1, size)), 
                        rp3d::Transform(rp3d::Vector3(0, -1, 0), rp3d::Quaternion::identity()),
                        rp3d::decimal(100)
                    );
                    ground->addCollisionShape(
                        new rp3d::BoxShape(rp3d::Vector3(1, size, size)),
                        rp3d::Transform(rp3d::Vector3(-size, 0, 0), rp3d::Quaternion::identity()),
                        rp3d::decimal(100)
                    );
                    ground->addCollisionShape(
                        new rp3d::BoxShape(rp3d::Vector3(1, size, size)),
                        rp3d::Transform(rp3d::Vector3(size, 0, 0), rp3d::Quaternion::identity()),
                        rp3d::decimal(100)
                    );
                    ground->addCollisionShape(
                        new rp3d::BoxShape(rp3d::Vector3(size, size, 1)),
                        rp3d::Transform(rp3d::Vector3(0, 0, -size), rp3d::Quaternion::identity()),
                        rp3d::decimal(100)
                    );
                    ground->addCollisionShape(
                        new rp3d::BoxShape(rp3d::Vector3(size, size, 1)),
                        rp3d::Transform(rp3d::Vector3(0, 0, size), rp3d::Quaternion::identity()),
                        rp3d::decimal(100)
                    );
                }

        };
    }
}
#endif
