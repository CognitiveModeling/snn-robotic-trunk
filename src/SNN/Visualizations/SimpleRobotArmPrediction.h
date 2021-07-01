#ifndef __SIMPLE_FULL_ROBOT_ARM_PREDICTION_H__ 
#define __SIMPLE_FULL_ROBOT_ARM_PREDICTION_H__

namespace SNN {

    namespace Visualizations {

        class SimpleRobotArmPrediction {

            private:

                /* the position and directions of this */
                FloatType x, y, z, xUp, yUp, zUp, xXDirection, yXDirection, zXDirection;

                /* pi constant */
                const FloatType PI;

            public:

                /* constructor */
                SimpleRobotArmPrediction(): 
                    x(0), y(0), z(0), xUp(0), yUp(1), zUp(0),
                    xXDirection(1), yXDirection(0), zXDirection(0), 
                    PI(3.141592653589793) { }


                /* sets the position of this of this */
                void setPosition(
                    FloatType x, 
                    FloatType y, 
                    FloatType z
                ) {
                    this->x = x;
                    this->y = y;
                    this->z = z;
                }

                /* sets the directions of this of this */
                void setDirections(
                    FloatType xUp, 
                    FloatType yUp, 
                    FloatType zUp,
                    FloatType xXDirection, 
                    FloatType yXDirection, 
                    FloatType zXDirection
                ) {
                    FloatType lengthUp = sqrt(pow(xUp, 2) + pow(yUp, 2) + pow(zUp, 2)); 
                    xUp /= lengthUp;
                    yUp /= lengthUp; 
                    zUp /= lengthUp; 
                    
                    FloatType lengthXDirection = sqrt(pow(xXDirection, 2) + pow(yXDirection, 2) + pow(zXDirection, 2));
                    xXDirection /= lengthXDirection;
                    yXDirection /= lengthXDirection;
                    zXDirection /= lengthXDirection;
                    
                    this->xUp = xUp; 
                    this->yUp = yUp; 
                    this->zUp = zUp;
                    this->xXDirection = xXDirection; 
                    this->yXDirection = yXDirection; 
                    this->zXDirection = zXDirection;
                }

                /* returns the position and orientation of this of this */
                void getPosition(
                    FloatType &x, 
                    FloatType &y, 
                    FloatType &z
                ) {
                    x = this->x;
                    y = this->y;
                    z = this->z;
                }
                void getUpVector(
                    FloatType &xUp, 
                    FloatType &yUp, 
                    FloatType &zUp
                ) {
                    xUp = this->xUp;               
                    yUp = this->yUp; 
                    zUp = this->zUp;
                }
                void getXDirection(
                    FloatType &xXDirection, 
                    FloatType &yXDirection, 
                    FloatType &zXDirection
                ) {
                    xXDirection = this->xXDirection; 
                    yXDirection = this->yXDirection; 
                    zXDirection = this->zXDirection;
                }

        };

    }
}
#endif
