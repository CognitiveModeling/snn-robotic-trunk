## Many-Joint Robot Arm Control with Recurrent Spiking NeuralNetworks

Repository containing the source code to train and infere both robots from the paper

### Compilation
Requirements are cmake, cuda, VTK and devil (image library).
From within the root of the repository run:
> mkdir build
> cd build
> cmake ..
> make

### Generating datatsets
From within the build directory run:
> ./RobotArmPredictor --generate-dataset 100000 --dataset-name dataset --realistic-v2

or

> ./RobotArmPredictor --generate-dataset 100000 --dataset-name dataset --realistic-v3

to generate a dataset for the V2 or V3 version of the robot with 100,000 samples

### Training
Example command to train a network and save the result:
> ./RobotArmPredictor --loadSimulation dataset -j 10 --quaternionRotation --realistic-v2 --device 0 --epochs 50000 --decay 0.5 --fr-decay 0.5 --decay-interval 10000 --back-propagation --numHidden 256 --batchSize 128 --evaluate

