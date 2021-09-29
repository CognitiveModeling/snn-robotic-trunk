## Many-Joint Robot Arm Control with Recurrent Spiking Neural Networks

This repository contains the source code for training and control of two robotic trunk variants from our paper *Many-Joint Robot Arm Control with Recurrent Spiking Neural Networks* (presented at IROS 2021).

```bibtex
@inproceedings{traub2021manyjoint,
    title = {Many-Joint Robot Arm Control with Recurrent Spiking Neural Networks},
    author = {Manuel Traub and Robert Legenstein and Sebastian Otte},
    booktitle = {IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
    year = {2021},
    month = sep,
    pages = {4895--4902}
}

```

A preprint is available on arXiv:
https://arxiv.org/abs/2104.04064


CAD trunk models can be found here:
https://www.myminifactory.com/object/3d-print-trunk-like-many-joint-robotic-arm-154230

See our video on youtube:
https://youtu.be/q9LP6b98Gqw

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

