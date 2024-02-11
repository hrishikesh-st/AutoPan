# AutoPan

### *RBE549: Computer Vision - [Worcester Polytechnic Institute](https://www.wpi.edu/), Spring 2024*

## Project Guidelines:
The project is divided into two phases. The first phase is to implement a classical image stitchinh pipeline. The second phase is to implement a deep learning-based image stitching pipeline. 
Details of the project can be found [here](https://rbe549.github.io/spring2024/proj/p1/).

## Phase 1: Traditional Approach

### Overview:
Phase 1 of the project focuses on creating a seamless panorama from a set of images using feature detection, feature matching, and image stitching to produce a comprehensive panoramic image

#### Steps to run the code:

To run the PBLite boundary detection, use the following command:

```bash
python Wrapper.py
```
Wrappery.py reads input images from "BSDS500" folder and all the ouptuts are stored in the "Outputs" folder.

```bash
python Wrapper.py --Train --ImageSet <IMAGESET_NAME>

usage: Wrapper.py [-h] [--Train] [--ImageSet IMAGESET] [-p] [-b]

optional arguments:
  -h, --help           show this help message and exit
  --Train              Choose the set to run the test on, Default:True
  --ImageSet IMAGESET  Choose the set to run the test on Options are Set1, Set2, Set3, CustomSet1, CustomSet2, Default:Set1
  -p, --Poisson        Choose whether to use Poisson blending or not, Default:False
  -b, --blending       Choose whether to use Blending or not, Default:False
``` 
Wrappery.py reads all the input images from "Data" folder relevant to the argument passed, i.e. Train if --Train or Test otherwise. --ImageSet argument specifies the set name to run the stitching algorithm. All the outputs are stored in "Results" folder under the folder name specied by the --Imageset argument.

Example to run the code on Set2:
```bash
python Wrapper.py --Train --ImageSet Set2 
```

The Data folder should have the following structure:
```bash
Data
├── Test
│   ├── TestSet1
└── Train
    ├── Set1
    ├── CustomSet1
```

### Results:

#### Input:
Original Image:
<p align="left">
  <img src="media/phase1_imgs/CustomSet1.png" alt="Original Image" style="width: 850px;"/>

#### Output:
Stitched Image:
<p align="left">
  <img src="media/phase1_imgs/CustomSet1_FullStitch.png" alt="Stitched Image" style="width: 850px;"/>