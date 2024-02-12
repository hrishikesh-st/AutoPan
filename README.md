# AutoPan

### *RBE549: Computer Vision - [Worcester Polytechnic Institute](https://www.wpi.edu/), Spring 2024*

## Project Guidelines:
The project is divided into two phases. The first phase is to implement a classical image stitching pipeline. The second phase is to implement a deep learning-based image stitching pipeline. 
Details of the project can be found [here](https://rbe549.github.io/spring2024/proj/p1/).

## Phase 1: Traditional Approach

### Overview:
Phase 1 of the project focuses on creating a seamless panorama from a set of images using feature detection, feature matching, and image stitching to produce a comprehensive panoramic image.

#### Steps to run the code:

To perform stitching of n images, use the following command:

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
Wrappery.py reads input images from "Data" folder and all the ouptuts are stored in the "Results" folder.

The Data folder should have the following structure:
```bash
Data
├── Test
│   └── TestSet1
└── Train
    ├── Set1
    └── CustomSet1
```

Wrappery.py reads all the input images from "Data" folder relevant to the argument passed, i.e. Train if --Train or Test otherwise. --ImageSet argument specifies the set name to run the stitching algorithm. All the outputs are stored in "Results" folder under the folder name specied by the --Imageset argument.

Example to run the code on Set2:
```bash
python Wrapper.py --Train --ImageSet Set2
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


## Phase 2: Deep Learning Approach

### Overview:
Phase 2 of the project focuses on finding the homography matrix between the two images to be stitched using a neural network. The training of the neural network is implemented using 2 regimes - Supervised and Unsupervised.

#### Steps to run the code:

To train the neural network, use the following command:

```bash
python Train.py --DatasetPath <PATH_TO_CREATED_DATASET> --LogDir <NAME_OF_LOGDIR> --Msg <TRIAL_RUN_MESSAGE> --NumEpochs <TRAINING_EPOCHS>

usage: Train.py [-h] [--DatasetPath DATASETPATH] [--CheckPointPath CHECKPOINTPATH] [--ModelType MODELTYPE] [--NumEpochs NUMEPOCHS]
                [--DivTrain DIVTRAIN] [--MiniBatchSize MINIBATCHSIZE] [--LoadCheckPoint LOADCHECKPOINT] [--LogsPath LOGSPATH]
                [--LogDir LOGDIR] [--LR LR] [--WD WD] [--GradientClip GRADIENTCLIP] [--Msg MSG]

options:
  -h, --help                          show this help message and exit
  --DatasetPath DATASETPATH           Base path of images, Default: HomographyDataset1
  --CheckPointPath CHECKPOINTPATH     Path to save Checkpoints, Default: ../Checkpoints/
  --ModelType MODELTYPE               Model type, Supervised or Unsupervised? Choose from Sup and Unsup, Default:Sup
  --NumEpochs NUMEPOCHS               Number of Epochs to Train for, Default:50
  --DivTrain DIVTRAIN                 Factor to reduce Train data by per epoch, Default:1
  --MiniBatchSize MINIBATCHSIZE       Size of the MiniBatch to use, Default:512
  --LoadCheckPoint LOADCHECKPOINT     Load Model from latest Checkpoint from CheckPointsPath?, Default:0
  --LogsPath LOGSPATH                 Path to save Logs for Tensorboard, Default=Logs/
  --LogDir LOGDIR                     name of the log file
  --LR LR                             Learning Rate
  --WD WD                             Weight Decay
  --GradientClip GRADIENTCLIP         Gradient Clipping
  --Msg MSG                           Message
```

Train.py reads input images from "Data" folder. The Data folder should have the following structure:
```bash
Data
└── HomographyDataset
    ├── train_names.txt     # text file containing names of all images in the Train/P_A directory.
    ├── Train
    │   ├── labels
    │   ├── P_A
    │   └── P_B
    └── Val
        ├── labels
        ├── P_A
        └── P_B
```

The code automatically creates the `Logs` folder and saves the logs of the current training run in the `LogDir` subfolder. It contains a `model` subfolder which has the best model weights and the model weights at every 5th epoch. It also contains a `plots` subfolder with the plots for the training and validation losses, plotted versus epochs. Finally, the text file `logs.txt` saves the training hyperparameters and logs for each epoch.

Example to train the supervised network:
```bash
python Train.py --DatasetPath HomographyDataset --LogDir TestLogs --Msg "supervised" --NumEpochs 100
```

Example to train the unsupervised network:
```bash
python Train.py --DatasetPath HomographyDataset --LogDir TestLogs --Msg "unsupervised trial" --NumEpochs 50 --ModelType Unsup --MiniBatchSize 256 --LR 0.0001
```

To perform stitching using the neural network, use the following command:
```bash
python Wrapper.py --ImageSet <IMAGESET_NAME> --ModelType <MODEL_TYPE> --CheckpointPath <PATH_TO_CHECKPOINT>

usage: Wrapper.py [-h] [--ImageSet IMAGESET] [--ModelType MODELTYPE] [--CheckpointPath CHECKPOINTPATH]

options:
  -h, --help            show this help message and exit
  --ImageSet IMAGESET               Choose the set to run the test on Options are Set1, Set2, Set3, CustomSet1, CustomSet2, Default:Set1
  --ModelType MODELTYPE             Model type, Supervised or Unsupervised? Choose from Sup and Unsup, Default:Sup
  --CheckpointPath CHECKPOINTPATH   checkpoint path to load model weights.
```

Example to perform panaroma stitching using the supervised model:
```bash
python Wrapper.py --ImageSet Set2 --ModelType Sup --CheckpointPath TestLogs
```