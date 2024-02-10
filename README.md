# P1: MyAutoPan

## Phase 1: Traditional Approach

### Steps to run the code:

To run the traditional stitching algorithm, use the following command:

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


## Phase 2: Deep Learning Approach

### Steps to run the code:

Train the model

```bash
python Train.py --DatasetPath <PATH_TO_CREATED_DATASET> --LogDir <NAME_OF_LOGDIR> --Msg <TRIAL_RUN_MESSAGE> --NumEpochs <TRAINING_EPOCHS>

usage: Train.py [-h] [--DatasetPath DATASETPATH] [--CheckPointPath CHECKPOINTPATH] [--ModelType MODELTYPE] [--NumEpochs NUMEPOCHS] [--DivTrain DIVTRAIN] [--MiniBatchSize MINIBATCHSIZE]
                [--LoadCheckPoint LOADCHECKPOINT] [--LogsPath LOGSPATH] [--LogDir LOGDIR] [--LR LR] [--WD WD] [--GradientClip GRADIENTCLIP] [--Msg MSG]

options:
  -h, --help            show this help message and exit
  --DatasetPath DATASETPATH
                        Base path of images, Default: HomographyDataset1
  --CheckPointPath CHECKPOINTPATH
                        Path to save Checkpoints, Default: ../Checkpoints/
  --ModelType MODELTYPE
                        Model type, Supervised or Unsupervised? Choose from Sup and Unsup, Default:Sup
  --NumEpochs NUMEPOCHS
                        Number of Epochs to Train for, Default:50
  --DivTrain DIVTRAIN   Factor to reduce Train data by per epoch, Default:1
  --MiniBatchSize MINIBATCHSIZE
                        Size of the MiniBatch to use, Default:512
  --LoadCheckPoint LOADCHECKPOINT
                        Load Model from latest Checkpoint from CheckPointsPath?, Default:0
  --LogsPath LOGSPATH   Path to save Logs for Tensorboard, Default=Logs/
  --LogDir LOGDIR       name of the log file
  --LR LR               Learning Rate
  --WD WD               Weight Decay
  --GradientClip GRADIENTCLIP
                        Gradient Clipping
  --Msg MSG             message
```

Example to Train the supervised model:
```bash
python Train.py --DatasetPath HomographyDataset --LogDir TestLogs --Msg "supervised" --NumEpochs 100
```

Example to Train the unsupervised model:

```bash
python Train.py --DatasetPath HomographyDataset --LogDir TestLogs --Msg "unsupervised trial" --NumEpochs 50 --ModelType Unsup --MiniBatchSize 256 --LR 0.0001
```


Panaroma Stitching using Deep Learning models

```bash
python Wrapper.py --ImageSet <IMAGESET_NAME> --ModelType <MODEL_TYPE> --CheckpointPath <PATH_TO_CHECKPOINT>

usage: Wrapper.py [-h] [--ImageSet IMAGESET] [--ModelType MODELTYPE] [--CheckpointPath CHECKPOINTPATH]

options:
  -h, --help            show this help message and exit
  --ImageSet IMAGESET   Choose the set to run the test on Options are Set1, Set2, Set3, CustomSet1, CustomSet2, Default:Set1
  --ModelType MODELTYPE
                        Model type, Supervised or Unsupervised? Choose from Sup and Unsup, Default:Sup
  --CheckpointPath CHECKPOINTPATH
                        checkpoint path to load model weights.
```

Exmaple to use the model:
```bash
python Wrapper.py --ImageSet Set2 --ModelType Sup --CheckpointPath TestLogs
```


