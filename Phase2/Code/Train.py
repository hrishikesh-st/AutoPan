#!/usr/bin/env python

"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""


# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)
# termcolor, do (pip install termcolor)

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torch.optim import AdamW, lr_scheduler
from Network.Network import HomographyModel
import cv2
import sys
import os
import numpy as np
import random
import skimage
import PIL
import os
import glob
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
import numpy as np
import time
from Misc.MiscUtils import *
from Misc.DataUtils import *
from Misc.LogUtils import Logger
from torchvision.transforms import ToTensor
import argparse
import shutil
import string
from termcolor import colored, cprint
import math as m
from tqdm import tqdm
import json
from natsort import natsorted

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def GenerateBatch(DatasetPath, ImageNames, MiniBatchSize, train=True):
    """
    Inputs:
    BasePath - Path to COCO folder without "/" at the end
    DirNamesTrain - Variable with Subfolder paths to train files
    NOTE that Train can be replaced by Val/Test for generating batch corresponding to validation (held-out testing in this case)/testing
    TrainCoordinates - Coordinatess corresponding to Train
    NOTE that TrainCoordinates can be replaced by Val/TestCoordinatess for generating batch corresponding to validation (held-out testing in this case)/testing
    ImageSize - Size of the Image
    MiniBatchSize is the size of the MiniBatch
    Outputs:
    I1Batch - Batch of images
    CoordinatesBatch - Batch of coordinates
    """
    images_batch = []
    H4pt_batch = []

    if train:
        BasePath = os.path.join(DatasetPath, 'Train')
    else:
        BasePath = os.path.join(DatasetPath, 'Val')

    pa_path = os.path.join(BasePath, 'P_A')
    pb_path = os.path.join(BasePath, 'P_B')
    labels_path = os.path.join(BasePath, 'labels')

    ImageNum = 0
    while ImageNum < MiniBatchSize:
        # Generate random image
        RandIdx = random.randint(0, len(ImageNames) - 1)

        RandImageName = ImageNames[RandIdx]
        ImageNum += 1

        P_A = cv2.imread(os.path.join(pa_path, RandImageName))
        P_B = cv2.imread(os.path.join(pb_path, RandImageName))
        label = json.load(open(os.path.join(labels_path, RandImageName[:-4]+'.txt')))

        stacked_image = np.float32(np.concatenate([P_A, P_B], axis=0))
        H4pt = label['H4pt']

        # Append All Images and Mask
        images_batch.append(torch.from_numpy(stacked_image).to(DEVICE))
        H4pt_batch.append(torch.tensor(H4pt).to(DEVICE))

    return torch.stack(images_batch), torch.stack(H4pt_batch)


def PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile):
    """
    Prints all stats with all arguments
    """
    print("Number of Epochs Training will run for " + str(NumEpochs))
    print("Factor of reduction in training data is " + str(DivTrain))
    print("Mini Batch Size " + str(MiniBatchSize))
    print("Number of Training Images " + str(NumTrainSamples))
    if LatestFile is not None:
        print("Loading latest checkpoint with the name " + LatestFile)


def TrainOperation(DatasetPath, DirNamesTrain, DirNamesVal, NumEpochs, MiniBatchSize,
                   CheckPointPath, LatestFile, LogsPath, ModelType,
                   LR, WD, GradientClip):
    """
    Inputs:
    ImgPH is the Input Image placeholder
    DirNamesTrain - Variable with Subfolder paths to train files
    TrainCoordinates - Coordinates corresponding to Train/Test
    NumTrainSamples - length(Train)
    ImageSize - Size of the image
    NumEpochs - Number of passes through the Train data
    MiniBatchSize is the size of the MiniBatch
    SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    CheckPointPath - Path to save checkpoints/model
    DivTrain - Divide the data by this number for Epoch calculation, use if you have a lot of dataor for debugging code
    LatestFile - Latest checkpointfile to continue training
    BasePath - Path to COCO folder without "/" at the end
    LogsPath - Path to save Tensorboard Logs
        ModelType - Supervised or Unsupervised Model
    Outputs:
    Saves Trained network in CheckPointPath and Logs to LogsPath
    """
    # Predict output with forward pass
    model = HomographyModel(ModelType=ModelType).to(DEVICE)

    ###############################################
    # Fill your optimizer of choice here!
    ###############################################
    Optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = lr_scheduler.ExponentialLR(Optimizer, gamma=0.9)

    # # Tensorboard
    # # Create a summary to monitor loss tensor
    # Writer = SummaryWriter(LogsPath)

    if LatestFile is not None:
        CheckPoint = torch.load(CheckPointPath + LatestFile + ".ckpt")
        # Extract only numbers from the name
        StartEpoch = int("".join(c for c in LatestFile.split("a")[0] if c.isdigit()))
        model.load_state_dict(CheckPoint["model_state_dict"])
        print("Loaded latest checkpoint with the name " + LatestFile + "....")
    else:
        StartEpoch = 0
        print("New model initialized....")

    _best_loss = 1e6
    NumTrainSamples = len(DirNamesTrain)
    NumValSamples = len(DirNamesVal)

    for Epochs in tqdm(range(StartEpoch, NumEpochs)):
        NumIterationsPerEpoch = int(NumTrainSamples / MiniBatchSize)
        _train_loss = 0.0
        start = time.time()
        for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):
            I1Batch, CoordinatesBatch = GenerateBatch(DatasetPath, DirNamesTrain, MiniBatchSize, train=True)

            # Predict output with forward pass
            # PredicatedCoordinatesBatch = model(I1Batch)
            # LossThisBatch = LossFn(PredicatedCoordinatesBatch, CoordinatesBatch)
            LossThisBatch = model.training_step(I1Batch, CoordinatesBatch)

            Optimizer.zero_grad()
            LossThisBatch.backward()
            Optimizer.step()

            if GradientClip:
                torch.nn.utils.clip_grad_value_(model.parameters(), GradientClip)

            _train_loss += LossThisBatch.detach().cpu().numpy()

            # Save checkpoint every some SaveCheckPoint's iterations
            # if PerEpochCounter % SaveCheckPoint == 0:
            if PerEpochCounter % 200 == 0:
                # Save the Model learnt in this epoch
                SaveName = (CheckPointPath+str(Epochs)+"a"+str(PerEpochCounter)+"model.ckpt")

                torch.save({"epoch": Epochs, "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": Optimizer.state_dict(), "loss": LossThisBatch}, SaveName)
                print("\n" + SaveName + " Model Saved...")

            # # Tensorboard
            # Writer.add_scalar(
            #     "LossEveryIter",
            #     result["val_loss"],
            #     Epochs * NumIterationsPerEpoch + PerEpochCounter,
            # )
            # # If you don't flush the tensorboard doesn't update until a lot of iterations!
            # Writer.flush()

        _train_loss /= PerEpochCounter
        logger.log(tag='train', epoch=Epochs, loss=_train_loss, time=time.time()-start)

        _val_loss = 0.0
        start = time.time()
        NumIterationsPerEpoch = int(NumValSamples/MiniBatchSize)
        for PerEpochCounter in range(NumIterationsPerEpoch):
            I1Batch, CoordinatesBatch = GenerateBatch(DatasetPath, DirNamesVal, MiniBatchSize, train=False)
            _val_loss += model.validation_step(I1Batch, CoordinatesBatch)

        _val_loss /= PerEpochCounter

        if _val_loss < _best_loss:
            _best_loss = _val_loss
            SaveName = LogsPath + '/model/best_model.pt'
            # torch.save({'epoch': Epochs,'model_state_dict': model.state_dict(),'optimizer_state_dict': Optimizer.state_dict(),'test_loss': _test_loss}, SaveName)
            torch.save(model.state_dict(), SaveName)
            print('\n' + SaveName + ' Model Saved...')
            logger.log(tag='model', loss=_val_loss)

        if Epochs % 5 == 0:
            SaveName = LogsPath + '/model/model'+str(Epochs)+'.pt'
            # torch.save({'epoch': Epochs,'model_state_dict': model.state_dict(),'optimizer_state_dict': Optimizer.state_dict(),'test_loss': _test_loss}, SaveName)
            torch.save(model.state_dict(), SaveName)
            print('\n' + SaveName + ' Model Saved...')

        logger.log(tag='val', epoch=Epochs, loss=_val_loss, time=time.time()-start)

        scheduler.step()

        # Save model every epoch
        SaveName = CheckPointPath + str(Epochs) + "model.ckpt"
        torch.save({"epoch": Epochs, "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": Optimizer.state_dict(), "loss": LossThisBatch}, SaveName)
        print("\n" + SaveName + " Model Saved...")

        logger.log(tag='plot')


def main():
    """
    Inputs:
    # None
    # Outputs:
    # Runs the Training and testing code based on the Flag
    #"""
    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument("--DatasetPath", default="",help="Base path of images, Default:")
    Parser.add_argument("--CheckPointPath", default="../Checkpoints/", help="Path to save Checkpoints, Default: ../Checkpoints/")
    Parser.add_argument("--ModelType", default="Unsup", help="Model type, Supervised or Unsupervised? Choose from Sup and Unsup, Default:Unsup")
    Parser.add_argument("--NumEpochs", type=int, default=50, help="Number of Epochs to Train for, Default:50")
    Parser.add_argument("--DivTrain", type=int, default=1, help="Factor to reduce Train data by per epoch, Default:1")
    Parser.add_argument("--MiniBatchSize", type=int, default=1, help="Size of the MiniBatch to use, Default:1")
    Parser.add_argument("--LoadCheckPoint", type=int, default=0, help="Load Model from latest Checkpoint from CheckPointsPath?, Default:0")
    Parser.add_argument("--LogsPath", default="Logs/", help="Path to save Logs for Tensorboard, Default=Logs/")
    Parser.add_argument('--LogDir', type=str, default='test_logs/', help='name of the log file')
    Parser.add_argument("--LR", type=float, default=0.001, help="Learning Rate")
    Parser.add_argument("--WD", type=float, default=0.0001, help="Weight Decay")
    Parser.add_argument("--GradientClip", type=float, default=0.1, help="Gradient Clipping")
    Parser.add_argument("--Msg", type=str, default="", help="message")

    Args = Parser.parse_args()
    NumEpochs = Args.NumEpochs
    DatasetPath = Args.DatasetPath
    DivTrain = float(Args.DivTrain)
    MiniBatchSize = Args.MiniBatchSize
    LoadCheckPoint = Args.LoadCheckPoint
    CheckPointPath = Args.CheckPointPath
    LogsPath = Args.LogsPath
    LogDir = Args.LogDir
    ModelType = Args.ModelType
    LR = Args.LR
    WD = Args.WD
    GradientClip = Args.GradientClip
    Msg = Args.Msg

    global logger
    logger = Logger(os.path.join(LogsPath, LogDir))

    logger.log(tag='args', num_epochs=NumEpochs, mini_batch_size=MiniBatchSize,
               lr=LR, weight_decay=WD, gradient_clip=GradientClip, msg=Msg)

    # Setup all needed parameters including file reading
    # (DirNamesTrain, SaveCheckPoint, ImageSize, NumTrainSamples, TrainCoordinates, NumClasses) = SetupAll(BasePath, CheckPointPath)

    train_path = os.path.join(DatasetPath, 'Train')
    val_path = os.path.join(DatasetPath, 'Val')
    DirNamesTrain = natsorted(os.listdir(os.path.join(train_path, 'P_A')))
    DirNamesVal = natsorted(os.listdir(os.path.join(val_path, 'P_A')))

    # Find Latest Checkpoint File
    if LoadCheckPoint == 1:
        LatestFile = FindLatestModel(CheckPointPath)
    else:
        LatestFile = None

    # Pretty print stats
    # PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile)

    TrainOperation(DatasetPath, DirNamesTrain, DirNamesVal, NumEpochs, MiniBatchSize,
                   CheckPointPath, LatestFile, LogsPath, ModelType,
                   LR, WD, GradientClip)


if __name__ == "__main__":
    main()
