#!/usr/bin/evn python

"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""


# Code starts here:

import os
import cv2
import torch
import kornia
import argparse
import numpy as np
from natsort import natsorted

from Network.Network import HomographyModel

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Add any python libraries here

def read_images(data_dir, set):

    images = []
    image_path = os.path.join(data_dir, set)

    for image_name in natsorted(os.listdir(image_path)):
        image = cv2.imread(os.path.join(image_path, image_name))
        if image is not None:
            images.append(image)

    return images

def find_homography(self, c_a, c_b):

    A = []
    for i in range(0, 8, 2):
        x1, y1 = c_a[i], c_a[i+1]
        x2, y2 = c_b[i], c_b[i+1]
        A.append([x1, y1, 1, 0, 0, 0, -x2*x1, -x2*y1, -x2])
        A.append([0, 0, 0, x1, y1, 1, -y2*x1, -y2*y1, -y2])

    A = np.array(A)

    U, S, V = np.linalg.svd(A) # Single value decomposition
    H = np.reshape(V[-1], (3, 3))
    H = (1 / H.item(8)) * H

    return H

def warp_two_images(image1, image2, H):

    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    p1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    p2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    p2_ = cv2.perspectiveTransform(p2, H)
    p = np.concatenate((p1, p2_), axis=0)

    [xmin, ymin] = np.int32(p.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(p.max(axis=0).ravel() + 0.5)
    t = [-xmin, -ymin]
    Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])  # translate

    result = cv2.warpPerspective(image2, Ht.dot(H), (xmax-xmin, ymax-ymin))
    result[t[1]:h1 + t[1], t[0]:w1 + t[0]] = image1

    return result


def TensorDLT(c_a, c_b):

    H = torch.tensor([]).to(DEVICE)
    A = []

    for i in range(0, 8, 2):
        x1, y1 = c_a[i], c_a[i+1]
        x2, y2 = c_b[i], c_b[i+1]

        A.append([x1, y1, 1, 0, 0, 0, -x2*x1, -x2*y1, -x2])
        A.append([0, 0, 0, x1, y1, 1, -y2*x1, -y2*y1, -y2])

    A = torch.tensor(A).to(DEVICE)
    _, _, V = torch.linalg.svd(A)
    h = V[-1]/V[-1][8]

    H = torch.cat((H, torch.unsqueeze(h.reshape(3, -1), dim=0)), axis=0)

    return H

def main():
    # Add any Command Line arguments here
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--ImageSet', default="TestSet1", help='Choose the set to run the test on Options are Set1, Set2, Set3, CustomSet1, CustomSet2, Default:Set1')
    Parser.add_argument("--ModelType", default="Sup", help="Model type, Supervised or Unsupervised? Choose from Sup and Unsup, Default:Sup")
    Parser.add_argument('--CheckpointPath', default='28Jan_2', help='checkpoint path to load model weights.')

    Args = Parser.parse_args()
    ImageSet = Args.ImageSet
    model_path = os.path.join('Logs', Args.CheckpointPath, 'model')

    results_dir = os.path.join("../Results", ImageSet)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    if Args.ModelType == 'Sup': input_channels = 6
    elif Args.ModelType == 'Unsup': input_channels = 2
    model = HomographyModel(input_channels).to(DEVICE)
    model.load_state_dict(torch.load(model_path+'/best_model.pt'))
    model.eval()

    """
    Read a set of images for Panorama stitching
    """
    images = read_images("../Data/P1TestSet/Phase1", ImageSet)

    pair = [images[0], images[1]]
    _idx = 1
    _iter = 0

    while True:

        _iter += 1

        save_path = os.path.join(results_dir, "Iter_"+str(_iter))
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        pa_resized = cv2.resize(pair[0][0:1000, 0:1000], (128, 128))
        pb_resized = cv2.resize(pair[1][0:1000, 0:1000], (128, 128))

        """
        Obtain Homography using Deep Learning Model (Supervised and Unsupervised)
        """

        patches_batch = []
        if Args.ModelType == 'Unsup':
            stacked_patches = np.float32(np.concatenate([pa_resized, pb_resized], axis=0))
            patches_batch.append(torch.from_numpy(stacked_patches).to(DEVICE))
        else:
            stacked_patches = np.float32(np.concatenate([pa_resized, pb_resized], axis=2))
            patches_batch.append(torch.from_numpy(stacked_patches).permute(2, 0, 1).to(DEVICE))
        input = torch.stack(patches_batch)
        h4pt = model.model(input)

        c_a = torch.Tensor([0, 0, 0, 128, 128, 128, 128, 0]).to(DEVICE)
        if Args.ModelType == 'Unsup': c_b = c_a + h4pt*32
        else: c_b = c_a + h4pt

        H = TensorDLT(c_a, c_b.squeeze())

        """
        Image Warping + Blending
        Save Panorama output as mypano.png
        """
        result = warp_two_images(pa_resized, pb_resized, H.squeeze().detach().cpu().numpy())
        cv2.imwrite(os.path.join(results_dir, "mypano.png"), result)

        _idx += 1
        if _idx == len(images): break
        pair = [result, images[_idx]]


if __name__ == "__main__":
    main()
