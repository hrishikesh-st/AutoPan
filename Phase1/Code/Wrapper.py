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

import numpy as np
import cv2

# Add any python libraries here
import os
import os.path as osp
import argparse
import distutils.util
from skimage.feature import peak_local_max


def readImages(data_dir, train=True, **kwargs):
    """
    Read a set of images for Panorama stitching

    :param data_dir: DataSet directory
    :type data_dir: str
    :param train: Select Train or Test set, defaults to True
    :type train: bool, optional
    :return: List of images, format: [[image_name, original_image, image_grayscale], ...]
    :rtype: list
    """
    images = []

    image_path = osp.join(data_dir, "Train" if train else "Test", kwargs["ImageSet"])

    for image_name in os.listdir(image_path):
        image = cv2.imread(osp.join(image_path, image_name))
        if image is not None:
            image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            images.append([osp.join(image_path, image_name), image, image_grayscale.astype(np.float32)])

    return images

def detectCorners(image, results_dir_path, method="Harris", plot=True, **kwargs):
    """
    Corner Detection with Harris or Shi-Tomasi

    :param image: list of [image_path, image, image_grayscale]
    :type image: list
    :param results_dir_path: Path to save the results
    :type results_dir_path: str
    :param method: Detection method, defaults to "Harris"
    :type method: str, optional
    :param plot: Flag for plotting, defaults to True
    :type plot: bool, optional
    :return: Corner detection output
    :rtype: np.ndarray
    """
    image_name = image[0].split("/")[-1]
    if method == "Harris":
        print("Detecting corners with Harris... for image: ", image_name)
        # Apply Harris Corner Detection
        corner_det_image = cv2.cornerHarris(image[2], blockSize=7, ksize=11, k=0.04)
        corner_det_image = cv2.dilate(corner_det_image, None)

        if plot:
            image[1][corner_det_image > 0.01 * corner_det_image.max()] = [0, 0, 255]

            # Save the corner detection output
            corner_image_path = osp.join(results_dir_path, f"corners_harris_{image_name}")
            cv2.imwrite(corner_image_path, image[1])

        return corner_det_image

    elif method == "ShiTomasi":
        print("Detecting corners with Shi-Tomasi... for image: ", image_name)
        num_corners = kwargs["NumFeatures"]
        corners = cv2.goodFeaturesToTrack(image[2], num_corners, 0.01, 10)

        if plot:
            if corners is not None:
                    # Draw corners on the original image
                    for i in corners:
                        x, y = i.ravel()
                        cv2.circle(image[1], (int(x), int(y)), 3, (0, 255, 0), -1)  # Ensure coordinates are integers

            # Save the Shi-Tomasi corner detection output
            shi_tomasi_image_path = osp.join(results_dir_path, f"corners_shi_tomasi_{image_name}")
            cv2.imwrite(shi_tomasi_image_path, image[1])

        return corners


def main():
    # Add any Command Line arguments here
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--NumFeatures', default=100, type=int, help='Number of best features to extract from each image, Default:100')
    Parser.add_argument('--SelectTrain', type=lambda x: bool(distutils.util.strtobool(x)), default=True, help='Choose the set to run the test on, Default:True')
    Parser.add_argument('--ImageSet', default="Set1", help='Choose the set to run the test on Options are Set1, Set2, Set3, CustomSet1, CustomSet2, Default:Set1')
    Parser.add_argument('--SelectDetector', default="Harris", help='Choose the detector to use Options are Harris, ShiTomasi, Default:Harris')

    Args = Parser.parse_args()
    NumFeatures = Args.NumFeatures
    SelectTrain = Args.SelectTrain
    ImageSet = Args.ImageSet
    SelectDetector = Args.SelectDetector

    """
    Read a set of images for Panorama stitching
    """
    # Initialize the list of images
    DATA = "../Data"

    # Create directory to save the results
    results_dir_path = osp.join(DATA, "Results")
    if not osp.exists(results_dir_path):
        os.makedirs(results_dir_path)

    images = readImages(DATA, SelectTrain, ImageSet=ImageSet)

    """
	Corner Detection
	Save Corner detection output as corners.png
	"""
    # Sanity check
    for image in images: # image is a list of [image_path, image, image_grayscale]
        corner_det_image = detectCorners(image, results_dir_path, method=SelectDetector, plot=True, NumFeatures=NumFeatures)


    """
	Perform ANMS: Adaptive Non-Maximal Suppression
	Save ANMS output as anms.png
	"""

    """
	Feature Descriptors
	Save Feature Descriptor output as FD.png
	"""

    """
	Feature Matching
	Save Feature Matching output as matching.png
	"""

    """
	Refine: RANSAC, Estimate Homography
	"""

    """
	Image Warping + Blending
	Save Panorama output as mypano.png
	"""


if __name__ == "__main__":
    main()
