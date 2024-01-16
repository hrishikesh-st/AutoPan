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
    images = []

	# Initialize the flag for data selection and select set of images
    # train = True
    # image_set = "Set1" # Options are Set1, Set2, Set3, CustomSet1, CustomSet2
    image_path = osp.join(DATA, "Train" if SelectTrain else "Test", ImageSet)

    # Create directory to save the results
    results_dir_path = osp.join(DATA, "Results")
    if not osp.exists(results_dir_path):
        os.makedirs(results_dir_path)

	# Read and save data in grayscale
    for image_name in os.listdir(image_path):
        # Read image
        image = cv2.imread(osp.join(image_path, image_name))
        if image is not None:
            image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            images.append([image_name, image_grayscale.astype(np.float32)])

    """
	Corner Detection
	Save Corner detection output as corners.png
	"""
    # Sanity check
    for image in images: # image is a list of [image_name, image_grayscale]

        og_image = cv2.imread(osp.join(image_path, image[0]))
        """"    
        # Sanity Checks
        print(f"Shape for {image[0]} is {image[1].shape}")
        print(f"Type for {image[0]} is {type(image[1])} with dtype {image[1].dtype}")
        print("*" * 33)
        """
        if SelectDetector == "Harris":
            print("Detecting corners using Harris Corner Detector")
            # Apply Harris Corner Detection
            corner_det_image = cv2.cornerHarris(image[1], blockSize=7, ksize=11, k=0.04)
            
            """
            # Save the corner detection output for visualization
            cv2.imwrite(osp.join(results_dir_path, f"corners_{image[0]}"), corner_det_image)
            """

            corner_det_image = cv2.dilate(corner_det_image, None)

            # Read the original image for plotting corners
            og_image[corner_det_image > 0.01 * corner_det_image.max()] = [0, 0, 255]

            # Save the corner detection output
            corner_image_path = osp.join(results_dir_path, f"corners_harris_{image[0]}")
            cv2.imwrite(corner_image_path, og_image)


        elif SelectDetector == "ShiTomasi":
            print("Detecting corners using Shi-Tomasi Corner Detector")
            # Shi-Tomasi Corner Detection
            # Specify the number of corners, Default: 100
            num_corners = NumFeatures 
            corners = cv2.goodFeaturesToTrack(image[1], num_corners, 0.01, 10)

            if corners is not None:
                # Draw corners on the original image
                for i in corners:
                    x, y = i.ravel()
                    cv2.circle(og_image, (int(x), int(y)), 3, (0, 255, 0), -1)  # Ensure coordinates are integers

            # Save the Shi-Tomasi corner detection output
            shi_tomasi_image_path = osp.join(results_dir_path, f"corners_shi_tomasi_{image[0]}")
            cv2.imwrite(shi_tomasi_image_path, og_image)


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
