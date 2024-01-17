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


def normalize_image(image):
    # Normalize the image with min value of 0 and max value of 255
    image = image - np.min(image)
    image = image / np.max(image)
    image = image * 255
    return image.astype(np.float32)

def read_images(data_dir, train=True, **kwargs):

    images = []

    image_path = osp.join(data_dir, "Train" if train else "Test", kwargs["ImageSet"])

    for image_name in os.listdir(image_path):
        image = cv2.imread(osp.join(image_path, image_name))
        if image is not None:
            image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            images.append([osp.join(image_path, image_name), image, image_grayscale.astype(np.float32)])

    return images

def detect_corners(image, results_dir_path, method="Harris", plot=True, **kwargs):

    image_name = image[0].split("/")[-1]
    if method == "Harris":
        print("Detecting corners with Harris... for image: ", image_name)

        # Apply Harris Corner Detection
        corner_det_image = cv2.cornerHarris(image[2], blockSize=7, ksize=11, k=0.04)

        if plot:
            _img = image[1].copy()
            corners = np.argwhere(corner_det_image > 0.01 * corner_det_image.max())

            # Draw corners on the original image
            for y, x in corners:
                cv2.drawMarker(_img, (x, y), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 10, 1)

            # Save the corner detection output
            corner_image_path = osp.join(results_dir_path, f"corners_harris_{image_name}")
            cv2.imwrite(corner_image_path, _img)


    # Just a function to detect corners with Shi-Tomasi
    # Not to be used for image stitching
    elif method == "ShiTomasi":
        print("Detecting corners with Shi-Tomasi... for image: ", image_name)
        num_corners = kwargs["NumFeatures"]
        corners = cv2.goodFeaturesToTrack(image[2], num_corners, 0.01, 10)
        corner_det_image = np.zeros_like(image[2])

        if plot:
            _img = image[1].copy()
            if corners is not None:
                    # Draw corners on the original image
                    for i in corners:
                        x, y = i.ravel()
                        cv2.drawMarker(_img, (int(x), int(y)), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 10, 1)
                        corner_det_image[int(y), int(x)] = 1

            # Save the Shi-Tomasi corner detection output
            shi_tomasi_image_path = osp.join(results_dir_path, f"corners_shi_tomasi_{image_name}")
            cv2.imwrite(shi_tomasi_image_path, _img)

    return corner_det_image


def anms(c_img, image, results_dir_path, n_best=150, plot=True, min_distance=3, threshold=0.001):

    image_name = image[0].split("/")[-1]
    coordinates = peak_local_max(c_img, min_distance=min_distance, threshold_rel=threshold)

    _r = np.ones(len(coordinates)) * np.inf
    _ED = None

    for i in range(len(coordinates)):
        for j in range(len(coordinates)):
            if(c_img[coordinates[i][0], coordinates[i][1]] < c_img[coordinates[j][0], coordinates[j][1]]):
                _ED = (coordinates[i][0] - coordinates[j][0])**2 + (coordinates[i][1] - coordinates[j][1])**2

            if _ED is not None and _ED < _r[i]:
                _r[i] = _ED

    best_corners_idx = np.argsort(_r)[::-1][:n_best]
    best_corners = coordinates[best_corners_idx]

    _img = image[1].copy()
    for corner in best_corners:
        cv2.drawMarker(_img, (corner[1], corner[0]), (255, 0, 0), cv2.MARKER_TILTED_CROSS, 10, 1)
        cv2.imwrite(osp.join(results_dir_path, f"ANMS_{image_name}"), _img)

    return best_corners

def get_feature_descriptors(image, keypoints, results_dir_path, **kwargs):

    _img = image[1].copy()
    _img = cv2.copyMakeBorder(image[1], 20, 20, 20, 20, cv2.BORDER_CONSTANT, None, value=0) # padding
    patch = np.zeros((41, 41, 3))
    feature_descriptors = []

    for point in keypoints:
        patch = _img[point[0]:point[0]+40, point[1]:point[1]+40, :].copy()
        blurred_patch = cv2.GaussianBlur(patch, (5, 5), 0)
        subsampled_patch = cv2.resize(blurred_patch, (8, 8))
        subsampled_vector = subsampled_patch.flatten()
        normalized_vector = (subsampled_vector-np.mean(subsampled_vector))/np.std(subsampled_vector)
        feature_descriptors.append(normalized_vector)

    return feature_descriptors


def main():
    # Add any Command Line arguments here
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--NumFeatures', default=5000, type=int, help='Number of best features to extract from each image, Default:100')
    Parser.add_argument('--SelectTrain', action='store_false', help='Choose the set to run the test on, Default:True')
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

    # Store the images in a list of [[image_path, original_image, grayscale_image], ...]
    images = read_images(DATA, SelectTrain, ImageSet=ImageSet)
    keypoints = []
    feature_descriptors = []

    # Sanity check
    for image in images: # image is a list of [image_path, image, image_grayscale]
        """
        Corner Detection
        Save Corner detection output as corners.png
        """
        corner_det_image = detect_corners(image, results_dir_path, method=SelectDetector, plot=True, NumFeatures=NumFeatures)
        # TODO: Apply threshold relative to standard deviation
        # Currently, the threshold is 0.01 * max value of the standardized corner detection output

        """
        Perform ANMS: Adaptive Non-Maximal Suppression
        Save ANMS output as anms.png
        """
        keypoints_ = anms(corner_det_image, image, results_dir_path, plot=True, min_distance=9)
        keypoints.append(keypoints_)

        """
        Feature Descriptors
        Save Feature Descriptor output as FD.png
        """
        feature_descriptors_ = get_feature_descriptors(image, keypoints_, results_dir_path)
        feature_descriptors.append(feature_descriptors_)


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
