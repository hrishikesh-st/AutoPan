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

def detect_corners(image, results_dir_path, method="Harris", plot=True, **kwargs):
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
        corner_det_image = cv2.cornerHarris(image[2], blockSize=3, ksize=11, k=0.04)

        # Normalize the corner detection output
        # corner_det_image = normalizeImage(corner_det_image)

        # Standardize the corner detection output
        mean = np.mean(corner_det_image)
        std = np.std(corner_det_image)
        standardized_corner_det_image = (corner_det_image - mean) / std
        # standardized_corner_det_image = normalizeImage(standardized_corner_det_image)      

        if plot:
            _img = image[1].copy()
            corners = np.argwhere(corner_det_image > 0.01 * corner_det_image.max())

            # Draw corners on the original image
            for y, x in corners:
                cv2.drawMarker(_img, (x, y), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 10, 1)
            
            # Save the corner detection output
            corner_image_path = osp.join(results_dir_path, f"corners_harris_{image_name}")
            cv2.imwrite(corner_image_path, _img)

        return corner_det_image, std, standardized_corner_det_image

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
                        # cv2.circle(_img, (int(x), int(y)), 3, (0, 0, 255), -1)  # Ensure coordinates are integers
                        corner_det_image[int(y), int(x)] = 1

            mean = np.mean(corner_det_image)
            std = np.std(corner_det_image)
            standardized_corner_det_image = (corner_det_image - mean) / std

            # Save the Shi-Tomasi corner detection output
            shi_tomasi_image_path = osp.join(results_dir_path, f"corners_shi_tomasi_{image_name}")
            cv2.imwrite(shi_tomasi_image_path, _img)

        return corner_det_image, std, standardized_corner_det_image
    
def apply_local_maxima(standardized_corner_det_image, image_path, results_dir_path, plot=True, min_distance=3, threshold=0.01):
    # (standardized_corner_det_image, image[0], results_dir_path, plot=True, min_distance=3, threshold_rel=std*0.01)
    """
    Apply Local Maxima to the output of Harris Corner Detection

    :param corner_det_image: Output from Harris Corner Detection
    :type corner_det_image: np.ndarray
    :param original_image: Original image
    :type original_image: np.ndarray
    :param results_dir_path: Path to save the results
    :type results_dir_path: str
    :param image_name: Name of the image file
    :type image_name: str
    :param min_distance: Minimum number of pixels separating peaks
    :type min_distance: int
    :param threshold_rel: Relative threshold to select the peaks
    :type threshold_rel: float
    :param plot: Flag for plotting, defaults to True
    :type plot: bool, optional
    """
    # Read the original image to plot:
    og_image = cv2.imread(image_path)
    image_name = image_path.split("/")[-1]

    print(f"Minimum distance between peaks: {min_distance}")

    # Find local maxima
    coordinates = peak_local_max(standardized_corner_det_image, min_distance=min_distance, threshold_rel=threshold)

    if plot:
        # Draw local maxima points on the original image
        for y, x in coordinates:
            cv2.drawMarker(og_image, (x, y), (0, 255, 0), cv2.MARKER_TILTED_CROSS, 20, 1)

        # Save the image with local maxima
        local_maxima_image_path = osp.join(results_dir_path, f"local_maxima_{image_name}")
        cv2.imwrite(local_maxima_image_path, og_image)

    return coordinates


def anms(c_img, image, results_dir_path, n_best=250, plot=True, min_distance=3, threshold=0.01):
    
    image_name = image[0].split("/")[-1]
    coordinates = peak_local_max(c_img, min_distance=min_distance, threshold_rel=threshold)

    _r = np.ones(len(coordinates)) * np.inf
    _ED = None

    for i in range(len(coordinates)):
        for j in range(len(coordinates)):
            if(c_img[coordinates[i][0]][coordinates[i][1]] < c_img[coordinates[j][0]][coordinates[j][1]]):
                _ED = (coordinates[i][0] - coordinates[j][0])**2 + (coordinates[i][1] - coordinates[j][1])**2
            
            if _ED is not None and _ED < _r[i]:
                _r[i] = _ED

    points = np.argsort(_r)[:n_best]

    _img = image[1].copy()
    for point in points:
        cv2.drawMarker(_img, (coordinates[point][1], coordinates[point][0]), (255, 0, 0), cv2.MARKER_TILTED_CROSS, 10, 1)
        cv2.imwrite(osp.join(results_dir_path, f"ANMS_{image_name}"), _img)

def main():
    # Add any Command Line arguments here
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--NumFeatures', default=5000, type=int, help='Number of best features to extract from each image, Default:100')
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

    # Store the images in a list of [[image_path, original_image, grayscale_image], ...]
    images = read_images(DATA, SelectTrain, ImageSet=ImageSet)

    """
	Corner Detection
	Save Corner detection output as corners.png
	"""
    # Sanity check
    for image in images: # image is a list of [image_path, image, image_grayscale]
        corner_det_image, std, standardized_corner_det_image = detect_corners(image, results_dir_path, method=SelectDetector, plot=True, NumFeatures=NumFeatures)
        # print(f"Standard deviation of corner detection output for image {image[0].split('/')[-1]}: {std}")
        # print(f"Max value of standardized corner detection output: {np.max(standardized_corner_det_image)}")
        # print(f"Min value of standardized corner detection output: {np.min(standardized_corner_det_image)}")
        # TODO: Apply threshold relative to standard deviation
        # Currently, the threshold is 0.01 * max value of the standardized corner detection output

        # coordinates = apply_local_maxima(standardized_corner_det_image, image, results_dir_path, plot=True, min_distance=5) 
        anms(standardized_corner_det_image, image, results_dir_path, plot=True, min_distance=5) 

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
