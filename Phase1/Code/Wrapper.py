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
        for j in range(i+1, len(coordinates)):
            if(c_img[coordinates[i][0], coordinates[i][1]] < c_img[coordinates[j][0], coordinates[j][1]]):
                _ED = (coordinates[i][0] - coordinates[j][0])**2 + (coordinates[i][1] - coordinates[j][1])**2

            if _ED is not None and _ED < _r[i]:
                _r[i] = _ED

    best_corners_idx = np.argsort(_r)[::-1][:n_best]
    best_corners = coordinates[best_corners_idx]
    keypoints = []

    _img = image[1].copy()
    for corner in best_corners:
        keypoints.append(cv2.KeyPoint(float(corner[1]), float(corner[0]), 1))
        cv2.drawMarker(_img, (corner[1], corner[0]), (255, 0, 0), cv2.MARKER_TILTED_CROSS, 10, 1)
        cv2.imwrite(osp.join(results_dir_path, f"ANMS_{image_name}"), _img)

    return keypoints

def get_feature_descriptors(image, keypoints, results_dir_path, **kwargs):

    _img = image[2].copy()
    _img = cv2.copyMakeBorder(image[2], 20, 20, 20, 20, cv2.BORDER_CONSTANT, None, value=0) # padding
    patch = np.zeros((41, 41))
    feature_descriptors = []

    for point in keypoints:
        patch = _img[int(point.pt[1]):int(point.pt[1])+41, int(point.pt[0]):int(point.pt[0])+41].copy()
        blurred_patch = cv2.GaussianBlur(patch, (5, 5), 0)
        subsampled_patch = cv2.resize(blurred_patch, (8, 8))
        subsampled_vector = subsampled_patch.flatten()
        normalized_vector = (subsampled_vector-np.mean(subsampled_vector))/np.std(subsampled_vector)
        feature_descriptors.append(normalized_vector)

    return feature_descriptors

def match_features(kp1, fd1, kp2, fd2):

    matches = []
    mapping = []

    for i, f1 in enumerate(fd1):
        _best_dist_1 = np.inf
        _best_dist_2 = np.inf
        _idx = 0

        for j, f2 in enumerate(fd2):
            _dist = np.linalg.norm(f1-f2)
            if _dist < _best_dist_1:
                _best_dist_2 = _best_dist_1
                _best_dist_1 = _dist
                _idx = j
            elif _dist < _best_dist_2:
                _best_dist_2 = _dist

        if _best_dist_1/_best_dist_2 < 0.75:
            matches.append(cv2.DMatch(i, _idx, _best_dist_1))
            mapping.append([kp1[i].pt[0], kp1[i].pt[1], kp2[_idx].pt[0], kp2[_idx].pt[1], i, _idx, _best_dist_1])

    return matches, mapping

def find_homography(pairs):

    A = []
    for pair in pairs:
        x1, y1, x2, y2, _, _, _ = pair
        A.append([x1, y1, 1, 0, 0, 0, -x2*x1, -x2*y1, -x2])
        A.append([0, 0, 0, x1, y1, 1, -y2*x1, -y2*y1, -y2])

    A = np.array(A)

    U, S, V = np.linalg.svd(A) # Single value decomposition
    H = np.reshape(V[-1], (3, 3))
    # H = (1 / H.item(8)) * H

    return H

def ransac(mapping, tau, n_max=1000):

    inliers = []
    inlier_matches = []
    for i in range(n_max):
        pairs = [mapping[i] for i in np.random.choice(len(mapping), 4)]
        H = find_homography(pairs)

        temp_inliers = []
        temp_matches = []
        for pair in mapping:
            x1, y1, x2, y2, i, j, _dist = pair
            _H = H.flatten().tolist()
            x2_hat = _H[0]*x1 + _H[1]*y1 + _H[2]
            y2_hat = _H[3]*x1 + _H[4]*y1 + _H[5]
            z2_hat = _H[6]*x1 + _H[7]*y1 + _H[8]

            if abs(x2_hat/(z2_hat+1e-6) - x2) + abs(y2_hat/(z2_hat+1e-6) - y2) < tau: # TODO: Check logic
                temp_inliers.append([x1, y1, x2, y2, i, j, _dist])
                temp_matches.append(cv2.DMatch(i, j, _dist))

        if len(temp_inliers) > len(inliers):
            inliers = temp_inliers
            inlier_matches = temp_matches

        if len(inliers) >= 0.9*len(mapping):
            break

    H_hat = find_homography(inliers)

    return H_hat, inliers, inlier_matches

def warpTwoImages(img1, img2, H):
    '''warp img2 to img1 with homograph H'''
    h1,w1 = img1.shape[:2]
    h2,w2 = img2.shape[:2]
    pts1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
    pts2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
    pts2_ = cv2.perspectiveTransform(pts2, H)
    pts = np.concatenate((pts1, pts2_), axis=0)
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
    t = [-xmin,-ymin]
    Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) # translate

    result = cv2.warpPerspective(img2, Ht.dot(H), (xmax-xmin, ymax-ymin))
    result[t[1]:h1+t[1],t[0]:w1+t[0]] = img1
    return result

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
    # [(a, b) for idx, a in enumerate(test_list) for b in test_list[idx + 1:]]

    matches, mapping = match_features(keypoints[0], feature_descriptors[0], keypoints[1], feature_descriptors[1])
    _matched_img = cv2.drawMatches(images[0][1], keypoints[0], images[1][1], keypoints[1], matches, None, matchColor=(0, 255, 255), flags=2)
    cv2.imwrite(osp.join(results_dir_path, "matching.png"), _matched_img)

    """
	Refine: RANSAC, Estimate Homography
	"""
    H, inliers, inlier_matches = ransac(mapping, tau=50, n_max=2000)
    _matched_img = cv2.drawMatches(images[0][1], keypoints[0], images[1][1], keypoints[1], inlier_matches, None, matchColor=(0, 255, 0), flags=2)
    cv2.imwrite(osp.join(results_dir_path, "matching_inliers.png"), _matched_img)

    """
	Image Warping + Blending
	Save Panorama output as mypano.png
	"""
    img = warpTwoImages(images[1][1], images[0][1], H)
    cv2.imwrite(osp.join(results_dir_path, "mypano.png"), img)


if __name__ == "__main__":
    main()
