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
import argparse
from skimage.feature import peak_local_max
from natsort import natsorted

def scale_image(image):
    # Sclaing the image to -1 to 1
    image = image / np.max(np.abs(image))
    return image.astype(np.float32)

def read_images(data_dir, set, train=True):

    images = []
    image_path = os.path.join(data_dir, "Train" if train else "Test", set)

    for image_name in natsorted(os.listdir(image_path)):
        image = cv2.imread(os.path.join(image_path, image_name))
        if image is not None:
            images.append(image)

    return images

def detect_corners(image, save_path, image_name):

    # Apply Harris Corner Detection
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corner_det_image = cv2.cornerHarris(image_gray, blockSize=7, ksize=11, k=0.06)
    corner_det_image = scale_image(corner_det_image)

    _img = image.copy()
    corners = np.argwhere(corner_det_image > 0.0075 * corner_det_image.max())

    for y, x in corners:
        cv2.drawMarker(_img, (x, y), (0, 255, 0), cv2.MARKER_TILTED_CROSS, 15   , 2)

    # Save the corner detection output
    cv2.imwrite(os.path.join(save_path, "corners_harris_"+image_name+".png"), _img)

    return corner_det_image


def anms(c_img, image, save_path, image_name, n_best=200, min_distance=3, threshold=0.01):
    coordinates = peak_local_max(c_img, min_distance=min_distance, threshold_rel=threshold)

    _img_local_maxima = image.copy()
    # Plot the peak local maximas
    for y, x in coordinates:
        cv2.drawMarker(_img_local_maxima, (x, y), (0, 255, 0), cv2.MARKER_TILTED_CROSS, 15, 2)
    cv2.imwrite(os.path.join(save_path, "anms_local_maxima_"+image_name+".png"), _img_local_maxima) 


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
    keypoints = []

    _img = image.copy()
    for corner in best_corners:
        keypoints.append(cv2.KeyPoint(float(corner[1]), float(corner[0]), 1))
        cv2.drawMarker(_img, (corner[1], corner[0]), (0, 255, 0), cv2.MARKER_TILTED_CROSS, 15, 2)
    cv2.imwrite(os.path.join(save_path, "ANMS_"+image_name+".png"), _img)

    return keypoints

def get_feature_descriptors(image, keypoints):

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _img = cv2.copyMakeBorder(image_gray, 20, 20, 20, 20, cv2.BORDER_CONSTANT, None, value=0) # padding
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
        _best_match = 0

        for j, f2 in enumerate(fd2):
            _dist = np.linalg.norm(f1-f2)
            if _dist < _best_dist_1:
                _best_dist_2 = _best_dist_1
                _best_dist_1 = _dist
                _best_match = j
            elif _dist < _best_dist_2:
                _best_dist_2 = _dist

        if _best_dist_1/_best_dist_2 < 0.75:
            matches.append(cv2.DMatch(i, _best_match, _best_dist_1))
            mapping.append([kp1[i].pt[0], kp1[i].pt[1], kp2[_best_match].pt[0], kp2[_best_match].pt[1], i, _best_match, _best_dist_1])

    return matches, mapping, 

def find_homography(pairs):

    A = []
    for pair in pairs:
        x1, y1, x2, y2, _, _, _ = pair
        A.append([x1, y1, 1, 0, 0, 0, -x2*x1, -x2*y1, -x2])
        A.append([0, 0, 0, x1, y1, 1, -y2*x1, -y2*y1, -y2])

    A = np.array(A)

    U, S, V = np.linalg.svd(A) # Single value decomposition
    H = np.reshape(V[-1], (3, 3))
    H = (1 / H.item(8)) * H # TODO: Check logic

    return H

def ransac(mapping, tau, n_max=1000):

    inliers = []
    inlier_matches = []
    valid_stitch = True

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
    if len(inliers) < 4:
        valid_stitch = False

    return H_hat, inliers, inlier_matches, valid_stitch


def warp_two_images(image1, image2, H, save_path, alpha=0.8, blending=False, poisson=False):

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
    

    if blending:
        if poisson:
            # Define the mask for seamlessClone
            mask = 255 * np.ones_like(image1, dtype=np.uint8)
            
            # Find the overlapping region of the two images and define the mask
            cv2.imwrite(os.path.join(save_path, "mask.png"), mask)

            # Perform Poisson blending
            result = cv2.seamlessClone(image1, result, mask, (int(t[0] + w1/2), int(t[1] + h1/2)), cv2.NORMAL_CLONE)
            # Visualize the center of the image
            # cv2.drawMarker(result, (int(t[0] + w1 / 2), int(t[1] + h1 / 2)), (0, 255, 0), cv2.MARKER_TILTED_CROSS, 15, 2)

        else:
            # Weighted Alpha blending
            for y in range(h1):
                for x in range(w1):
                    # Calculate the position in the result image
                    result_y = y + t[1]
                    result_x = x + t[0]

                    # Ensure the pixel is within the bounds of the result image
                    if 0 <= result_x < result.shape[1] and 0 <= result_y < result.shape[0]:
                        # Blend the pixel values using weighted alpha
                        result[result_y, result_x] = alpha * image1[y, x] + (1 - alpha) * result[result_y, result_x]
    else:
        result[t[1]:h1 + t[1], t[0]:w1 + t[0]] = image1

    return result

def main():
    # Add any Command Line arguments here
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--Train', action='store_true', help='Choose the set to run the test on, Default:True')
    Parser.add_argument('--ImageSet', default="Set1", help='Choose the set to run the test on Options are Set1, Set2, Set3, CustomSet1, CustomSet2, Default:Set1')
    Parser.add_argument('-p', '--Poisson', action='store_true', help='Choose whether to use Poisson blending or not, Default:False')
    Parser.add_argument('-b', '--blending', action='store_true', help='Choose whether to use Blending or not, Default:False')

    Args = Parser.parse_args()
    Train = Args.Train
    ImageSet = Args.ImageSet
    Poisson = Args.Poisson
    blending = Args.blending

    results_dir = os.path.join("../Results", ImageSet)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    """
    Read a set of images for Panorama stitching
    """
    print("Reading images...")
    images = read_images("../Data", ImageSet, Train)

    pair = [images[0], images[1]]
    # pair = [images[1], images[0]]
    _idx = 1
    _iter = 0

    while(True):

        _iter += 1
        keypoints = []
        feature_descriptors = []

        save_path = os.path.join(results_dir, "Iter_"+str(_iter))
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        print("Iteration "+str(_iter))

        for i, img in enumerate(pair):
            """
            Corner Detection
            Save Corner detection output as corners.png
            """
            print("Corner detection..."+str(i))
            c_img = detect_corners(img, save_path, str(i))
            # TODO: Apply threshold relative to standard deviation
            # Currently, the threshold is 0.01 * max value of the standardized corner detection output

            """
            Perform ANMS: Adaptive Non-Maximal Suppression
            Save ANMS output as anms.png
            """
            print("ANMS..."+str(i))
            keypoints_ = anms(c_img, img, save_path, str(i), min_distance=9)
            keypoints.append(keypoints_)

            """
            Feature Descriptors
            Save Feature Descriptor output as FD.png
            """
            print("Feature Descriptors..."+str(i))
            feature_descriptors_ = get_feature_descriptors(img, keypoints_)
            feature_descriptors.append(feature_descriptors_)

        """
        Feature Matching
        Save Feature Matching output as matching.png
        """
        print("Feature Matching..."+str(_iter))

        matches, mapping = match_features(keypoints[0], feature_descriptors[0], keypoints[1], feature_descriptors[1])
        _matched_img = cv2.drawMatches(pair[0], keypoints[0], pair[1], keypoints[1], matches, None, matchColor=(0, 255, 255), flags=2)
        cv2.imwrite(os.path.join(save_path, "matching.png"), _matched_img)

        """
        Refine: RANSAC, Estimate Homography
        """
        print("RANSAC..."+str(_iter))
        H, inliers, inlier_matches, valid_stitch = ransac(mapping, tau=55, n_max=1000)
        _matched_img = cv2.drawMatches(pair[0], keypoints[0], pair[1], keypoints[1], inlier_matches, None, matchColor=(0, 255, 0), flags=2)
        cv2.imwrite(os.path.join(save_path, "matching_inliers.png"), _matched_img)

        if not valid_stitch:
            print("Not a valid stitch. Moving to the next image in sequence...")
            _idx += 1
            if _idx == len(images): break
            pair = [result, images[_idx]]
            continue

        """
        Image Warping + Blending
        Save Panorama output as mypano.png
        """
        print("Warping..."+str(_iter))
        result = warp_two_images(pair[1], pair[0], H, save_path, blending=blending, poisson=Poisson)
        cv2.imwrite(os.path.join(save_path, "mypano.png"), result)

        _idx += 1
        if _idx == len(images): break
        pair = [result, images[_idx]]
        # pair = [images[_idx], result]


if __name__ == "__main__":
    main()
