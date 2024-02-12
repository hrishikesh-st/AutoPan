import os
import cv2
import json
import numpy as np

from tqdm import tqdm
from natsort import natsorted

DATA_PATH = '../../Data/MSCOCO'
SAVE_PATH = '../../Data/TestSet'


class DataGenerator:

    def __init__(self, P_A_path, P_B_path, label_path, images_path, patch_size=128, max_perturbation=32, max_translation=32, buffer=25, patches_per_image=1):

        self.P_A_path = P_A_path
        self.P_B_path = P_B_path
        self.label_path = label_path
        self.images_path = images_path

        self.patch_size = patch_size
        self.max_perturbation = max_perturbation
        self.max_translation = max_translation
        self.buffer = buffer
        self.patches_per_image = patches_per_image

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

    def generate_data(self, image, image_name):

        h, w = image.shape
        allowed_h = [self.buffer, h-(self.buffer+self.patch_size+self.max_translation)]
        allowed_w = [self.buffer, w-(self.buffer+self.patch_size+self.max_translation)]

        for _ in range(self.patches_per_image):

            x = np.random.randint(low=allowed_w[0], high=allowed_w[1])
            y = np.random.randint(low=allowed_h[0], high=allowed_h[1])
            translation = np.random.randint(self.max_translation)

            C_A = [x, y, x+self.patch_size, y, x, y+self.patch_size, x+self.patch_size, y+self.patch_size]
            C_B = []
            H4pt = []

            for i in range(0, 8, 2):
                perturbation = [np.random.randint(low=-self.max_perturbation, high=self.max_perturbation)+translation,
                                np.random.randint(low=-self.max_perturbation, high=self.max_perturbation)+translation]
                C_B.append(C_A[i]+perturbation[0])
                C_B.append(C_A[i+1]+perturbation[1])
                H4pt.append(perturbation[0])
                H4pt.append(perturbation[1])

            H = self.find_homography(C_A, C_B)
            H_inv = np.linalg.pinv(H)

            transformed_img = cv2.warpPerspective(image, H_inv, (w, h))

            P_A = image[y:y+self.patch_size, x:x+self.patch_size]
            P_B = transformed_img[y:y+self.patch_size, x:x+self.patch_size]

            label = {
                'H4pt': H4pt,
                'C_A': C_A
            }

            cv2.imwrite(os.path.join(self.P_A_path, image_name+'_'+str(_)+'.png'), P_A)
            cv2.imwrite(os.path.join(self.P_B_path, image_name+'_'+str(_)+'.png'), P_B)
            cv2.imwrite(os.path.join(self.images_path, image_name+'_tr.png'), transformed_img)
            with open (os.path.join(self.label_path, image_name+'_'+str(_)+'.txt'), 'w') as f:
                json.dump(label, f)


if __name__ == "__main__":

    org_train_data = os.path.join(DATA_PATH, 'Train')
    org_valid_data = os.path.join(DATA_PATH, 'Val')

    # TRAIN

    save_train_data = os.path.join(SAVE_PATH, 'Train')

    pa_path = os.path.join(save_train_data, 'P_A')
    if not os.path.exists(pa_path): os.makedirs(pa_path)

    pb_path = os.path.join(save_train_data, 'P_B')
    if not os.path.exists(pb_path): os.makedirs(pb_path)

    labels_path = os.path.join(save_train_data, 'labels')
    if not os.path.exists(labels_path): os.makedirs(labels_path)

    images_path = os.path.join(save_train_data, 'images')
    if not os.path.exists(images_path): os.makedirs(images_path)

    generator = DataGenerator(pa_path, pb_path, labels_path, images_path)

    for i, img_path in tqdm(enumerate(natsorted(os.listdir(org_train_data)))):

        img_gray = cv2.imread(os.path.join(org_train_data, img_path), 0)
        img_resized = cv2.resize(img_gray, (320, 240))

        generator.generate_data(img_resized, str(i+1))

    # VALID

    save_valid_data = os.path.join(SAVE_PATH, 'Val')

    pa_path = os.path.join(save_valid_data, 'P_A')
    if not os.path.exists(pa_path): os.makedirs(pa_path)

    pb_path = os.path.join(save_valid_data, 'P_B')
    if not os.path.exists(pb_path): os.makedirs(pb_path)

    labels_path = os.path.join(save_valid_data, 'labels')
    if not os.path.exists(labels_path): os.makedirs(labels_path)

    generator = DataGenerator(pa_path, pb_path, labels_path)

    for i, img_path in tqdm(enumerate(natsorted(os.listdir(org_valid_data)))):

        img_gray = cv2.imread(os.path.join(org_valid_data, img_path), 0)
        img_resized = cv2.resize(img_gray, (320, 240))

        generator.generate_data(img_resized, str(i+1))