import numpy as np
import cv2
import os
import shutil
import random


def homographyAugmentation(img, random_limits = (0.9, 1.1)):
        '''
        Creates an augmentation by computing a homography from three
        points in the image to three randomly generated points
        '''
        y, x = img.shape[:2]
        fx = float(x)
        fy = float(y)
        src_point = np.float32([[fx / 2, fy / 3, ],
                                [2 * fx / 3, 2 * fy / 3],
                                [fx / 3, 2 * fy / 3]])
        random_shift = (np.random.rand(3, 2) - 0.5) * 2 * (random_limits[1] - random_limits[0]) / 2 + np.mean(
            random_limits)
        dst_point = src_point * random_shift.astype(np.float32)
        transform = cv2.getAffineTransform(src_point, dst_point)
        # border_value = 0
        if img.ndim == 3:
            border_value = np.median(np.reshape(img, (img.shape[0] * img.shape[1], -1)), axis=0)
        else:
            border_value = np.median(img)
        warped_img = cv2.warpAffine(img, transform, dsize=(x, y), borderValue=border_value.astype(float))
        return warped_img


folder = 'segdata'
for classfolder in os.listdir(folder):
    images = os.listdir(folder + '/' + classfolder)
    length = len(images)
    if 0 < length <= 20:
        for i in range(20):
            randIndex = random.randint(1, length)
            randIndex = randIndex - 1
            imgName = str(images[randIndex])
            img = cv2.imread(folder + '/' + classfolder + '/' + imgName)
            img = homographyAugmentation(img=img)
            cv2.imwrite(folder + '/' + classfolder + '/' + "augmented_" + str(i) + '.png', img)
