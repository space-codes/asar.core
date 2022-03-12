""" https://programmer.group/5e8098ca6095e.html """


import cv2 as cv
import numpy as np
import preprocessing
import os


# Vertical back projection
def vProject(binary):
    h, w = binary.shape
    # Vertical projection
    vprojection = np.zeros(binary.shape, dtype=np.uint8)
    # Create an array with w length of 0
    w_w = [0] * w
    for i in range(w):
        for j in range(h):
            if binary[j, i] == 0:
                w_w[i] += 1
    for i in range(w):
        for j in range(w_w[i]):
            vprojection[j, i] = 255
    return w_w


def segment(image, i):
    # Character recognition matching and segmentation
    test_data = preprocessing.Filters(image)
    #test_data = cv.cvtColor(test_data, cv.COLOR_BGR2GRAY)
    h, w = test_data.shape
    crops=[]
    cnt = 0
    wstart=0
    wend=0
    w_start=0
    w_end=0
    w_w = vProject(test_data)
    for j in range(len(w_w)):
            if w_w[j] > 0 and wstart == 0:
                w_start = j
                wstart = 1
                wend = 0
            if w_w[j] == 0 and wstart == 1 and cnt > 10:
                w_end = j
                wstart = 0
                wend = 1
            # Save coordinates when start and end points are confirmed
            if wend == 1:
                crops.append([w_start, w_end])
                wend = 0
            cnt += 1
    crops.reverse()
    cnt_j=0
    for w1, w2 in (crops):
        crop_word = image[0:h, w1:w2]
        newpath = r'segmented_data\page_' + str(1)
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        cv.imwrite(newpath + '/line' + str(i) + '_word' + str(cnt_j) + '.jpg', crop_word)
        cnt_j += 1
