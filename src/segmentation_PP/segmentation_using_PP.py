""" https://programmer.group/5e8098ca6095e.html """
# Read the original drawing
import cv2
import cv2 as cv
import numpy as np
import preprocessing
import os


# Horizontal projection
def hProject(binary):
    h, w = binary.shape
    # Horizontal projection
    hprojection = np.zeros(binary.shape, dtype=np.uint8)
    # Create an array with h length of 0
    h_h = [0] * h
    for j in range(h):
        for i in range(w):
            if binary[j, i] == 0:
                h_h[j] += 1

    # Draw a projection
    for j in range(h):
        for i in range(h_h[j]):
            hprojection[j, i] = 255
    return h_h


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


def segment(image, page_num):
    # Character recognition matching and segmentation
    test_data = image
    #test_data = cv.cvtColor(test_data, cv.COLOR_BGR2GRAY)
    h, w = test_data.shape
    h_h = hProject(test_data)
    start = 0
    h_start, h_end = [], []
    position = []

    cnt = 0
    for i in range(len(h_h)):
        print(h_h[i])
        if h_h[i] > 5 and start == 0:
            h_start.append(i)
            print("start {}".format(i))
            start = 1
        if h_h[i] <= 25 and start == 1 and cnt >= 30:
            h_end.append(i)
            start = 0
            print("end {}".format(i))
            cnt = 0
        cnt += 1

    print(h_start)
    print(h_end)
    cnt_i = 0
    cnt_j = 0
    for i in range(len(h_start)):
        cropImg = test_data[h_start[i]:h_end[i], 0:w]
        cv.imwrite('segmented_data/lines/line' + str(i) + '.jpg', cropImg)
        if i == 0:
            pass
        w_w = vProject(cropImg)
        wstart, wend, w_start, w_end = 0, 0, 0, 0
        crops = []
        cnt = 0
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
                position.append([w_start, h_start[i], w_end, h_end[i]])
                crops.append([w_start, w_end])
                wend = 0
            cnt += 1
        crops.reverse()
        cnt_j=0
        for w1, w2 in (crops):
            crop_word = test_data[h_start[i]:h_end[i], w1:w2]
            newpath = r'segmented_data\page_' + str(page_num)
            if not os.path.exists(newpath):
                os.makedirs(newpath)
            cv.imwrite(newpath + '/line' + str(cnt_i) + '_word' + str(cnt_j) + '.jpg', crop_word)
            cnt_j += 1
        cnt_i += 1


images = preprocessing.load_images_from_folder('Dataset_for_test')
k = 1
for img in images:
    img = preprocessing.Filters(img)
    #img = preprocessing.Remove_borders(img)
    cv2.imshow('Image',img)
    cv2.waitKey(0)
    segment(img,k)
    k +=1
