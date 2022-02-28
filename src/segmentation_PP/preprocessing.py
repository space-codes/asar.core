
import numpy as np
import cv2
import os

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

# Normalization
def Normalization(img):
    #img = cv2.equalizeHist(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl_image = clahe.apply(img)
    return cl_image

# binarization
def binary_thresholding(img):
    # Values below 127 goes to 0 (black, everything above goes to 255 (white)
    ret,binary_th = cv2.threshold(img,200, 255, cv2.THRESH_BINARY)
    return binary_th

def adaptive_thresholding(img):
    # It's good practice to blur images as it removes noise
    #img = cv2.GaussianBlur(img, (3, 3), 0)
    adaptive_th=cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,51,30)
    return adaptive_th


# Median Filtering Logic 
def median_subtract(img, ksize=23):
    background=cv2.medianBlur(img, ksize)
    result=cv2.subtract(background, img)
    result=cv2.bitwise_not(result)
    return (result, background)


# Morphological operation
def edge_detection_dilation_erosion(img):
    edges=cv2.Canny(img, 100, 100)
    edges=cv2.bitwise_not(edges)
    kernel = np.ones((3,3),np.uint8)
    dilation = cv2.dilate(cv2.bitwise_not(edges),kernel,iterations = 1)
    dilation=cv2.bitwise_not(dilation)
    kernel1 = np.ones((9,9),np.uint8)
    erosion=cv2.erode(cv2.bitwise_not(img), kernel1,iterations=1)
    erosion=cv2.bitwise_not(erosion)
    return (edges, dilation, erosion)


def Filters(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('gray', img)
    # cv2.waitKey(0)
    # Normalization
    # img=preprocessing.Normalization(img)
    # cv2.imshow('Equalized', img)
    # cv2.waitKey(0)

    #adaptive_thresholding
    result =adaptive_thresholding(img)
    # cv2.imshow('adaptive_thresholding', result)
    # cv2.waitKey(0)

    # Peform median filtering over dirty image
    #result, background = median_subtract(result)
    #cv2.imshow('median_subtract', result)
    #cv2.waitKey(0)

    #   binary_thresholding
    #result = binary_thresholding(img)
    #cv2.imshow('binary_thresholding', result)
    #cv2.waitKey(0)
    return result

def Remove_borders(th):
    h, w = th.shape
    h_h = [0] * h
    done=0
    for j in range(100):
        for i in range(w):
            if th[j, i] == 0:
                h_h[j] += 1
        if h_h[j]>10:
            for k in range(0,j+50):
                for i in range(w):
                    th[k, i] = 255
            done=1
        if done:
            break
    ###################################################
    done=0
    for j in range(h-1,h-100,-1):
        for i in range(w):
            if th[j, i] == 0:
                h_h[j] += 1
        if h_h[j] > 10:
            for k in range(h-1, j -50,-1):
                for i in range(w):
                    th[k, i] = 255
            done = 1
        if done:
            break
   ########################################################

    done=0
    w_w = [0] * w
    for i in range(100):
        for j in range(h):
            if th[j, i] == 0:
                w_w[i] += 1
        if w_w[i]>10:
            for k in range(0,i+50):
                for j in range(h):
                    th[j, k] = 255
            done=1
        if done:
            break
    #######################################################
    done=0
    for i in range(w-1,w-100,-1):
        for j in range(h):
            if th[j, i] == 0:
                w_w[i] += 1
        if w_w[i] > 10:
            for k in range(w-1, i - 50,-1):
                for j in range(h):
                    th[j, k] = 255
            done = 1
        if done:
            break
    #######################################################
    return th



