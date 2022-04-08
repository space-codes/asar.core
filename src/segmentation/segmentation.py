import cv2
from VPP_Character_segmentation import segment
from preprocessing import morphological_operation,gaus_thresh
from word_segmentation_with_contour import letter_seg
import os
image = cv2.imread('3.png')

imgcpy1 = image.copy()
imgcpy2 = image.copy()

thresh = gaus_thresh(image)
dilation = morphological_operation(thresh)
(contours1, _) = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
i = 1
contours1=list(contours1)
contours1.reverse()
height,width=imgcpy1.shape[0],imgcpy1.shape[1]
for cnt in contours1:
    area = cv2.contourArea(cnt)
    if area > 30000:
        x, y, w, h = cv2.boundingRect(cnt)
        y2=y-5
        if y2<0:
            y2=0
        h2= y2 + h+10
        if h2>height-1:
            h2=height-1
        line = imgcpy1[y2:h2, x:x + w]
        newpath = 'lines/'
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        cv2.imwrite(newpath+ '_' + str(i) + '.png', line)
        # segment(line,i)
        letter_seg(line,i)
        i += 1


