import cv2
from random import randint
from VPP_Character_segmentation import segment
from preprocessing import morphological_operation,gaus_thresh,Resize
import os
image = cv2.imread('image.png')

image=Resize(image)

imgcpy1 = image.copy()
imgcpy2 = image.copy()

thresh = gaus_thresh(image)
dilation = morphological_operation(thresh)
(contours1, _) = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
i = 1
contours1=list(contours1)
contours1.reverse()

for cnt in contours1:
    area = cv2.contourArea(cnt)
    if area > 30000:
        x, y, w, h = cv2.boundingRect(cnt)
        line = imgcpy1[y:y + h, x:x + w]
        newpath = 'lines/'
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        cv2.imwrite(newpath+ '_' + str(i) + '.png', line)
        cv2.rectangle(imgcpy2, (x-1, y-5), (x + w, y + h), (randint(0, 255), randint(0, 255), randint(0, 255)), 5)
        segment(line,i)
        i += 1
cv2.imwrite('imgContoure.png', imgcpy2)


