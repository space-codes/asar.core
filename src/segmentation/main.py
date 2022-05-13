import cv2
from preprocessing import morphological_operation,gaus_thresh
from line_and_word_segmentation_with_contour import line_segmentation

# Read image
image = cv2.imread('image2.png')

# Apply Threshoding convert to binary image
thresh = gaus_thresh(image)

# Detect lines positions by morphological operation
dilation = morphological_operation(thresh)

# extract lines from the image
(contours1,_) = cv2.findContours(dilation,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

# Apply line segmentation then word segmentation with contour
line_segmentation(image,contours1)
