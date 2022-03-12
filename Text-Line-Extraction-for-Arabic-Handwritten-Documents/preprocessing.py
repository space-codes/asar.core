import numpy as np
import cv2


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

def gaus_thresh(image):
    image = cv2.GaussianBlur(image, (5, 5), 1)
    cv2.imwrite('FinalOutput/02_gaussian.png', image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("FinalOutput/03_grey.jpg", gray)

    gray = np.array(255 * (gray / 255) ** 1, dtype='uint8')
    cv2.imwrite("FinalOutput/04_greyImgGamaCorrelation.jpg", gray)

    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imwrite('FinalOutput/05_otsus.png', thresh)
    #thresh = add_padding(thresh)
    # cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    # for c in cnts:
    #     area = cv2.contourArea(c)
    #     if area < 400:
    #         cv2.drawContours(thresh, [c], -1, (0, 0, 0), -1)
    # cv2.imwrite('FinalOutput/06_nonoise.png', thresh)
    return thresh

def iteration(image: np.ndarray, value: int) -> np.ndarray:
    """
    This method iterates over the provided image by converting 255's to 0's if the number of consecutive 255's are
    less the "value" provided
    """

    rows, cols = image.shape
    for row in range(0, rows):
        try:
            start = image[row].tolist().index(0)  # to start the conversion from the 0 pixel
        except ValueError:
            start = 0  # if '0' is not present in that row

        count = start
        for col in range(start, cols):
            if image[row, col] == 0:
                if (col - count) <= value and (col - count) > 0:
                    image[row, count:col] = 0
                count = col
    return image

def rlsa(image: np.ndarray, horizontal: bool = True, vertical: bool = True, value: int = 0) -> np.ndarray:
    """
    rlsa(RUN LENGTH SMOOTHING ALGORITHM) is to extract the block-of-text or the Region-of-interest(ROI) from the
    document binary Image provided. Must pass binary image of ndarray type.
    """

    if isinstance(image, np.ndarray):  # image must be binary of ndarray type
        value = int(value) if value >= 0 else 0  # consecutive pixel position checker value to convert 255 to 0
        try:
            # RUN LENGTH SMOOTHING ALGORITHM working horizontally on the image
            if horizontal:
                image = iteration(image, value)

                # RUN LENGTH SMOOTHING ALGORITHM working vertically on the image
            if vertical:
                image = image.T
                image = iteration(image, value)
                image = image.T

        except (AttributeError, ValueError) as e:
            image = None
            print("ERROR: ", e, "\n")
            print(
                'Image must be an np ndarray and must be in "binary". Use Opencv/PIL to convert the image to binary.\n')
            print("import cv2;\nimage=cv2.imread('path_of_the_image');\ngray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY);\n\
                (thresh, image_binary) = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)\n")
            print("method usage -- rlsa.rlsa(image_binary, True, False, 10)")
    else:
        print('Image must be an np ndarray and must be in binary')
        image = None
    return image

def morphological_operation(thresh):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 5))
    line_img = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 1))
    close = cv2.morphologyEx(line_img, cv2.MORPH_CLOSE, close_kernel, iterations=3)
    #cv2.imwrite('FinalOutput/07_lineIMG.png', close)

    kernel = np.ones((1, 60), np.uint8)
    close = cv2.erode(close, kernel, iterations=3)
    #cv2.imwrite('FinalOutput/08_erosion.png', close)

    blur = cv2.blur(close, (99, 1), 0)
    #cv2.imwrite('FinalOutput/09_blureImg.tif', blur)

    _, imgPart = cv2.threshold(blur, 1, 255, cv2.THRESH_BINARY)
    #cv2.imwrite('FinalOutput/10_imgPage.png', imgPart)

    image_rlsa_horizontal = rlsa(imgPart, True, False, 20)
    #cv2.imwrite('FinalOutput/18_rlsa.png', image_rlsa_horizontal)

    close_kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 1))
    #close1 = cv2.morphologyEx(image_rlsa_horizontal, cv2.MORPH_CLOSE, close_kernel1, iterations=3)
    kernel1 = np.ones((1, 25), np.uint8)
    #close1 = cv2.erode(close1, kernel1, iterations=4)
    #cv2.imwrite('FinalOutput/19_close1.png', close1)
    # opening = cv2.morphologyEx(close1, cv2.MORPH_OPEN, np.ones((1,10), np.uint8))
    # cv2.imwrite('FinalOutput/close11.png',opening)
    dilation = cv2.dilate(image_rlsa_horizontal, close_kernel1, iterations=8)
    #cv2.imwrite('FinalOutput/20_close111.png', dilation)
    return dilation
