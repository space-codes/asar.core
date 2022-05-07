import numpy as np
import cv2


def Resize(image):
    h,w=image.shape[0],image.shape[1]
    if w>1200:
        return image
    mn=w*20
    factor=20
    for i in range(1,11):
        dif=abs(w*i-1200)
        if dif<mn:
            mn=dif
            factor=i
    image = cv2.resize(image, (w*factor,h*factor), interpolation=cv2.INTER_AREA)
    return image


def gaus_thresh(image):
    image = cv2.GaussianBlur(image, (5, 5), 1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.array(255 * (gray / 255) ** 1, dtype='uint8')
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
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

    kernel = np.ones((1, 60), np.uint8)
    close = cv2.erode(close, kernel, iterations=3)

    blur = cv2.blur(close, (99, 1), 0)

    _, imgPart = cv2.threshold(blur, 1, 255, cv2.THRESH_BINARY)

    image_rlsa_horizontal = rlsa(imgPart, True, False, 20)

    close_kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 1))

    dilation = cv2.dilate(image_rlsa_horizontal, close_kernel1, iterations=8)

    return dilation


def remove_dots(img):
    height = img.shape[0]
    width = img.shape[1]

    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(
        img.astype(np.uint8), connectivity=8)

    new_img = np.zeros_like(img)

    for i in range(1, ret):
        cc_width = stats[i, cv2.CC_STAT_WIDTH]
        cc_height = stats[i, cv2.CC_STAT_HEIGHT]
        y=int(centroids[i][1])
        if ( y > 0.3*height and y < 0.7*height and cc_height >= 0.15 * height)or\
                (cc_height >= 0.3 * height )or\
                (cc_height >= 0.2 * height and y > 0.3*height):
          new_img[labels == i] = 255

    return new_img