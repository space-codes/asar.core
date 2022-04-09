import cv2
import numpy as np


def remove_dots(img):
    height = img.shape[0]
    width = img.shape[1]
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(
        img.astype(np.uint8), connectivity=8)
    new_img = np.zeros_like(img)

    for i in range(1, ret):
        cc_width = stats[i, cv2.CC_STAT_WIDTH]
        cc_height = stats[i, cv2.CC_STAT_HEIGHT]

        if cc_width >= 0.2 * width or cc_height >= 0.2 * height:
            new_img[labels == i] = 1

    return new_img


def letter_seg(src_img, i):
    height = src_img.shape[0]
    width = src_img.shape[1]

    grey_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)

    bin_img = cv2.adaptiveThreshold(grey_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 20)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    final_thr = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)
    final_thr = remove_dots(final_thr)
    final_thr = cv2.convertScaleAbs(final_thr, alpha=(255.0))

    letter_k = []
    contours, hierarchy = cv2.findContours(final_thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) >= 70:
            x, y, w, h = cv2.boundingRect(cnt)
            letter_k.append((x, y, w, h))

    letter = sorted(letter_k, key=lambda student: student[0])
    letter_index = len(letter)
    end = 0
    for e in range(len(letter)):
        if (letter[e][0] + letter[e][2] - 5) < end:
            continue
        start = letter[e][0]
        if start < end:
            start = end
        end = letter[e][0] + letter[e][2]
        letter_index -= 1
        h2 = height - 1
        if h2 > h + 50:
            h2 = h + 50
        letter_img_tmp = src_img[0:h2, start:end]

        cv2.imwrite('segmented_data/' + str(i) + '_' + str(letter_index) + '.jpg', letter_img_tmp)



# img = cv2.imread('lines/_2.png', 1)
#
#
# letter_seg(img, 2)
