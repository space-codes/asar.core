import cv2
from preprocessing import remove_dots
import os


def word_segmentation_from_left_to_right(src_img, i):
    height = src_img.shape[0]
    width = src_img.shape[1]

    # preprocessing for word segmentation
    grey_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    ret, bin_img = cv2.threshold(grey_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    final_thr = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)
    final_thr = remove_dots(final_thr)
    final_thr = cv2.convertScaleAbs(final_thr, alpha=(255.0))

    # finding words contours
    letter_k = []
    contours, hierarchy = cv2.findContours(final_thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # selecting important contours
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if cv2.contourArea(cnt) >= 70 or 2 * w <= h:
            letter_k.append((x, y, w, h, (-x - w)))

    # sorting contours positions from left to right
    letter = sorted(letter_k, key=lambda sort_by: sort_by[0])

    letter_index = len(letter)
    end = 0

    # cropping each contour from the original image
    for e in range(len(letter)):
        ok = 1
        # if this word considered before neglect it
        if (letter[e][0] + letter[e][2]) < end:
            continue
        # if the difference between ends less than 5 and it doesn't alef ,neglect it
        if ((letter[e][0] + letter[e][2]) - end) < 5 and 2 * letter[e][2] > letter[e][3]:
            ok = 0
        if ((letter[e][0] + letter[e][2]) - end) < 0.25 * (letter[e][2]):
            ok = 0

        start = letter[e][0]
        if start < end:
            start = end
        end = letter[e][0] + letter[e][2]

        if ok == 0:
            continue

        letter_index -= 1
        h1 = 0
        if letter[e][1] - 50 > 0:
            h1 = letter[e][1] - 50
        h2 = height - 1
        if h2 > letter[e][3] + letter[e][1] + 50:
            h2 = letter[e][3] + letter[e][1] + 50

        w1 = start - 5
        if w1 < 0:
            w1 = 0
        w2 = end + 2
        if w2 >= width:
            w2 = width - 1

        if (w2 - w1) <= 5:
            continue

        letter_img_tmp = src_img[h1:h2, w1:w2]

        newpath = 'segmented_data_LR/'
        if not os.path.exists(newpath):
            os.makedirs(newpath)

        if w2 > w1 and h2 > h1:
            cv2.imwrite(newpath + str(i) + '_' + str(letter_index) + '.jpg', letter_img_tmp)


def word_segmentation_from_right_to_left(src_img, i):
    height = src_img.shape[0]
    width = src_img.shape[1]

    # preprocessing for word segmentation
    grey_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    bin_img = cv2.adaptiveThreshold(grey_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 20)
    final_thr = remove_dots(bin_img)

    # finding words contours
    letter_k = []
    contours, hierarchy = cv2.findContours(final_thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # selecting important contours
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if cv2.contourArea(cnt) >= 70 or 2 * w <= h:
            letter_k.append((x, y, w, h, (-x - w)))

    # sorting contours positions from right to left
    letter = sorted(letter_k, key=lambda sort_by: sort_by[4])

    letter_index = 1
    end = width - 1

    # cropping each contour from the original image
    for e in range(len(letter)):
        ok = 1
        # if height isn't greater than width
        if (end - letter[e][0]) < 3 and 2 * letter[e][2] > letter[e][3]:
            ok = 0
        if (end - letter[e][0]) < 5 and letter[e][2] > 10:
            ok = 0
        if (end - letter[e][0]) < 0.25 * (letter[e][2]):
            ok = 0
        start = letter[e][0] + letter[e][2]

        if start > end:
            start = end

        end = letter[e][0]
        if start - end < 3:
            ok = 0
        if ok == 0:
            continue

        letter_index += 1
        h1 = 1
        if (letter[e][1] - 50) > 0:
            h1 = letter[e][1] - 50
        h2 = height - 2
        if h2 > (letter[e][3] + letter[e][1] + 50):
            h2 = (letter[e][3] + letter[e][1] + 50)
        w1 = end
        w2 = start + 5
        if w1 < 0:
            w1 = 0
        if w2 >= width:
            w2 = width - 1

        letter_img_tmp = src_img[h1:h2, w1:w2]

        newpath = 'segmented_data_RL/'
        if not os.path.exists(newpath):
            os.makedirs(newpath)

        if w2 > w1 and h2 > h1:
            cv2.imwrite(newpath + str(i) + '_' + str(letter_index) + '.jpg', letter_img_tmp)


def line_segmentation(src_img, contours):
    height, width = src_img.shape[0], src_img.shape[1]
    contours = list(contours)
    contours.reverse()
    i = 1
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 30000:
            x, y, w, h = cv2.boundingRect(cnt)
            y2 = y - 5
            if y2 < 0:
                y2 = 0
            h2 = y2 + h + 15
            if h2 > height - 1:
                h2 = height - 1
            line = src_img[y2:h2, x:x + w]
            newpath = 'lines/'
            if not os.path.exists(newpath):
                os.makedirs(newpath)
            cv2.imwrite(newpath + '_' + str(i) + '.png', line)
            word_segmentation_from_left_to_right(line, i)
            word_segmentation_from_right_to_left(line, i)
            i += 1
