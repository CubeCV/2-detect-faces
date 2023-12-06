import cv2
import numpy as np

from math import sqrt


# INFO: utils
def distance(x1, y1, x2, y2):
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


# INFO: all base functions
def fill_image(img, change=True):
    return cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if change else img, 0, 240, cv2.THRESH_BINARY)[1]


def mask_white(img):
    temp = img.copy()
    temp[np.where((temp != [0, 0, 0]).any(axis=2))] = [255, 255, 255]
    return temp


def setup_contours(img, epsilon):
    cannied = cv2.Canny(img, threshold1=200, threshold2=600)

    contours0, _ = cv2.findContours(cannied.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cv2.approxPolyDP(cnt, epsilon, True) for cnt in contours0]

    return contours


def produce_contours(img, epsilon=10):
    contours = setup_contours(img, epsilon)
    contours = list(map(lambda x: x[0], contours[0]))

    pairs = zip(contours, np.append(contours[1:], [contours[0]], axis=0))

    return pairs


def show_bounding_lines(img, contours, blank=False):
    vis = img.copy() if not blank else np.zeros(img.shape, np.uint8)

    for i, (c1, c2) in enumerate(contours):
        print(i, c1, c2, distance(c1[0], c1[1], c2[0], c2[1]))
        temp = np.zeros(img.shape, np.uint8)
        cv2.line(temp, c1, c2, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.line(vis, c1, c2, (255, 255, 255), 3, cv2.LINE_AA)

    return vis
