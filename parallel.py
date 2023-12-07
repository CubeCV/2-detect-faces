import cv2
import numpy as np

from math import atan2, acos, pi
from itertools import combinations


def find_angle(v1, v2):
    return acos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def find_parallel(contours, img, debug=False):
    dirs = []

    for i, (a, b) in enumerate(contours):
        dir = np.array([a[0] - b[0], a[1] - b[1], i])
        dirs.append(dir)

    best1 = list(sorted(combinations(dirs, 2), key=(lambda x: find_angle(x[0], x[1]) / pi), reverse=True)[:3])

    parallel_lines = list(map(lambda x: (x[0][2], x[1][2]), best1))

    if debug:
        for i, b in enumerate(parallel_lines):
            x, y = b

            lines = cv2.line(np.zeros(img.shape, np.uint8), contours[x][0], contours[x][1], (255, 255, 255), 3, cv2.LINE_AA)
            lines = cv2.line(lines, contours[y][0], contours[y][1], (255, 255, 255), 3, cv2.LINE_AA)
            cv2.imshow(f"line{i}", lines)

    return parallel_lines
