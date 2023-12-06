import cv2
import numpy as np
import sys
from math import atan2, acos, pi
from itertools import combinations

from processing.contour import produce_contours, show_bounding_lines, mask_white


original = cv2.imread(f"./data/isolated_cube{x[1] if len(x:=sys.argv) > 1 else 0}.jpg")

# original = cv2.rotate(original, cv2.ROTATE_90_CLOCKWISE)  # WARN: why???


WIDTH, HEIGHT = 1488, 1530
KERNEL_SIZE = x if (x := int((WIDTH + HEIGHT) / 30)) % 2 else (x + 1)


# INFO: find bounding lines
img = mask_white(original.copy())

img = cv2.GaussianBlur(img, (KERNEL_SIZE, KERNEL_SIZE), 0)
img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)[1]

contours = produce_contours(img, epsilon=KERNEL_SIZE // 2)

final = show_bounding_lines(original, contours, blank=False)

cv2.imshow("final", final)


# INFO: find pieces

# def compute_gradient(x1, y1, x2, y2):
#     return (y2 - y1) / (x2 - x1)


# def compute_angle(m1, m2):
#     return atan2((m1 - m2) / (1 + m1 * m2))


# def find_parallel(contours):
#     gradients = []

#     for a, b in contours:
#         gradients.append(compute_gradient(a[0], a[1], b[0], b[1]))

#     for i in range(len(contours)):
#         print("\n")
#         for j in range(i):
#             print(f"{i}, {j}, -> {compute_angle(gradients[i], gradients[j])}")

def find_angle(v1, v2):
    return acos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def find_parallel(contours):
    dirs = []

    for i, (a, b) in enumerate(contours):
        dir = np.array([a[0] - b[0], a[1] - b[1], i])
        dirs.append(dir)

    best1 = list(sorted(combinations(dirs, 2), key=(lambda x: find_angle(x[0], x[1]) / pi), reverse=True)[:3])

    parallel_lines = list(map(lambda x: (x[0][2], x[1][2]), best1))

    for i, b in enumerate(parallel_lines):
        x, y = b

        lines = cv2.line(np.zeros(original.shape, np.uint8), contours[x][0], contours[x][1], (255, 255, 255), 3, cv2.LINE_AA)
        lines = cv2.line(lines, contours[y][0], contours[y][1], (255, 255, 255), 3, cv2.LINE_AA)
        cv2.imshow(f"line{i}", lines)


find_parallel(contours)


# INFO: keep windows
cv2.waitKey(0)
cv2.destroyAllWindows()
