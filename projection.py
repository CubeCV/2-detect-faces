import cv2
import numpy as np
import sys

from itertools import combinations


def connect_parallel_lines(lines: list[tuple[int, int], tuple[int, int]]):
    for i, l in enumerate(lines):
        z = list(zip(l[0], l[1]))

        # WARN: hardcoded value 5 for number of contours... 
        # should be len of contours or something?
        # or always pass in 6 connected lines...

        if not any(map(lambda x: abs(x[1] - x[0]) % 5 <= 1, z)):
            lines[i] = (l[0], (l[1][1], l[1][0]))

    return lines


def find_corners(img, contours, parallel, debug=False):
    faces = list(combinations(parallel, 2))
    paired_parallel_lines = connect_parallel_lines(faces)

    x = np.zeros((2,), np.float64)

    all_corners = []

    for j in [0, 1, 2]:
        for i in [0, 1]:
            work = paired_parallel_lines[j]

            actual_min = max if sorted([work[0][i], work[1][i]]) == [0, 5] else min
            actual_max = min if sorted([work[0][i], work[1][i]]) == [0, 5] else max

            edge1 = contours[actual_min(work[0][i], work[1][i])]
            edge2 = contours[actual_max(work[0][i], work[1][i])]

            mid = edge1[1]

            dif1 = edge1[1] - edge1[0]
            dif2 = edge2[0] - edge2[1]

            x += (mid - dif1 - dif2)

            corners = [
                edge1[0],
                mid,
                mid - dif1 - dif2,
                edge2[1]
            ]

            all_corners.append(corners)

            for c in corners:
                img = cv2.circle(img, c, 4, (255, 255, 255), 4)

    if debug:
        print(list(faces))

        print(x)
        img = cv2.circle(img, (int(x[0]) // 6, int(x[1]) // 6), 4, (0, 255, 255), 4)
        cv2.imshow("all", img)

    return all_corners


def project(img, corners):
    for i, c in enumerate(corners):
        c = np.array(c).astype('float32')

        destination = np.array([
            [0, 0],
            [0, 200],
            [200, 0],
            [200, 200]], dtype="float32")

        m = cv2.getPerspectiveTransform(c, destination)

        output = cv2.warpPerspective(img, m, (200, 200))

        cv2.imshow(f"final{i}", output)
