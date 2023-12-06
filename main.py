import cv2
import numpy as np
import sys

from processing.contour import produce_contours, show_bounding_lines, mask_white


original = cv2.imread(f"./data/isolated_cube{x[1] if len(x:=sys.argv) > 1 else 0}.jpg")


WIDTH, HEIGHT = 1488, 1530
KERNEL_SIZE = x if (x := int((WIDTH + HEIGHT) / 30)) % 2 else (x + 1)


# original = cv2.rotate(original, cv2.ROTATE_90_CLOCKWISE)

img = mask_white(original.copy())

img = cv2.GaussianBlur(img, (KERNEL_SIZE, KERNEL_SIZE), 0)
img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)[1]

contours = produce_contours(img, epsilon=KERNEL_SIZE // 2)

final = show_bounding_lines(original, contours, blank=False)

cv2.imshow("final", final)


cv2.waitKey(0)
cv2.destroyAllWindows()
