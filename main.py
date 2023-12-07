import cv2
import numpy as np
import sys

from contour import bounding_lines
from parallel import find_parallel
from projection import find_corners, project


DEBUG = True

original = cv2.imread(f"../images/isolated/cube{x[1] if len(x:=sys.argv) > 1 else 0}.jpg")

img = original.copy()

contours = bounding_lines(img, debug=DEBUG)
parallel = find_parallel(contours, img, debug=DEBUG)

all_corners = find_corners(img, contours, parallel, debug=DEBUG)

project(img, all_corners)


# INFO: keep windows
cv2.waitKey(0)
cv2.destroyAllWindows()
