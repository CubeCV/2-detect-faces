import cv2
import numpy as np
import sys

from processing.contour import bounding_lines
from processing.parallel import find_parallel


DEBUG = True

original = cv2.imread(f"../images/isolated/cube{x[1] if len(x:=sys.argv) > 1 else 0}.jpg")

img = original.copy()

contours = bounding_lines(img, debug=DEBUG)
parallel = find_parallel(contours, img, debug=DEBUG)

print(parallel)


# INFO: keep windows
cv2.waitKey(0)
cv2.destroyAllWindows()
