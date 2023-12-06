import cv2
from math import sqrt
import numpy as np
import sys
from PIL import Image, ImageFilter

original = cv2.imread(f"./data/isolated_cube{x[1] if len(x:=sys.argv) > 1 else 0}.jpg")

original = cv2.rotate(original, cv2.ROTATE_90_CLOCKWISE)


def fill_image(img, change=True):
    return cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if change else img, 0, 240, cv2.THRESH_BINARY)[1]


def mask_white(img):
    temp = img.copy()
    temp[np.where((temp != [0, 0, 0]).any(axis=2))] = [255, 255, 255]
    return temp


def setup_contours(img, epsilon):
    cannied = cv2.Canny(img, threshold1=200, threshold2=600)

    cv2.imshow("cannied", cannied)

    contours0, _ = cv2.findContours(cannied.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cv2.approxPolyDP(cnt, epsilon, True) for cnt in contours0]

    return contours


def distance(x1, y1, x2, y2):
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def produce_contours(img, epsilon=10):
    contours = setup_contours(img, epsilon)

    print(contours)

    contours = list(map(lambda x: x[0], contours[0]))

    for c in contours:
        print(c)

    # vis = np.zeros(img.shape, np.uint8)
    vis = original.copy()

    # pairs = zip(contours[:-1], contours[1:])
    pairs = zip(contours, np.append(contours[1:], [contours[0]], axis=0))

    new = sorted(pairs, key=lambda x: distance(x[0][0], x[0][1], x[1][0], x[1][1]), reverse=True)

    for i, (c1, c2) in enumerate(new):
        print(i, c1, c2, distance(c1[0], c1[1], c2[0], c2[1]))
        temp = np.zeros(img.shape, np.uint8)
        cv2.line(temp, c1, c2, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.line(vis, c1, c2, (255, 255, 255), 3, cv2.LINE_AA)
        # cv2.imshow(f"{i}", temp)

    return vis


WIDTH, HEIGHT = 1488, 1530
KERNEL_SIZE = x if (x := int((WIDTH + HEIGHT) / 30)) % 2 else (x + 1)


img = original.copy()

# img = fill_image(img)

img = mask_white(img)
cv2.imshow("filled", img)

img = cv2.GaussianBlur(img, (KERNEL_SIZE, KERNEL_SIZE), 0)
img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)[1]

# for _ in [0] * 10:
#     img = cv2.GaussianBlur(img, (KERNEL_SIZE, KERNEL_SIZE), 0)



# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow("processed", img)



# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# pil_image = Image.fromarray(img)

# pil_image = pil_image.filter(ImageFilter.ModeFilter(size=KERNEL_SIZE))

# pil_image.save('output_image.png')


# img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# img = cv2.ximgproc.anisotropicDiffusion(img, KERNEL_SIZE / 1500, 200, 100)
# cv2.imshow("test", img)


# img = cv2.GaussianBlur(original, (KERNEL_SIZE, KERNEL_SIZE), 0)
# cv2.imshow("blurred", img)


img = produce_contours(img, epsilon=KERNEL_SIZE // 2)
cv2.imshow("sdf", img)


cv2.waitKey(0)
cv2.destroyAllWindows()
