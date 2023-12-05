import cv2
from math import sqrt
import numpy as np

img = cv2.imread("./test.jpg")


def fill_image(img):
    return cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 0, 240, cv2.THRESH_BINARY)[1]


def setup_contours(img, epsilon):
    cannied = cv2.Canny(img, threshold1=200, threshold2=600)
    contours0, _ = cv2.findContours(cannied.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cv2.approxPolyDP(cnt, epsilon, True) for cnt in contours0]

    return contours


def produce_contours(img, epsilon=10):
    contours = setup_contours(img, epsilon)

    vis = np.zeros(img.shape, np.uint8)

    for i, (c1, c2) in enumerate(zip(contours[0][:-1], contours[0][1:])):
        print(i, c1, c2)
        temp = np.zeros(img.shape, np.uint8)
        cv2.line(temp, c1[0], c2[0], (255, 255, 255), 3, cv2.LINE_AA)
        cv2.imshow(f"{i}", temp)

    return cv2.drawContours(vis, contours, -1, (255, 255, 255), 3, cv2.LINE_AA)


WIDTH, HEIGHT = 1488, 1530
KERNEL_SIZE = x if (x := int((WIDTH + HEIGHT) / 25)) % 2 else (x + 1)
print(KERNEL_SIZE)


img = cv2.GaussianBlur(img, (KERNEL_SIZE, KERNEL_SIZE), 0)
img = fill_image(img)
cv2.imshow("sdf2", img)

# new = cv2.GaussianBlur(y, (KERNEL_SIZE, KERNEL_SIZE), 0)
# cv2.imshow("dsfsdf", new)

img = produce_contours(img, epsilon=KERNEL_SIZE // 3)
cv2.imshow("sdf", img)



cv2.waitKey(0)
cv2.destroyAllWindows()

