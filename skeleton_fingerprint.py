# Copyright (C) 2020  Ngô Ngọc Đức Huy
from cv2 import (
    imread, threshold, countNonZero, medianBlur,
    getStructuringElement, MORPH_CROSS,
    erode, dilate, subtract, bitwise_or
)
import numpy as np
import matplotlib.pyplot as plt


def skeleton(img):
    img = medianBlur(img, 5)
    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)

    ret, img = threshold(img, 127, 255, 0)
    element = getStructuringElement(MORPH_CROSS, (3, 3))
    done = False

    while(not done):
        eroded = erode(img, element)
        temp = dilate(eroded, element)
        temp = subtract(img, temp)
        skel = bitwise_or(skel, temp)
        img = eroded.copy()

        zeros = size - countNonZero(img)
        if zeros == size:
            done = True
    return skel


img = imread('dataset/right-hand.png', 0)
plt.subplot(221)
plt.title('Original image')
plt.imshow(img, cmap='gray')

bl = medianBlur(img, 9)
plt.subplot(222)
plt.title('Blurred image')
plt.imshow(bl, cmap='gray')

skel = skeleton(img)
plt.subplot(223)
plt.imshow(skel, cmap='gray')

skel = skeleton(bl)
plt.subplot(224)
plt.imshow(skel, cmap='gray')

plt.show()
