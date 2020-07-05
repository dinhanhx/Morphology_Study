# Copyright (C) 2020  Ngô Ngọc Đức Huy
from cv2 import (
    imread, threshold, countNonZero,
    getStructuringElement, MORPH_CROSS,
    erode, dilate, subtract, bitwise_or
)
import numpy as np
import matplotlib.pyplot as plt


def skeleton(filename):
    img = imread(filename, 0)
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


img = imread('dataset/hand.png')
plt.subplot(211)
plt.imshow(img, cmap='gray')

skel = skeleton('dataset/hand.png')
plt.subplot(212)
plt.imshow(skel, cmap='gray')

plt.show()
