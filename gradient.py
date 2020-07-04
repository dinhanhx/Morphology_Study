# Copyright (C) 2020  Ngô Ngọc Đức Huy
from sys import argv
from cv2 import (
    imread,
    cvtColor, COLOR_BGR2GRAY,
    threshold, THRESH_BINARY,
    morphologyEx, MORPH_GRADIENT
)
from numpy import ones, uint8
import matplotlib.pyplot as plt

kernel_list = [
    ones((3, 3), uint8),
    ones((5, 5), uint8)
]


def to_gray(image, thr):
    """Convert an image to gray."""
    image = cvtColor(image, COLOR_BGR2GRAY)
    return image


def gradient(image, kernel):
    """Apply gradient transform on a grayscale image."""
    image = morphologyEx(image, MORPH_GRADIENT, kernel)
    return image


if __name__ == "__main__":
    img = imread(argv[1])
    img = to_gray(img, 100)
    plt.subplot(2, 3, 1)
    plt.imshow(img, cmap='gray')
    i = 2

    for kernel in kernel_list:
        res = gradient(img, kernel)
        plt.subplot(2, 3, i)
        i += 1
        plt.imshow(res, cmap='gray')
        plt.subplot(2, 3, i)
        res = threshold(res, 80, 255, THRESH_BINARY)[1]
        i += 1
        plt.imshow(res, cmap='gray')
    plt.show()
