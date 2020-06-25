# Copyright (C) 2020  Ngô Ngọc Đức Huy
from cv2 import (
    imread, blur,
    cvtColor, COLOR_BGR2GRAY,
    threshold, THRESH_BINARY_INV,
    morphologyEx, MORPH_HITMISS
)
from numpy import ones, uint8, array
import matplotlib.pyplot as plt

kernel_list = [
    ones((3, 3), uint8),
    ones((5, 5), uint8),
    array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], uint8),
    array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], uint8),
    array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], uint8),
    array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], uint8),
    array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], uint8),
    array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], uint8),
    array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], uint8),
    array([[0, 1, 0], [1, -1, 1], [0, 1, 0]], uint8),
    array([[0, -1, -1], [1, 1, -1], [0, 1, 0]], uint8),
    array([[-1, -1, 0], [-1, 1, 0], [-1, -1, 0]], uint8),
]


def to_binary(image, thr):
    """Convert a grayscale image to a binary image."""
    image = cvtColor(image, COLOR_BGR2GRAY)
    image = blur(image, (3, 3))
    return threshold(image, thr, 255, THRESH_BINARY_INV)[1]


def hit_miss(image, kernel):
    """Apply hit-or-miss transform on a binary image."""
    image = morphologyEx(image, MORPH_HITMISS, kernel)
    return image


if __name__ == "__main__":
    img = imread('dataset/captcha/1.jpg')
    img = to_binary(img, 100)
    plt.subplot(5, 3, 2)
    plt.imshow(img, cmap='gray')
    i = 4

    for kernel in kernel_list:
        res = hit_miss(img, kernel)
        plt.subplot(5, 3, i)
        i += 1
        plt.imshow(res, cmap='gray')
    plt.show()
