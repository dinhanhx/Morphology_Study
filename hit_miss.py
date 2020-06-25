# Copyright (C) 2020  Ngô Ngọc Đức Huy
from cv2 import (
    imread, blur,
    cvtColor, COLOR_BGR2GRAY,
    threshold, THRESH_BINARY,
    morphologyEx, MORPH_HITMISS
)
from numpy import ones, uint8
import matplotlib.pyplot as plt


def to_binary(image, thr):
    """Convert a grayscale image to a binary image."""
    image = cvtColor(image, COLOR_BGR2GRAY)
    image = blur(image, (3, 3))
    return threshold(image, thr, 255, THRESH_BINARY)[1]


def hit_miss(image):
    """Apply hit-or-miss transform on a binary image."""
    image = morphologyEx(image, MORPH_HITMISS, ones((5, 5), uint8))
    return image


if __name__ == "__main__":
    img = imread('dataset/captcha/1.jpg')
    res = to_binary(img, 100)
    plt.subplot(211)
    plt.imshow(res, cmap='gray')
    plt.subplot(212)
    res = hit_miss(res)
    plt.imshow(res, cmap='gray')
    plt.show()
