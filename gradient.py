# Copyright (C) 2020  Ngô Ngọc Đức Huy
from cv2 import (
    imread,
    cvtColor, COLOR_BGR2GRAY,
    threshold, THRESH_BINARY,
    morphologyEx, MORPH_GRADIENT
)
from numpy import ones, uint8, array
import matplotlib.pyplot as plt

kernel_list = {
    '3x3 square': ones((3, 3), uint8),
    '5x5 square': ones((5, 5), uint8),
    '3x3 cross': array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], uint8),
    '5x5 cross': array([[0, 0, 1, 0, 0],
                        [0, 0, 1, 0, 0],
                        [1, 1, 1, 1, 1],
                        [0, 0, 1, 0, 0],
                        [0, 0, 1, 0, 0]], uint8)
}


def to_gray(image, thr):
    """Convert an image to gray."""
    image = cvtColor(image, COLOR_BGR2GRAY)
    return image


def gradient(image, kernel):
    """Apply gradient transform on a grayscale image."""
    image = morphologyEx(image, MORPH_GRADIENT, kernel)
    return image


if __name__ == "__main__":
    img = imread('dataset/left-hand.png')
    img = to_gray(img, 100)
    plt.subplot(3, 4, 1)
    plt.title('Original')
    plt.imshow(img, cmap='gray')
    i = 5

    for kernel in kernel_list:
        plt.subplot(3, 4, i)
        i += 1
        res = gradient(img, kernel_list[kernel])
        plt.title(kernel)
        plt.imshow(res, cmap='gray')

        plt.subplot(3, 4, i)
        i += 1
        res = threshold(res, 80, 255, THRESH_BINARY)[1]
        plt.title(f'{kernel}, thresholded')
        plt.imshow(res, cmap='gray')
    plt.show()
