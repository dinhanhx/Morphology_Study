import cv2
import numpy as np

fpath = 'dataset/captcha.jpg'
img = cv2.imread(fpath, 0)
img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]

kernel = np.ones((3, 3), np.uint8)

img_erosion = cv2.erode(img, kernel, iterations=1)
img_dilation = cv2.dilate(img, kernel, iterations=1)

# cv2.imshow('Original',img)
# cv2.imshow('Eroded',img_erosion)
# cv2.imshow('Dilated',img_dilation)

cv2.imwrite('output/captcha-eroded.jpg', img_erosion)
cv2.imwrite('output/captcha-dilated.jpg', img_dilation)
