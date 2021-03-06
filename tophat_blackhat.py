import cv2
import numpy as np
from utils import url_2_img

url = 'https://cdn.technologynetworks.com/tn/images/thumbs/jpeg/640_360/cancer-cells-vs-normal-cells-307366.jpg'
# Education purpose

img = url_2_img(url)
cv2.imwrite('dataset/cancer-vs-normal-cell.jpg', img)
img = cv2.threshold(img, 175, 255, cv2.THRESH_BINARY_INV)[1]

kernel = np.ones((51, 51), np.uint8)
img_tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)

kernel = np.ones((87, 87), np.uint8)
img_blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)

# cv2.imshow('Black and White', img)
# cv2.imshow('Tophat', img_tophat)
# cv2.imshow('Blackhat', img_blackhat)
# cv2.waitKey(0)

# cv2.imwrite('output/cancer-vs-normal-call-tophat.jpg', img_tophat)
# cv2.imwrite('output/cancer-vs-normal-call-blackhat.jpg', img_blackhat)

import matplotlib.pyplot as plt
plt.subplot(311)
plt.imshow(img, cmap = 'gray')
plt.title('The Original')

plt.subplot(312)
plt.imshow(img_tophat, cmap = 'binary_r')
plt.title('White top-hat transform')

plt.subplot(313)
plt.imshow(img_blackhat, cmap = 'binary_r')
plt.title('Bottom-hat transform')

plt.tight_layout()
# plt.show()
plt.savefig('output/cancer.png')
