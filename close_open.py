import cv2
import numpy as np
from utils import url_2_img

url = 'http://baochi.nlv.gov.vn/baochi/cgi-bin/imageserver/imageserver.pl?oid=Hueo19400701.2.3&area=1&width=700&color=all&ext=jpg&key='
# Bao chi gov vietnam
# Public domain

img = url_2_img(url)
cv2.imwrite('dataset/title.jpg', img)
img = cv2.threshold(img, 141, 255, cv2.THRESH_BINARY_INV)[1]

# kernel = np.ones((1, 1), np.uint8)
kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)

img_close = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

img_open = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

# cv2.imshow('Original', img)
# cv2.imshow('Close', img_close)
# cv2.imshow('Open', img_open)
# cv2.waitKey(0)

cv2.imwrite('output/title-closed.jpg', img_close)
cv2.imwrite('output/title-opened.jpg', img_open)
