import numpy as np
import cv2
import matplotlib.pyplot as plt

n = 3
strel = np.ones((n, n))
# structuring element, can change to different shapes like cross or something
# does not have to be square

baw_img = cv2.imread(r'dataset/hand/1.png', cv2.IMREAD_GRAYSCALE)
# any image

def morph (baw_image, strel, type = None):

    baw_image = baw_image.astype(float)

    col, row = baw_image.shape
    new_image = np.zeros((col, row))

    nbh_row_full, nbh_col_full = strel.shape
    nbh_row = nbh_row_full // 2
    nbh_col = nbh_col_full // 2

    if nbh_row == 1:
    	baw_image = np.insert(baw_image, 0, np.nan, axis = 0) 
    else: baw_image = np.insert(baw_image, np.zeros(nbh_row), np.nan, axis = 0)
    if nbh_col == 1: baw_image = np.insert(baw_image, 0, np.nan, axis = 1)
    else: baw_image = np.insert(baw_image, np.zeros(nbh_col), np.nan, axis = 1)
    baw_image = np.insert(baw_image, np.full(nbh_row, len(baw_image)), np.nan, axis = 0)
    baw_image = np.insert(baw_image, np.full(nbh_col, len(baw_image[1])), np.nan, axis = 1)

    for i in range(col):
        for j in range(row):
            sample = baw_image[i:i+nbh_col_full, j:j+nbh_row_full]
            if type == "ero": new_image[i, j] = np.nanmin(sample)
            if type == "dil": new_image[i, j] = np.nanmax(sample)
    return new_image.astype(int)

new_img = morph(baw_img, strel, type="dil")
plt.imshow(new_img, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.show()
