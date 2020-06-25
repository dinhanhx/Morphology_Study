import cv2
import numpy as np
from matplotlib import pyplot as plt
plt.style.use('dark_background')

def plot_img_list(img_list, figure_name = None, show_me = True):
    """
    Draw list of images then show and save figure.
    """
    plt.figure()
    for i, img in zip(range(1, 11), img_list):
        plt.subplot(2, 5, i)
        plt.imshow(img, cmap = 'gray')

    plt.gcf().set_size_inches(8, 4.5)
    plt.tight_layout()
    if show_me:
        plt.show()

    if figure_name is not None:
        plt.savefig(figure_name, dpi = 300)

    return None

# Setup things
folder_path = 'dataset/captcha/'
fpath_list = [folder_path+str(i)+'.jpg' for i in range(10)]
transformation_list = [cv2.MORPH_TOPHAT, cv2.MORPH_BLACKHAT]
figure_name_list = ['output/captcha/' + ele + '.png' for ele in ['top hat', 'black hat']]
kernel = np.ones((3,3), np.uint8)

# Get grayscale images as a list
img_list = [cv2.imread(fpath, 0) for fpath in fpath_list]

# Convert them to binary images
bin_img_list = [cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)[1] for img in img_list]

plot_img_list(bin_img_list, figure_name = 'dataset/captcha/original.png', show_me = False)

# Apply morphological transformations to them
# And save them
for transformation, figure_name in zip(transformation_list, figure_name_list):
    print('===')
    print(figure_name)
    app_img_list = [cv2.morphologyEx(img, transformation, kernel) for img in bin_img_list]
    plot_img_list(app_img_list, figure_name = figure_name, show_me = False)
