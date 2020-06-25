import cv2
import numpy as np
from matplotlib import pyplot as plt
plt.style.use('dark_background')

def plot_img_list(img_list, figure_name = None, save_me = False, show_me = True):
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

    if figure_name is not None and save_me == True:
        plt.savefig(figure_name, dpi = 300)

    return None

# Setup things
folder_path = 'dataset/captcha/'
fpath_list = [folder_path+str(i)+'.jpg' for i in range(10)]
kernel = np.ones((3,3), np.uint8)
transformation_list = [cv2.MORPH_OPEN, cv2.MORPH_CLOSE, cv2.MORPH_GRADIENT, cv2.MORPH_TOPHAT, cv2.MORPH_BLACKHAT]
figure_name_list = ['output/captcha/' + ele + '.jpg' for ele in ['open', 'close', 'gradient', 'white top hat', 'black top hat']]

# Get grayscale images as a list
img_list = [cv2.imread(fpath, 0) for fpath in fpath_list]

# Convert them to binary images
bin_img_list = [cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1] for img in img_list]

# Apply morphological transformations to them
# And save them
for transformation, figure_name in zip(transformation_list, figure_name_list):
    print('===')
    print(figure_name)
    app_img_list = [cv2.morphologyEx(img, transformation, kernel) for img in bin_img_list]
    plot_img_list(app_img_list, figure_name = figure_name, save_me = True, show_me = False)
