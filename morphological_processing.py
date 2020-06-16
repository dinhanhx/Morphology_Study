import numpy as np
import cv2
import matplotlib.pyplot as plt

class morphological_processing:
    def __init__(self):
        pass

    def morph(self, baw_image, str_element, type = None):
        """
        Morphological processing of an image using a predefined structure element

        Args:
            str_element (2D numpy array): structuring element.
            Can be of any size (m, n) but odd numbers are recommended.
            Pixels can be zero or one, depending on the desired shape.

            baw_image (2D numpy array): black and white image.
            The background should be white and the object is in shades of black.

            type: "dilation", "erosion", "open", "close", "border", "tophat", "bottomhat"

        Returns:
            new_image (2D numpy array)

        """

        self.image = baw_image
        self.str_element = str_element

        if type == "dilation" or type == "erosion":
            return self.process(self.image, self.type)

        elif type == "open":
            return self.process(self.process(self.image, type="erosion"), type="dilation")

        elif type == "close":
            return self.process(self.process(self.image, type="dilation"), type="erosion")

        elif type == "border":
            return self.process(self.image, type="dilation") - self.process(self.image, type="erosion")

        elif type == "tophat":
            return self.image - self.morph(self.image, self.str_element, type="open")

        elif type == "bottomhat":
            return self.morph(self.image, self.str_element, type="close") - self.image


    def process(self, baw_image, type):

        baw_image = baw_image.astype(float)
        col, row = baw_image.shape
        new_image = np.zeros((col, row))

        # expand image for easy iteration
        neighbor_col, neighbor_row = [x//2 for x in self.str_element.shape]

        if neighbor_col == 1: baw_image = np.insert(baw_image, 0, np.nan, axis = 0)
        else: baw_image = np.insert(baw_image, np.zeros(neighbor_col), np.nan, axis = 0)

        if neighbor_row == 1: baw_image = np.insert(baw_image, 0, np.nan, axis = 1)
        else: baw_image = np.insert(baw_image, np.zeros(neighbor_row), np.nan, axis = 1)

        baw_image = np.insert(baw_image, np.full(neighbor_col, len(baw_image)), np.nan, axis = 0)
        baw_image = np.insert(baw_image, np.full(neighbor_row, len(baw_image[1])), np.nan, axis = 1)

        # take samples that have the same size as structure element
        for i in range(col):
            for j in range(row):
                sample = baw_image[i : i + self.str_element.shape[0], j : j + self.str_element.shape[1]]

                # map each sample with the "valid" pixels of structure element
                if sample.shape != self.str_element.shape:
                    print("fuck. error here:\n", sample.shape, self.str_element.shape)
                else: sample = sample[self.str_element != 0]

                # take the highest/lowest value of each sample and write to return image
                if type == "dilation": new_image[i, j] = np.nanmin(sample)
                if type == "erosion": new_image[i, j] = np.nanmax(sample)

        return new_image.astype(int)


if __name__ == "__main__":

    # cross-shape structure element size n = 0 (mod 3)
    n = 3
    str_element = np.ones((n, n))
    n = int(n/3)
    str_element[:n, :n] = 0
    str_element[:n, -n:] = 0
    str_element[-n:, :n] = 0
    str_element[-n:, -n:] = 0

    plt.imshow(str_element, cmap='gray', interpolation='bicubic')
    plt.xticks([]), plt.yticks([])
    # plt.show()

    # any black and white image
    baw_image = cv2.imread(r'test.webp', cv2.IMREAD_GRAYSCALE)
    plt.imshow(baw_image, cmap='gray', interpolation='bicubic')
    plt.xticks([]), plt.yticks([])
    # plt.show()

    new_image = morphological_processing()
    new_image = new_image.morph(baw_image, str_element, type="bottomhat")

    plt.imshow(new_image, cmap='gray', interpolation='bicubic')
    plt.xticks([]), plt.yticks([])
    plt.show()


