# morphological_processing.py

## Requirements
Python 3

Site packages:
  - matplotlib
  - numpy
  - opencv-python

## Documentation
### morphological_processing.__init__()
```Python
morphological_processing.__init__(self, baw_image)
  """
    Create instance to process an image

    baw_image (2D numpy array): black and white image.
    Better if image is binary. Should perform thresholding on image first.
    The background should be white and the object is in shades of black.

    """
```
### morphological_processing.morph()
```Python
morphological_processing.morph(self, str_element, type = None)
  """
  Morphological processing of an image using a predefined structure element

  Args:
      str_element (2D numpy array): structuring element.
      Can be of any size (m, n) but odd numbers are recommended.
      Pixels can be zero or one, depending on the desired shape.

      type: "dilation", "erosion", "open", "close", "gradient", "tophat", "bottomhat"

  Returns:
      new_image (2D numpy array)

  """
```
## Author

[Huong Larne](https://github.com/huonglarne)
