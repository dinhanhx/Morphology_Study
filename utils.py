import urllib
import cv2
import numpy

def url_2_img(url):
    """
    Get an image from url as img object in grayscale
    """
    resp = urllib.urlopen(url)
	image = np.asarray(bytearray(resp.read()), dtype="uint8")
	image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
    return image
