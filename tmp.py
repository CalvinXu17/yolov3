

import cv2

from PIL import Image



img = cv2.imread('data/samples/bus.jpg')

print(img.shape)

img = Image.open('data/samples/bus.jpg')

import numpy as np

print(np.array(img).shape)

print()