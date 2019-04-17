import numpy as np
import matplotlib.pyplot as plt
import skimage.data as data
import skimage.segmentation as seg
import skimage.filters as filters
import skimage.draw as draw
import skimage.color as color
from skimage import io


image = io.imread('Data/TL_41_10_129161_15.jpg', cmap='grey')
images = io.ImageCollection('Data/*.jpg')
plt.imshow(image)
plt.show()

image_thr = image > 180
plt.imshow(image_thr)
plt.show()
# image_threshold = filters.threshold_local(image, block_size=93, offset=16)
image_threshold = filters.threshold_local(image, block_size=51, offset=10)

image_seg = image > image_threshold
plt.imshow(image_seg)
plt.show()

image_threshold2 = filters.threshold_otsu(image, nbins=256)
image_seg2 = image > image_threshold2
plt.imshow(image_seg2)
plt.show()

fig = filters.try_all_threshold(image, figsize=(8, 5))
# plt.imshow(fig)
plt.show()

# images = io.ImageCollection('Data/*.jpg')
# print('Type:', type(images))
# images.files


