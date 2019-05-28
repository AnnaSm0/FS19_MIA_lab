import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from skimage import io
import skimage.filters as filters


image = io.imread('Data/TL_41_10_129161_15.jpg', cmap='grey')

image_threshold = filters.threshold_local(image, block_size=51, offset=10)

image_seg = image > image_threshold

mask = image_seg.astype(np.float)


hist, bin_edges = np.histogram(image, bins=60)
bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])

plt.figure(figsize=(11,4))

plt.subplot(131)
plt.imshow(image)
plt.axis('off')
plt.subplot(132)
plt.plot(bin_centers, hist, lw=2)
plt.axvline(0.5, color='r', ls='--', lw=2)
plt.text(0.57, 0.8, 'histogram', fontsize=20, transform = plt.gca().transAxes)
plt.yticks([])
plt.subplot(133)
plt.imshow(mask, cmap=plt.cm.gray, interpolation='nearest')
plt.axis('off')

plt.subplots_adjust(wspace=0.02, hspace=0.3, top=1, bottom=0.1, left=0, right=1)
plt.show()
