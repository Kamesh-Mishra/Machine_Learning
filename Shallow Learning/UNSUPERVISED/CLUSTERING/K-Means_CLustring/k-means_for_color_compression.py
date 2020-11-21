import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# https://jakevdp.github.io/PythonDataScienceHandbook/05.11-k-means.html

"""
One interesting application of clustering is in color compression within images.
  For example, imagine you have an image with millions of colors.
  In most images, a large number of the colors will be unused, and 
  many of the pixels in the image will have similar or even identical colors.

For example, consider the image shown in the following figure, 
which is from the Scikit-Learn datasets module (for this to work, you'll have
                                                to have the pillow Python package installed).

"""


# Note: this requires the ``pillow`` package to be installed
from sklearn.datasets import load_sample_image
china = load_sample_image("china.jpg")
ax = plt.axes(xticks=[], yticks=[])
ax.imshow(china);




"""
The image itself is stored in a three-dimensional array of size (height, width, RGB), 
containing red/blue/green contributions as integers from 0 to 255:
"""
    
china.shape

"""
One way we can view this set of pixels is as a cloud of points in a three-dimensional color space.
 We will reshape the data to [n_samples x n_features], and rescale the colors so that they lie between 0 and 1:
"""


data = china / 255.0 # use 0...1 scale
data = data.reshape(427 * 640, 3)
data.shape


# We can visualize these pixels in this color space, using a subset of 10,000 pixels for efficiency:

def plot_pixels(data, title, colors=None, N=10000):
    if colors is None:
        colors = data
    
    # choose a random subset
    rng = np.random.RandomState(0)
    i = rng.permutation(data.shape[0])[:N]
    colors = colors[i]
    R, G, B = data[i].T
    
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    ax[0].scatter(R, G, color=colors, marker='.')
    ax[0].set(xlabel='Red', ylabel='Green', xlim=(0, 1), ylim=(0, 1))

    ax[1].scatter(R, B, color=colors, marker='.')
    ax[1].set(xlabel='Red', ylabel='Blue', xlim=(0, 1), ylim=(0, 1))

    fig.suptitle(title, size=20);

plot_pixels(data, title='Input color space: 16 million possible colors')





"""
Now let's reduce these 16 million colors to just 16 colors,
 using a k-means clustering across the pixel space. Because we are dealing with a very 
 large dataset, we will use the mini batch k-means, which operates on subsets of the data to
 compute the result much more quickly than the standard k-means algorithm:
"""


import warnings; warnings.simplefilter('ignore')  # Fix NumPy issues.

from sklearn.cluster import MiniBatchKMeans
kmeans = MiniBatchKMeans(16)
kmeans.fit(data)
new_colors = kmeans.cluster_centers_[kmeans.predict(data)]

plot_pixels(data, colors=new_colors,
            title="Reduced color space: 16 colors")



"""
The result is a re-coloring of the original pixels, where each pixel is assigned the
 color of its closest cluster center. Plotting these new colors in the image space rather
 than the pixel space shows us the effect of this:
    
"""


china_recolored = new_colors.reshape(china.shape)

fig, ax = plt.subplots(1, 2, figsize=(16, 6),
                       subplot_kw=dict(xticks=[], yticks=[]))
fig.subplots_adjust(wspace=0.05)
ax[0].imshow(china)
ax[0].set_title('Original Image', size=16)
ax[1].imshow(china_recolored)
ax[1].set_title('16-color Image', size=16);




"""
Some detail is certainly lost in the rightmost panel, but the overall image is 
still easily recognizable. This image on the right achieves a compression factor of 
around 1 million! While this is an interesting application of k-means, there are certainly 
better way to compress information in images. But the example shows the power of thinking 
outside of the box with unsupervised methods like k-means.
"""

