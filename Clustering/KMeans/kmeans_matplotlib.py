'''
 Created on Sat Nov 05 2016

 Copyright (c) 2016 Leniel Macaferi's Consulting
'''

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from matplotlib import pyplot
from numpy import genfromtxt
import numpy as np

# Matrix in the format Movies x User ratings
data = genfromtxt('\\Data\\ratings massaged.csv', delimiter=',')

# Movies titles
movies = genfromtxt('\\Data\\movies.csv', delimiter=',',
                    dtype=None, usecols=(0, 1))

pca = PCA(n_components=10, whiten=True).fit_transform(data)

k = 15

km = KMeans(init='k-means++', n_clusters=k, n_init=10)

km.fit(pca)

labels = km.labels_
centroids = km.cluster_centers_

for i in range(k):
    # select only data observations with cluster label == i
    ds = data[np.where(labels == i)]

    # plot the data observations
    pyplot.plot(ds[:, 0], ds[:, 1], 'o')

    # plot the centroids
    lines = pyplot.plot(centroids[i, 0], centroids[i, 1], 'kx')

    # make the centroid x's bigger
    pyplot.setp(lines, ms=15.0)
    pyplot.setp(lines, mew=2.0)

pyplot.show()
