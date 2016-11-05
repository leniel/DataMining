'''
 Created on Sat Nov 05 2016

 Copyright (c) 2016 Leniel Macaferi's Consulting
'''

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from numpy import genfromtxt

import numpy as np

# Matrix in the format Movies x User ratings
data = genfromtxt('\\Data\\ratings massaged.csv', delimiter=',')

pca = PCA(n_components=10, whiten=True).fit_transform(data)

for x in range(1, 30):

    k = x

    km = KMeans(n_clusters=k, init='k-means++', n_init=10, verbose=0)

    km.fit(pca)

    print(km.inertia_)
