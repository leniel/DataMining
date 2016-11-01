from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from numpy import genfromtxt
import numpy as np

#Matrix in the format Movies x User ratings
data = genfromtxt('\\Data\\ratings massaged 100.csv', delimiter=',')

#Movies titles
movies = genfromtxt('\\Data\\movies.csv', delimiter=',',
                    dtype=None, usecols=(0, 1))

pca = PCA(n_components=2, whiten=True).fit_transform(data)

k = 15

km = KMeans(init='k-means++', n_clusters=k, n_init=10)

km.fit(pca)

# Step size of the mesh. Decrease to increase the quality of the VQ.
h = 0.2  # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = pca[:, 0].min() - 1, pca[:, 0].max() + 1
y_min, y_max = pca[:, 1].min() - 1, pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = km.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.get_cmap('Paired'),
           aspect='auto', origin='lower')

plt.plot(pca[:, 0], pca[:, 1], 'k.', markersize=5)
# Plot the centroids as a white X
centroids = km.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.title('K-means clustering on the Movie Lens 1 MB dataset (PCA-reduced data)\n'
          'Centroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())

plt.show()
