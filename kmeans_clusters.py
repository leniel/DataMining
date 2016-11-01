from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from numpy import genfromtxt

import numpy as np

#Matrix in the format Movies x User ratings
data = genfromtxt('\\Data\\ratings massaged.csv', delimiter=',')

#Movies titles
movies = genfromtxt('\\Data\\movies.csv', delimiter=',', dtype=None, usecols=(0, 1))

pca = PCA(n_components = 10, whiten = True).fit_transform(data)

k = 30

km = KMeans(n_clusters = k, init = 'k-means++', n_init = 10, verbose = 0)

km.fit(pca)

clusters = km.fit_predict(pca)

 # Finding the Movies in each predicted cluster...
cluster_1 = np.where(clusters==0)
cluster_2 = np.where(clusters==1)
cluster_3 = np.where(clusters==2)
cluster_4 = np.where(clusters==3)
cluster_5 = np.where(clusters==4)
cluster_6 = np.where(clusters==5)
cluster_7 = np.where(clusters==6)
cluster_8 = np.where(clusters==7)
cluster_9 = np.where(clusters==8)
cluster_10 = np.where(clusters==9)
cluster_11 = np.where(clusters==10)
cluster_12 = np.where(clusters==11)
cluster_13 = np.where(clusters==12)
cluster_14 = np.where(clusters==13)
cluster_15 = np.where(clusters==14)
cluster_16 = np.where(clusters==15)
cluster_17 = np.where(clusters==16)
cluster_18 = np.where(clusters==17)
cluster_19 = np.where(clusters==18)
cluster_20 = np.where(clusters==19)
cluster_21 = np.where(clusters==20)
cluster_22 = np.where(clusters==21)
cluster_23 = np.where(clusters==22)
cluster_24 = np.where(clusters==23)
cluster_25 = np.where(clusters==24)
cluster_26 = np.where(clusters==25)
cluster_27 = np.where(clusters==26)
cluster_28 = np.where(clusters==27)
cluster_29 = np.where(clusters==28)
cluster_30 = np.where(clusters==29)

cf = open('clusters.txt', 'w')

cf.write('Cluster 1\n')
for movie in cluster_1:
    for m in movie:
      cf.write("%s\n" % movies[m][1])

cf.write('\nCluster 2\n')
for movie in cluster_2:
    for m in movie:
      cf.write("%s\n" % movies[m][1])

cf.write('\nCluster 3\n')
for movie in cluster_3:
    for m in movie:
      cf.write("%s\n" % movies[m][1])

cf.write('\nCluster 4\n')
for movie in cluster_4:
    for m in movie:
      cf.write("%s\n" % movies[m][1])

cf.write('\nCluster 5\n')
for movie in cluster_5:
    for m in movie:
      cf.write("%s\n" % movies[m][1])

cf.write('\nCluster 6\n')
for movie in cluster_6:
    for m in movie:
      cf.write("%s\n" % movies[m][1])

cf.write('\nCluster 7\n')
for movie in cluster_7:
    for m in movie:
      cf.write("%s\n" % movies[m][1])

cf.write('\nCluster 8\n')
for movie in cluster_8:
    for m in movie:
      cf.write("%s\n" % movies[m][1])

cf.write('\nCluster 9\n')
for movie in cluster_9:
    for m in movie:
      cf.write("%s\n" % movies[m][1])

cf.write('\nCluster 10\n')
for movie in cluster_10:
    for m in movie:
      cf.write("%s\n" % movies[m][1])

cf.write('\nCluster 11\n')
for movie in cluster_11:
    for m in movie:
      cf.write("%s\n" % movies[m][1])

cf.write('\nCluster 12\n')
for movie in cluster_12:
    for m in movie:
      cf.write("%s\n" % movies[m][1])

cf.write('\nCluster 13\n')
for movie in cluster_13:
    for m in movie:
      cf.write("%s\n" % movies[m][1])

cf.write('\nCluster 14\n')
for movie in cluster_14:
    for m in movie:
      cf.write("%s\n" % movies[m][1])

cf.write('\nCluster 15\n')
for movie in cluster_15:
    for m in movie:
      cf.write("%s\n" % movies[m][1])

cf.write('\nCluster 16\n')
for movie in cluster_16:
    for m in movie:
      cf.write("%s\n" % movies[m][1])

cf.write('\nCluster 17\n')
for movie in cluster_17:
    for m in movie:
      cf.write("%s\n" % movies[m][1])

cf.write('\nCluster 18\n')
for movie in cluster_18:
    for m in movie:
      cf.write("%s\n" % movies[m][1])

cf.write('\nCluster 19\n')
for movie in cluster_19:
    for m in movie:
      cf.write("%s\n" % movies[m][1])

cf.write('\nCluster 20\n')
for movie in cluster_20:
    for m in movie:
      cf.write("%s\n" % movies[m][1])

cf.write('\nCluster 21\n')
for movie in cluster_21:
    for m in movie:
      cf.write("%s\n" % movies[m][1])

cf.write('\nCluster 22\n')
for movie in cluster_22:
    for m in movie:
      cf.write("%s\n" % movies[m][1])

cf.write('\nCluster 23\n')
for movie in cluster_23:
    for m in movie:
      cf.write("%s\n" % movies[m][1])

cf.write('\nCluster 24\n')
for movie in cluster_24:
    for m in movie:
      cf.write("%s\n" % movies[m][1])

cf.write('\nCluster 25\n')
for movie in cluster_25:
    for m in movie:
      cf.write("%s\n" % movies[m][1])

cf.write('\nCluster 26\n')
for movie in cluster_26:
    for m in movie:
      cf.write("%s\n" % movies[m][1])

cf.write('\nCluster 27\n')
for movie in cluster_27:
    for m in movie:
      cf.write("%s\n" % movies[m][1])

cf.write('\nCluster 28\n')
for movie in cluster_28:
    for m in movie:
      cf.write("%s\n" % movies[m][1])

cf.write('\nCluster 29\n')
for movie in cluster_29:
    for m in movie:
      cf.write("%s\n" % movies[m][1])

cf.write('\nCluster 30\n')
for movie in cluster_30:
    for m in movie:
      cf.write("%s\n" % movies[m][1])