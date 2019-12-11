import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

def DBSCANClustering(X, peps, pmin_sample):
    clustering = DBSCAN(eps=peps, min_samples=pmin_sample).fit(X)
    labels = clustering.labels_

    for x in range(len(X)):
        color = 'C' + str(8-labels[x])
        plt.plot(X[x], 0, color=color, marker='o')

    plt.show()
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    return labels

def KMeansClustering(X, clusters):
    kmeans = KMeans(n_clusters=clusters)
    kmeans = kmeans.fit(X)
    labels = kmeans.predict(X)
    return labels