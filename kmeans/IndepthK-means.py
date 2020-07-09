'''
Clustering: Clustering algorithms seek to learn, from the properties
of the data, an optimal division or discrete labeling of groups of points. 

'''

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()# for plot styling
import numpy as np
from sklearn.datasets.samples_generator import make_blobs

# generating 2D dataset containing 4 distinct blobs
X, y_true = make_blobs(n_samples = 300, centers = 4, cluster_std = 0.60, random_state = 0)
print(X.shape)
print(X)
print(y_true)
plt.scatter(X[:,0], X[:,1], s = 30) # s is size of scattered dots
#plt.show()

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 4)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

# plotting predicted values with centers
plt.scatter(X[:,0], X[:,1], c = y_kmeans, s = 30)
centers = kmeans.cluster_centers_
plt.scatter(centers[:,0], centers[:,1], c = "black", s = 100, alpha = 0.3) # alpha is level of transparency
plt.show()
