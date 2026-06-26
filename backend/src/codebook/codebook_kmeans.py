import numpy as np
from sklearn.cluster import KMeans


class VectorCodebook:
    def __init__(self, n_clusters, random_state=42):
        self.n_clusters= n_clusters
        self.random_state= random_state
        self.centroids= None

    def build_codebook(self, features_matriz):
        kmeans= KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init='auto')
        kmeans.fit(features_matriz)
        self.centroids= kmeans.cluster_centers_
        return self.centroids