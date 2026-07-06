import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans


class VectorCodebook:
    """Codebook compartido para imagen (SIFT) y audio (MFCC) via K-Means.

    minibatch=True usa MiniBatchKMeans (mas rapido, aproximado) para
    colecciones grandes; minibatch=False (default) usa KMeans exacto.
    Cualquier kwarg extra (n_init, max_iter, batch_size, ...) se reenvia
    tal cual al estimador de sklearn subyacente.
    """

    def __init__(self, n_clusters, random_state=42, minibatch=False, **kmeans_kwargs):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.minibatch = minibatch
        self.kmeans_kwargs = {"n_init": "auto", **kmeans_kwargs}
        self.centroids = None
        self._model = None

    def build_codebook(self, features_matriz):
        model_cls = MiniBatchKMeans if self.minibatch else KMeans
        self._model = model_cls(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            **self.kmeans_kwargs,
        )
        self._model.fit(features_matriz)
        self.centroids = self._model.cluster_centers_
        return self.centroids

    def predict(self, features):
        """Asigna cada descriptor/frame nuevo al codeword (centroide) mas cercano."""
        if self._model is None:
            raise RuntimeError("Llamar a build_codebook() antes de predict()")
        return self._model.predict(features)