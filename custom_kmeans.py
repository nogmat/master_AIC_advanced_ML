from sklearn.cluster import KMeans
import tensorflow as tf


class CustomKMeans:
    def __init__(self, n_cluster=32):
        self.n_clusters = n_cluster
        self.kmeans = KMeans(n_clusters=self.n_clusters)

    def fit(self, Fs):
        self.kmeans.fit(Fs)

    def predict(self, F):
        return (self.kmeans.predict(F), F)
