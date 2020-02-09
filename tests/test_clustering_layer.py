from lfp4uda.clustering_layer import ClusteringLayer
import numpy as np
import pytest
from sklearn.cluster import KMeans
import tensorflow as tf


class TestClusteringLayer:
    def test_call(self):

        x_init = np.random.random((100, 4))
        kmeans = KMeans(n_clusters=32)
        kmeans.fit(x_init)

        x = np.random.random((2, 3, 4))

        y_expected = np.reshape(
            np.array(
                [
                    kmeans.cluster_centers_[i]
                    for i in kmeans.predict(np.reshape(x, (6, 4)))
                ]
            ),
            (2, 3, 4),
        )

        sess = tf.Session()
        with sess.as_default():

            [x_out, y_out] = ClusteringLayer(
                kmeans=kmeans, input_shape=(2, 3, 4)
            )(tf.convert_to_tensor(x))

            assert np.array_equal(x, x_out)
            assert np.array_equal(y_expected, y_out)


TestClusteringLayer().test_call()
