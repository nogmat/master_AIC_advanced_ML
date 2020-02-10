from lfp4uda.clustering_layer import ClusteringLayer
import numpy as np
import pytest
import tensorflow as tf
from tensorflow.contrib.factorization import KMeansClustering

tf.get_logger().setLevel('ERROR')


class TestClusteringLayer:
    def test_call(self):

        x_init = tf.convert_to_tensor(
            np.random.random((100, 4)), dtype=tf.float32
        )
        kmeans = KMeansClustering(32)
        kmeans.train(
            lambda: tf.compat.v1.train.limit_epochs(x_init, num_epochs=1)
        )

        x = tf.convert_to_tensor(np.random.random((2, 3, 4)), dtype=tf.float32)

        y_expected = tf.convert_to_tensor(
            np.reshape(
                np.array(
                    [
                        kmeans.predict(lambda v: v)
                        for v in np.reshape(x, (6, 4))
                    ]
                ),
                (2, 3, 4),
            ),
            dtype=tf.float32,
        )

        x_out, y_out = ClusteringLayer(kmeans=kmeans, input_shape=(2, 3, 4))(x)

        with tf.Session() as sess:

            assert sess.run(tf.reduce_all(tf.equal(x, x_out)))
            assert sess.run(tf.reduce_all(tf.equal(y_expected, y_out)))


TestClusteringLayer().test_call()
