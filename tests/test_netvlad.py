from lfp4uda.netvlad import NetVLAD
import numpy as np
import pytest
import tensorflow as tf


class TestNetVLAD:

    def testcall(self):

        features = np.array(
            [[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]
        )
        kmeans_centers = np.array([[[1.0, 1.0], [2.0, 3.0]]])
        f = tf.keras.layers.Input(shape=(2, 2, 2))
        k = tf.keras.layers.Input(shape=(2, 2))
        netvlad = NetVLAD(1, kmeans_centers)
        model = tf.keras.models.Model(inputs=[f, k], outputs=netvlad)
        print(model([features, kmeans_centers]))
