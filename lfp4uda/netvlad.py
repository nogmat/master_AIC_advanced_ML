#!/bin/python

from __future__ import absolute_import, division, print_function, \
    unicode_literals

import numpy as np
import tensorflow as tf

from tensorflow.python.ops import array_ops


class NetVLAD(tf.keras.layers.Layer):

    def __init__(self, alpha, **kwargs):
        self.alpha = alpha
        super(NetVLAD, self).__init__(**kwargs)

    def call(self, inputs):
        features, kmeans_centers = inputs
        _, i, j, d = features.shape
        _, k, _ = kmeans_centers.shape
        features = tf.keras.backend.repeat_elements(
            features,
            rep=k,
            axis=2)
        # features = self.reshape1(features)
        features = array_ops.reshape(
            features,
            (array_ops.shape(features)[0],)+(i, j, k, d))
        distance = tf.math.subtract(features, kmeans_centers)
        similarities = tf.keras.backend.softmax(
            -self.alpha *
            tf.keras.backend.sum(
                tf.keras.backend.pow(distance, 2),
                axis=3),
            axis=-1)
        similarities_repl = array_ops.reshape(
            tf.keras.backend.repeat_elements(
                similarities,
                rep=d,
                axis=3),
            (array_ops.shape(similarities)[0],)+(i, j, k, d)
        )
        vlad_extended = tf.keras.backend.sum(
            tf.keras.backend.sum(
                tf.math.multiply(similarities_repl, distance),
                axis=1
            ),
            axis=1
        )
        vlad = tf.keras.backend.dropout(
            tf.keras.backend.l2_normalize(
                array_ops.reshape(
                    vlad_extended,
                    (array_ops.shape(vlad_extended)[0],)+(k*d,))),
            0.5)
        return [vlad, similarities, distance]


def netVLAD(alpha, inputs):
    return NetVLAD(alpha)(inputs)


if __name__ == "__main__":

    features = np.array([[[[1.0, 2.0], [3.0, 4.0]],
                          [[5.0, 6.0], [7.0, 8.0]]]])
    kmeans_centers = np.array([[[1.0, 1.0], [2.0, 3.0]]])
    f = tf.keras.layers.Input(shape=(2, 2, 2))
    k = tf.keras.layers.Input(shape=(2, 2))
    netvlad = netVLAD(1, [f, k])
    model = tf.keras.models.Model(inputs=[f, k], outputs=netvlad)
    print(model([features, kmeans_centers]))
