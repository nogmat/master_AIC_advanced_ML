#!/bin/python

from __future__ import absolute_import, division, print_function, \
    unicode_literals

import numpy as np
import tensorflow as tf


class NetVLAD(tf.keras.layers.Layer):

    def __init__(self, alpha, features_shape, kmeans_shape, **kwargs):
        i, j, d = features_shape
        k, d = kmeans_shape
        self.alpha = alpha
        self.trainable = True
        self.reshape1 = tf.keras.layers.Reshape(
            (i, j, k, d), input_shape=(i, j*k, d))
        self.reshape2 = tf.keras.layers.Reshape(
            (i, j, k, d), input_shape=(i, j, k*d))
        self.flatten = tf.keras.layers.Flatten()
        super(NetVLAD, self).__init__(**kwargs)

    def call(self, inputs):

        features, kmeans_centers = inputs
        _, i, j, d = features.shape
        _, k, _ = kmeans_centers.shape
        features = tf.keras.backend.repeat_elements(
            features,
            rep=k,
            axis=2)
        features = self.reshape1(features)
        distance = tf.math.subtract(features, kmeans_centers)
        similarities = tf.keras.backend.softmax(
            -self.alpha *
            tf.keras.backend.sum(
                tf.keras.backend.pow(distance, 2),
                axis=3),
            axis=-1)
        similarities_repl = self.reshape2(
            tf.keras.backend.repeat_elements(
                similarities,
                rep=d,
                axis=3))
        vlad_extended = tf.keras.backend.sum(
            tf.keras.backend.sum(
                tf.math.multiply(similarities_repl, distance),
                axis=1
            ),
            axis=1
        )
        vlad = tf.keras.backend.l2_normalize(
            self.flatten(vlad_extended))
        return [vlad]


if __name__ == "__main__":

    features = np.array([[[[1.0, 2.0], [3.0, 4.0]],
                          [[5.0, 6.0], [7.0, 8.0]]]])
    kmeans_centers = np.array([[[1.0, 1.0], [2.0, 3.0]]])
    f = tf.keras.layers.Input(shape=(2, 2, 2))
    k = tf.keras.layers.Input(shape=(2, 2))
    netvlad = NetVLAD(1, (2, 2, 2), (2, 2))([f, k])
    model = tf.keras.models.Model(inputs=[f, k], outputs=netvlad)
    print(model.predict([features, kmeans_centers]))
