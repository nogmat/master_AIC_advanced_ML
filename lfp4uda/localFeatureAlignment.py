#!/bin/python

from __future__ import absolute_import, division, print_function, \
    unicode_literals

import numpy as np
import tensorflow as tf


class LocalFeatureAlignment(tf.keras.layers.Layer):

    def __init__(self, feature_shape, kmeans_shape, **kwargs):
        i, j, d = feature_shape
        k, d_ = kmeans_shape
        assert(d == d_)
        super(LocalFeatureAlignment, self).__init__(self)

    def call(self, inputs):
        distance, kmeans_centers, similarities = inputs
        hard_assign = tf.keras.backend.argmax(
            similarities,
            axis=-1)
        return [hard_assign]


if __name__ == "__main__":
    # s = tf.constant([[[1.0, 2.0], [3.0, 4.0]],
    #                  [[5.0, 6.0], [7.0, 8.0]]])
    # print(tf.keras.backend.argmax(s, axis=2))
    # print(tf.keras.backend.permute_dimensions(s, pattern=(0, 2, 1)))

    # features = tf.keras.layers.Input(shape=(2, 2, 2))
    # centers = tf.keras.layers.Input(shape=(2, 2))
    # similarities = tf.keras.layers.Input(shape=(2, 2, 2))
    # localFeatureAlignment = LocalFeatureAlignment((2, 2, 2), (2, 2))(
    #     [
    #         features,
    #         centers,
    #         similarities
    #     ])

    # features_t = np.array([[[[1.0, 2.0], [3.0, 4.0]],
    #                         [[5.0, 6.0], [7.0, 8.0]]]])
    # kmeans_centers_t = np.array([[[1.0, 1.0], [2.0, 3.0]]])
    # similarities_t = np.array([
    #     [[[7.3105860e-01, 2.6894143e-01],
    #       [9.9330717e-01, 6.6928510e-03]],
    #      [[9.9987662e-01, 1.2339458e-04],
    #       [9.9999774e-01, 2.2603242e-06]]]
    # ])
    # model = tf.keras.models.Model(
    #     inputs=[features, centers, similarities],
    #     outputs=localFeatureAlignment
    # )
    # print(model.predict([
    #     features_t,
    #     kmeans_centers_t,
    #     similarities_t
    # ]))

    features = tf.constant(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]]
        ]
    )
    hard_assign = tf.constant(
        [
            [0, 0],
            [0, 0]
        ]
    )
