#!/bin/python

import numpy as np
import tensorflow as tf

from tensorflow.python.ops import array_ops


class NetVLAD(tf.keras.layers.Layer):

    def __init__(self, alpha, kmeans_centers, **kwargs):
        self.alpha = alpha
        super(NetVLAD, self).__init__(**kwargs)
        self.centers = self.add_weight(
            shape=kmeans_centers.shape,
            initializer=tf.keras.initializers.Constant(value=kmeans_centers),
            trainable=False)

    def call(self, inputs):
        features, kmeans_centers = inputs, self.centers
        _, i, j, d = features.shape
        _, k, _ = kmeans_centers.shape
        features = tf.keras.backend.repeat_elements(
            features,
            rep=k,
            axis=2)
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
