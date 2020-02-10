#!/bin/python

from __future__ import absolute_import, division, print_function, \
    unicode_literals

import numpy as np
import tensorflow as tf

from tensorflow.python.ops import array_ops


class LocalFeatureAlignment(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(LocalFeatureAlignment, self).__init__(self, **kwargs)

    def call(self, inputs):
        distance, similarities = inputs
        _, i, j, k, d = distance.shape
        _, i, j, k_ = similarities.shape
        assert(k == k_)
        distance = array_ops.reshape(
            distance,
            (array_ops.shape(distance)[0],)+(i*j, k, d))
        argmx = tf.cast(
            array_ops.reshape(
                tf.keras.backend.argmax(similarities),
                (array_ops.shape(similarities)[0],)+(i*j, 1)),
            dtype=tf.int32)
        ones = tf.cast(tf.keras.backend.ones_like(argmx), dtype=tf.int32)
        selector = tf.concat(
            [tf.math.multiply(
                ones,
                tf.keras.backend.reshape(tf.range(i*j), shape=(i*j, 1))),
             argmx], axis=-1)
        residuals = tf.gather_nd(distance, selector, batch_dims=1)
        aligned_residuals = tf.concat(
            [residuals, tf.cast(argmx, dtype=tf.float32)],
            axis=-1)
        return [aligned_residuals]
