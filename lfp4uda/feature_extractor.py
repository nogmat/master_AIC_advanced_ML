#!/bin/python

from __future__ import absolute_import, division, print_function, \
    unicode_literals

import numpy as np
import tensorflow as tf


class FeatureExtractor(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(FeatureExtractor, self).__init__(**kwargs)
        self.vgg16 = tf.keras.applications.VGG16(
            weights="imagenet",
            include_top=False
        )
        # vgg16.layers[-3] is block5_conv2
        # vgg16.layers[-2] is block5_conv3
        # vgg16.layers[-1] is block5_pool
        for layer in self.vgg16.layers[:-2]:
            layer.trainable = False

    def call(self, inputs):
        return self.vgg16(inputs)
