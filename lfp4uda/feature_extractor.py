#!/bin/python

from __future__ import absolute_import, division, print_function, \
    unicode_literals

import numpy as np
import tensorflow as tf


class FeatureExtractor(tf.keras.layers.Layer):

    def __init__(self, training=False, **kwargs):
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
        self.vgg16.layers[-2] = training
        self.vgg16.layers[-1] = training

    def set_trainable(self, trainable=True):
        for layer in self.vgg16.layers[-2:]:
            layer.trainable = trainable

    def call(self, inputs):
        return self.vgg16(inputs)
