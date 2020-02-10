#!/bin/python

from __future__ import absolute_import, division, print_function, \
    unicode_literals

import numpy as np
import tensorflow as tf


class SingleLayerClassifier(tf.keras.layers.Layer):

    def __init__(self, nb_of_class, **kwargs):
        super(SingleLayerClassifier, self).__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(nb_of_class)

    def call(self, inputs):
        return self.dense(inputs)
