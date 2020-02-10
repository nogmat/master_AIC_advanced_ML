from __future__ import absolute_import, division, print_function, \
    unicode_literals

import numpy as np
import tensorflow as tf


class HolisticDiscriminator(tf.keras.layers.Layer):

    def __init__(self, nunits_1, nunits_2, **kwargs):
        super(HolisticDiscriminator, self).__init__(**kwargs)
        self.dense_1 = tf.keras.layers.Dense(nunits_1, activation='relu')
        self.dense_2 = tf.keras.layers.Dense(nunits_2, activation='relu')
        self.dense_3 = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        output = self.dense_1(inputs)
        output = self.dense_2(output)
        output = self.dense_3(output)
        return [output]
