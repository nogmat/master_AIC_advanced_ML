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


if __name__ == "__main__":

    aggregated_residuals = np.array(
        [[0.6064591,  0.7948939, -0.01324965, -0.01324965]])

    input_1 = tf.keras.layers.Input(shape=(4,))
    classifier = SingleLayerClassifier(2)(input_1)
    model = tf.keras.models.Model(inputs=[input_1], outputs=classifier)
    print(model([aggregated_residuals]))
