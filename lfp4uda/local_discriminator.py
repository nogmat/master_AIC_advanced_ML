from __future__ import absolute_import, division, print_function, \
    unicode_literals

import numpy as np
import tensorflow as tf


class LocalDiscriminator(tf.keras.layers.Layer):

    def __init__(self, nunits_1, nunits_2, **kwargs):
        super(LocalDiscriminator, self).__init__(**kwargs)
        self.dense_1 = tf.keras.layers.Dense(nunits_1, activation='relu')
        self.dense_2 = tf.keras.layers.Dense(nunits_2, activation='relu')
        self.dense_3 = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        output = self.dense_1(inputs)
        output = self.dense_2(output)
        output = self.dense_3(output)
        return [output]


if __name__ == "__main__":

    input_ = tf.keras.layers.Input(shape=(2, 2, 3))
    dense = LocalDiscriminator(2048, 4096)(input_)
    model = tf.keras.models.Model(inputs=[input_], outputs=dense)

    aligned_residuals = np.array([[[1., -1., 0.],
                                   [4., -4., 1.],
                                   [5., -5., 0.],
                                   [7., -7., 0.]]])

    print(model([aligned_residuals]))
