#!/bin/python

from __future__ import absolute_import, division, print_function, \
    unicode_literals

import tensorflow as tf


class Addition(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(Addition, self).__init__(**kwargs)

    def call(self, x):
        assert(isinstance(x, list))
        assert(len(x) >= 2)
        return tf.math.add_n(x)


class NetVLAD(tf.keras.layers.Layer):

    def __init__(self, alpha, **kwargs):
        self.alpha = alpha
        self.trainable = True
        super(NetVLAD, self).__init__(**kwargs)

    def call(self, inputs):

        # TEST ENVIRONMENT ##
        # kmeans_centers = tf.constant([[1.0, 1.0], [2.0, 3.0]])
        # features = inputs
        # print(features.shape)
        # features = tf.keras.backend.repeat_elements(
        #     features,
        #     rep=2,
        #     axis=2)
        # features = tf.keras.backend.reshape(
        #     features,
        #     shape=(inputs.shape[0], 2, 2, 2, 2))
        # return [features]

        return [inputs]
        ######################

        # assert(isinstance(inputs, list))
        # assert(len(inputs) == 2)
        # features = tf.cast(inputs[0], dtype=tf.float32)
        # kmeans_centers = tf.cast(inputs[1], dtype=tf.float32)
        # assert(len(list(features.shape)) == 3)
        # assert(len(list(kmeans_centers.shape)) == 2)
        # assert(features.shape[-1] == kmeans_centers.shape[-1])

        # F = tf.keras.backend.reshape(
        #     tf.keras.backend.repeat_elements(
        #         features,
        #         rep=kmeans_centers.shape[0],
        #         axis=1),
        #     shape=(
        #         features.shape[0],
        #         features.shape[1],
        #         kmeans_centers.shape[0],
        #         features.shape[2]))
        # c = kmeans_centers
        # S = tf.math.exp(
        #     -self.alpha *
        #     tf.keras.backend.sum(
        #         tf.keras.backend.pow(
        #             tf.math.subtract(F, c),
        #             2),
        #         axis=3))
        # sum_S = tf.keras.backend.sum(
        #     S,
        #     axis=2)
        # S = tf.transpose(
        #     tf.math.divide(
        #         tf.transpose(S),
        #         tf.transpose(sum_S)))
        # V = tf.keras.backend.sum(
        #     tf.math.multiply(
        #         tf.transpose(tf.math.subtract(F, c)),
        #         tf.transpose(S)
        #     ),
        #     axis=2)
        # V_h = tf.math.l2_normalize(
        #     tf.keras.backend.reshape(
        #         tf.transpose(V),
        #         shape=(
        #             kmeans_centers.shape[0]*kmeans_centers.shape[1],
        #             1)),
        #     axis=0,
        #     epsilon=1e-12,
        #     name=None
        # )
        # return [V_h, S]


class SingleLayer(tf.keras.Model):

    def __init__(self, layer_):
        super(SingleLayer, self).__init__()
        self.layer = layer_

    def call(self, inputs):
        return self.layer(inputs)


if __name__ == "__main__":

    features = tf.constant([[[1.0, 2.0], [3.0, 4.0]],
                            [[5.0, 6.0], [7.0, 8.0]]])
    model = SingleLayer(NetVLAD(1))
    print(model.predict([features]))
    # sums = tf.keras.backend.repeat_elements(
    #     features,
    #     rep=2,
    #     axis=1)
    # print(sums)
    # sums = tf.keras.backend.reshape(
    #     sums,
    #     shape=(2, 2, 2, 2)
    # )
    # print(sums)
    # kmeans_centers = tf.constant([[1.0, 1.0], [2.0, 3.0]])
    # sums = tf.math.subtract(
    #     sums,
    #     kmeans_centers
    # )
    # print(sums)
    # model = SingleLayer(netVLAD)
