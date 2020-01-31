#!/bin/python

from __future__ import absolute_import, division, print_function, \
    unicode_literals

import tensorflow as tf


class Addition(tf.keras.layers.Layer):
    def __init__(self):
        super(Addition, self).__init__()

    def call(self, inputs):
        print(inputs.shape)
        return tf.keras.backend.sum(inputs, keepdims=True)


class NetVLAD(tf.keras.layers.Layer):

    def __init__(self, alpha, kmeans, **kwargs):
        self.alpha = 5000
        self.kmeans = kmeans
        super(NetVLAD, self).__init__(**kwargs)

    def diff(self, a, c):
        v = tf.keras.backend.sum(tf.math.multiply(
            a-c, a-c), axis=2, keepdims=True)
        v = tf.math.exp(-self.alpha * v)
        return v

    def similarities(self, inputs):
        S_ij_k = []
        assert(inputs.shape[:-1] == self.kmeans.cluster_centers_[:-1])
        for c_k in self.kmeans.cluster_centers_:
            S_ij_k.append(self.diff(inputs, c_k))
        S_ij_sum = tf.math.add_n(S_ij_k)
        for k in range(len(S_ij_k)):
            S_ij_k[k] = tf.math.divide(S_ij_k[k], S_ij_sum)
        return tf.keras.backend.concatenate(S_ij_k, axis=2)

    def residuals(self, inputs):
        R_ij_k = []
        for c_k in self.kmeans.cluster_centers_:
            R_ij_k.append(inputs - tf.Variable(c_k))
        R_ij_k = tf.keras.backend.concatenate(R_ij_k, axis=2)

    def call(self, inputs):
        # inputs represents F
        S = self.similarities(inputs)
        R = self.residuals(inputs)
        St, Rt = tf.transpose(S), tf.transpose(R)
        ARt = tf.math.multiply(St, Rt)
        return [tf.keras.backend.sum(ARt, axis=1)]


if __name__ == "__main__":

    # Tests
    a = tf.Variable([[[4.0, 2.0], [2.0, 1.0]],
                     [[1.0, 0.0], [3.0, 2.0]]
                     ])
    c1 = tf.Variable([1.0, 0.0])
    c2 = tf.Variable([0.0, 1.0])

    def diff(a, c):
        v = a-c
        v = tf.math.multiply(a-c, a-c)
        v = tf.keras.backend.sum(v, axis=2, keepdims=True)
        # v = tf.math.exp(-v)
        return v

    S_ij1 = diff(a, c1)
    print(S_ij1)
    S_ij2 = diff(a, c2)
    print(S_ij2)
    s_ijsum = tf.math.add_n([S_ij1, S_ij2])
    print(s_ijsum)
    S_ijk = tf.keras.backend.concatenate([S_ij1, S_ij2], axis=2)
    print(S_ijk)
    S_ijk = tf.transpose(S_ijk)
    s_ijsum = tf.transpose(s_ijsum)
    S_ijk /= s_ijsum
    S_ijk = tf.transpose(S_ijk)
    print(S_ijk)
    # print(model.predict([[2, 3]]))

    # Define VGG-16
    # Pre-train using ImageNet dataset
    # Define NetVLAD and adapt to the equations of descriptors
    #   Don't forget loss functions
    # Define discriminators
