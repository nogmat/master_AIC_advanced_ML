from __future__ import absolute_import, division, print_function, \
    unicode_literals

import tensorflow as tf

import numpy as np

import netVLAD_v2

# a = tf.keras.layers.Input(shape=(1, 2))
# b = tf.keras.layers.Input(shape=(1, 2))
# c = tf.keras.layers.Input(shape=(1, 2))
# added = Addition()([a, b, c])
# model = tf.keras.models.Model(inputs=[a, b, c], outputs=added)
# print(model.predict([np.array([[[1, 1]]]),
#                      np.array([[[1, 1]]]),
#                      np.array([[[1, 1]]])]))
b = tf.constant([[[1, 2], [3, 4]],
                 [[5, 6], [7, 8]]])
c = tf.constant([[1.0, 2.0], [3.0, 4.0]])
# print(b.shape)
# b = tf.keras.backend.reshape(b, shape=tuple(b.shape + [1]))
# print(b)
# b = tf.keras.backend.repeat_elements(b, 2, axis=3)
# print(b)
# c = tf.transpose(c)
# print(c)
# d = tf.math.subtract(b, c)
# print(d)
e = tf.keras.backend.reshape(tf.transpose(c), shape=(4, 1))
print(e)
print(tf.math.l2_normalize(e, axis=0, epsilon=1e-12, name=None))

# # Tests
# a = tf.Variable([[[4.0, 2.0], [2.0, 1.0]],
#                  [[1.0, 0.0], [3.0, 2.0]]
#                  ])
# c1 = tf.Variable([1.0, 0.0])
# c2 = tf.Variable([0.0, 1.0])

# def diff(a, c):
#     v = a-c
#     v = tf.math.multiply(a-c, a-c)
#     v = tf.keras.backend.sum(v, axis=2, keepdims=True)
#     # v = tf.math.exp(-v)
#     return v

# S_ij1 = diff(a, c1)
# print(S_ij1)
# S_ij2 = diff(a, c2)
# print(S_ij2)
# s_ijsum = tf.math.add_n([S_ij1, S_ij2])
# print(s_ijsum)
# S_ijk = tf.keras.backend.concatenate([S_ij1, S_ij2], axis=2)
# print(S_ijk)
# S_ijk = tf.transpose(S_ijk)
# s_ijsum = tf.transpose(s_ijsum)
# S_ijk /= s_ijsum
# S_ijk = tf.transpose(S_ijk)
# print(S_ijk)
# # print(model.predict([[2, 3]]))

# Define VGG-16
# Pre-train using ImageNet dataset
# Define NetVLAD and adapt to the equations of descriptors
#   Don't forget loss functions
# Define discriminators
