import numpy as np
import tensorflow as tf


class LossSimilarities(tf.keras.losses.Loss):

    def __init__(self, **kwargs):
        super(LossSimilarities, self).__init__(**kwargs)
        self.m = -0.02

    def sparsity(self, similarities):
        entropy = tf.keras.backend.sum(
            similarities*tf.keras.backend.log(similarities),
            axis=-1
        )
        threshold = self.m * tf.keras.backend.ones_like(entropy)
        re_entropy = tf.keras.backend.maximum(
            entropy,
            threshold
        )
        return -tf.keras.backend.mean(re_entropy)

    def call(self, y_true, similarities):
        return self.sparsity(similarities)
