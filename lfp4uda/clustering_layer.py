from keras import backend as K
from keras.layers import InputSpec, Layer, Reshape
import numpy as np
import tensorflow as tf
from tensorflow.contrib.factorization import KMeansClustering


class ClusteringLayer(Layer):
    def __init__(self, kmeans: KMeansClustering, **kwargs):
        super(ClusteringLayer, self).__init__(**kwargs)

        self.kmeans = kmeans

    # def predict_vector(self, v: tf.Tensor):
    #     return self.kmeans.cluster_centers()[
    #         self.kmeans.predict_cluster_index(v)
    #     ]

    def call(self, x, **kwargs):

        input_shape = x.shape
        flatten_x = tf.reshape(
            x, (input_shape[0] * input_shape[1], input_shape[2])
        )

        flatten_y = tf.map_fn(lambda v: self.kmeans.predict(v), flatten_x)
        y = tf.reshape(flatten_y, input_shape)

        return [x, y]

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 3
        return [input_shape, input_shape]
