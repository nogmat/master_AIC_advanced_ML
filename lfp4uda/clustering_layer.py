from sklearn.cluster import KMeans
from keras import backend as K
from keras.layers import InputSpec, Layer, Reshape
import numpy as np
import tensorflow as tf


class ClusteringLayer(Layer):
    def __init__(self, *, kmeans: KMeans, input_shape, **kwargs):
        super(ClusteringLayer, self).__init__(**kwargs)

        self.kmeans = kmeans
    
    def predict_vector(self, v: tf.Tensor):
        # Calling eval might slow down the whole layer
        return self.kmeans.cluster_centers_[self.kmeans.predict(v.eval())]

    def call(self, x, **kwargs):

        input_shape = x.shape
        flatten_x = tf.reshape(
            x, (input_shape[0] * input_shape[1], input_shape[2])
        )

        flatten_y = tf.map_fn(self.predict_vector, flatten_x)
        y = tf.reshape(flatten_y, input_shape)

        return [x, y]

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 3
        return [input_shape, input_shape]
