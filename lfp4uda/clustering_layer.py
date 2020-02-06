from sklearn.cluster import KMeans
from keras import backend as K
from keras.layers import InputSpec, Layer, Reshape
import numpy as np
import tensorflow as tf


class ClusteringLayer(Layer):
    def __init__(self, kmeans: KMeans, *, input_shape, **kwargs):
        super(ClusteringLayer, self).__init__(**kwargs)

        self.kmeans = kmeans
        self.input_spec = InputSpec(dtype=K.floatx(), shape=input_shape)

        self.reshape_in = Reshape(
            (input_shape[0] * input_shape[1], input_shape[2]),
            input_shape=input_shape,
        )
        self.reshape_out = Reshape(
            input_shape,
            input_shape=(input_shape[0] * input_shape[1], input_shape[2]),
        )

    def call(self, x, **kwargs):
        flatten_x = self.reshape_in(x)
        flatten_y = tf.convert_to_tensor(
            np.fromiter(
                self.kmeans.cluster_centers_[i]
                for i in self.kmeans.predict(flatten_x)
            ),
            np.float32,
        )
        y = self.reshape_out(flatten_y)
        return [x, y]

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 3
        return [input_shape, input_shape]
