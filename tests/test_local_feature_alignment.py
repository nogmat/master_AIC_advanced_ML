from lfp4uda.local_feature_alignment import LocalFeatureAlignment
import numpy as np
import pytest
import tensorflow as tf


class TestLocalFeatureAlignment:
    def testcall(self):

        # Simulation of F[i][j][d] - c[k][d]
        distance = np.array(
            [
                [
                    [[[1.0, -1.0], [2.0, -2.0]], [[3.0, -3.0], [4.0, -4.0]]],
                    [[[5.0, -5.0], [6.0, -6.0]], [[7.0, -7.0], [8.0, -8.0]]],
                ]
            ]
        )
        # Simulation of S[i][j][k]
        similarities = np.array([[[[10, 1], [1, 10]], [[20, 2], [27, 3]]]])

        input1 = tf.keras.layers.Input(shape=(2, 2, 2, 2))
        input2 = tf.keras.layers.Input(shape=(2, 2, 2))
        layer = LocalFeatureAlignment()([input1, input2])
        model = tf.keras.models.Model(inputs=[input1, input2], outputs=layer)
        
        # Must return [F[i][j][d]-c[a[i][j]][d],a[i][j]]
        # where a is the best similarities array
        m = model([distance, similarities], training=False)
        assert (1, 4, 3) == m.shape
