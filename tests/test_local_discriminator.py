from lfp4uda.local_discriminator import LocalDiscriminator
import numpy as np
import pytest
import tensorflow as tf


class TestLocalDiscriminator:

    def testcall(self):

        input_ = tf.keras.layers.Input(shape=(2, 2, 3))
        dense = LocalDiscriminator(2048, 4096)(input_)
        model = tf.keras.models.Model(inputs=[input_], outputs=dense)

        aligned_residuals = np.array(
            [
                [
                    [1.0, -1.0, 0.0],
                    [4.0, -4.0, 1.0],
                    [5.0, -5.0, 0.0],
                    [7.0, -7.0, 0.0],
                ]
            ]
        )

        m = model([aligned_residuals])
        assert (1, 4, 1) == m.shape
        assert np.all(np.logical_and(m >= -1, m <= 1))
