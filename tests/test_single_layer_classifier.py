from lfp4uda.single_layer_classifier import SingleLayerClassifier
import numpy as np
import pytest
import tensorflow as tf


class TestSingleLayerClassifier:

    def testcall(self):

        aggregated_residuals = np.array(
            [[0.6064591,  0.7948939, -0.01324965, -0.01324965]])

        input_1 = tf.keras.layers.Input(shape=(4,))
        classifier = SingleLayerClassifier(2)(input_1)
        classifie = tf.keras.layers.Softmax()(classifier)
        model = tf.keras.models.Model(inputs=[input_1], outputs=classifie)

        m = model([aggregated_residuals])
        assert (1, 2) == m.shape
