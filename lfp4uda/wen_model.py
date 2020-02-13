from feature_extractor import FeatureExtractor
from netvlad import NetVLAD
from local_feature_alignment import LocalFeatureAlignment
from holistic_discriminator import HolisticDiscriminator
from local_discriminator import LocalDiscriminator
from single_layer_classifier import SingleLayerClassifier
from office_31_preprocessing import Office31
from loss_model_2 import LossSimilarities
import numpy as np
from sklearn.cluster import KMeans
import tensorflow as tf
from tensorflow.python.data.ops.dataset_ops import MapDataset

IMG_WIDTH = 256
IMG_HEIGHT = 256
BATCH_SIZE = 32


class WenModel:

    def __init__(self, source: MapDataset, target: MapDataset):

        # Initialize cluster centroids with k means
        image_batch, label_batch = next(iter(source))

        feature_extractor = FeatureExtractor()

        input_0 = tf.keras.layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
        model_0 = tf.keras.models.Model(
            inputs=[input_0], outputs=feature_extractor(input_0))
        features = model_0(image_batch)

        features_np = tf.keras.backend.reshape(
            features,
            shape=(np.prod(features.shape[:3]), features.shape[3])
        ).numpy()

        kmeans = KMeans(n_clusters=32)
        kmeans.fit(features_np)
        centroids = kmeans.cluster_centers_

        # Pass centroids to NetVLAD
        netvlad = NetVLAD(0.01, centroids, False)

        # Define layers after NetVLAD
        single_layer_classifier = SingleLayerClassifier(31)

        softmax = tf.keras.layers.Softmax()
        holistic_discriminator = HolisticDiscriminator(768, 1536)

        feature_alignment = LocalFeatureAlignment()
        local_discriminator = LocalDiscriminator(2048, 4096)

        # Step 1 : Classifier training
        image_input_1 = tf.keras.layers.Input(shape=(256, 256, 3))
        layer1_1 = feature_extractor(image_input_1)
        layer1_2, _, _ = netvlad(layer1_1)
        layer1_3 = single_layer_classifier(layer1_2)
        layer1_4 = softmax(layer1_3)
        model_step_1 = tf.keras.models.Model(
            inputs=[image_input_1], outputs=layer1_4)

        adam = tf.keras.optimizers.Adam(learning_rate=0.01)
        model_step_1.compile(loss='categorical_crossentropy', optimizer=adam)
        model_step_1.fit(image_batch, label_batch)

        print("tvb step 1")

        # Step 2 : Source fine tuning
        feature_extractor.set_trainable(True)
        netvlad = NetVLAD(0.01, centroids, True)

        layer2_1 = feature_extractor(image_input_1)
        vlad, similarities, _ = netvlad(layer2_1)
        layer2_3 = single_layer_classifier(vlad)
        layer2_4 = softmax(layer2_3)
        model_step_2 = tf.keras.models.Model(
            inputs=[image_input_1], outputs=[layer2_4, similarities])

        adam = tf.keras.optimizers.Adam(learning_rate=0.0001)
        model_step_2.compile(
            loss={
                'softmax': tf.keras.losses.categorical_crossentropy,
                'net_vlad_1': LossSimilarities()},
            loss_weights={
                'softmax': 1,
                'net_vlad_1': 1
            },
            target_tensors={
                'softmax': layer2_4,
                'net_vlad_1': similarities},
            optimizer=adam)
        print("tvb")
        # Step 3 : Domain adaptation


if __name__ == "__main__":

    office_31 = Office31().import_folder()

    WenModel(office_31[0], office_31[1])
