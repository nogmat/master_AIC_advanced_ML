from feature_extractor import FeatureExtractor
from netvlad import NetVLAD
from local_feature_alignment import LocalFeatureAlignment
from holistic_discriminator import HolisticDiscriminator
from local_discriminator import LocalDiscriminator
from single_layer_classifier import SingleLayerClassifier
from office_31_preprocessing import Office31
import tensorflow as tf
from tensorflow.python.data.ops.dataset_ops import MapDataset


class WenModel:

    def __init__(self, source: MapDataset, target: MapDataset):

        # Init

        # Source list of tensors
        xs = source.map(lambda a, b: a)
        ys = source.map(lambda a, b: b)

        feature_extractor = FeatureExtractor()
        input_0 = tf.keras.layers.Input(shape=(256, 256, 3))
        model_0 = tf.keras.models.Model(
            inputs=[input_0], outputs=feature_extractor(input_0))

        kmeans = tf.compat.v1.estimator.experimental.KMeans(num_clusters=32)
        kmeans.train(model_0(xs))
        centroids = kmeans.cluster_centers()
        print("Tvb")
        # netvlad = NetVLAD(0.01, centroids)
        # single_layer_classifier = SingleLayerCLassifer()

        # softmax = tf.keras.layers.Softmax()
        # holistic_discriminator = HolisticDiscriminator(768, 1536)

        # feature_alignment = LocalFeatureAlignment()
        # local_discriminator = LocalDiscriminator(2048, 4096)


if __name__ == "__main__":

    office_31 = Office31().import_folder()

    WenModel(office_31[0], office_31[1])
