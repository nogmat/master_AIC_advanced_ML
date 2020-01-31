from keras.applications import VGG16
from keras.models import Sequential


class Feature_Extractor:
    def __init__(self, image_size):

        self.model = Sequential()

        vgg16 = VGG16(
            weights="imagenet",
            include_top=False,
            input_shape=(image_size, image_size, 3),
        )
        # vgg16.layers[-3] is block5_conv2
        # vgg16.layers[-2] is block5_conv3
        # vgg16.layers[-1] is block5_pool
        for layer in vgg16.layers[:-2]:
            layer.trainable = False
        self.model.add(vgg16)
