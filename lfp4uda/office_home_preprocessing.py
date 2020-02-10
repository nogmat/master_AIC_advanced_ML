from enum import Enum
import numpy as np
import os
import pathlib
import re
import tensorflow as tf

IMG_WIDTH = 256
IMG_HEIGHT = 256


class OfficeHomeFolders(Enum):
    All = "All"
    Art = "Art"
    Clipart = "Clipart"
    Product = "Product"
    RealWorld = "Real World"


class OfficeHome:
    # cf https://www.tensorflow.org/tutorials/load_data/images
    def get_label(self, file_path, class_names):
        # convert the path to a list of path components
        parts = tf.strings.split(file_path, os.path.sep)
        # The second to last is the class-directory
        return parts[-2] == class_names

    def decode_img(self, img):
        # convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_jpeg(img, channels=3)
        # Use `convert_image_dtype` to convert to floats in the [0,1] range.
        img = tf.image.convert_image_dtype(img, tf.float32)
        # resize the image to the desired size.
        return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])

    def process_path(self, file_path, class_names):
        label = self.get_label(file_path, class_names)
        # load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = self.decode_img(img)
        return img, label

    def import_folder(self, folder=OfficeHomeFolders.All):

        if folder == OfficeHomeFolders.All:

            folders = np.array(
                [
                    pathlib.Path(
                        f"{os.path.dirname(os.path.realpath(__file__))}/../office-home/{member.value}"
                    )
                    for member in OfficeHomeFolders.__members__.values()
                ]
            )

            all_class_names = np.array(
                [[item.name for item in f.glob("*")]
                 for f in folders]
            )

        else:
            folders = np.array([folder.value])
            all_class_names = np.array([[
                item.name
                for item in pathlib.Path(
                    f"{os.path.dirname(os.path.realpath(__file__))}/../office-home/{folder.value}"
                ).glob("*")
            ]])

        datasets = [
            tf.data.Dataset.list_files(f"{os.path.dirname(os.path.realpath(__file__))}/../office-home/{folders[i]}/*/*").map(
                lambda path: self.process_path(path, all_class_names[i])
            )
            for i in range(len(folders))
        ]

        return datasets
