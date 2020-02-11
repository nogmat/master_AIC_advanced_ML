from enum import Enum
import numpy as np
import os
import pathlib
import re
import tensorflow as tf

IMG_WIDTH = 256
IMG_HEIGHT = 256
BATCH_SIZE = 32


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

    def prepare_for_training(self, ds, cache=True, shuffle_buffer_size=1000):
        # This is a small dataset, only load it once, and keep it in memory.
        # use `.cache(filename)` to cache preprocessing work for datasets that
        # don't fit in memory.
        if cache:
            if isinstance(cache, str):
                ds = ds.cache(cache)
            else:
                ds = ds.cache()

        ds = ds.shuffle(buffer_size=shuffle_buffer_size)

        # Repeat forever
        ds = ds.repeat()

        ds = ds.batch(BATCH_SIZE)

        # `prefetch` lets the dataset fetch batches in the background while the
        # model is training.
        ds = ds.prefetch(buffer_size=100)

        return ds

    def import_folder(self, folder=OfficeHomeFolders.All):

        if folder == OfficeHomeFolders.All:

            folders = np.array(
                [
                    pathlib.Path(
                        f"{os.path.dirname(os.path.realpath(__file__))}/../office-home/{member.value}"
                    )
                    for member in OfficeHomeFolders.__members__.values() if member.value != "All"
                ]
            )

        else:
            folders = np.array(
                [
                    pathlib.Path(
                        f"{os.path.dirname(os.path.realpath(__file__))}/../office-home/{folder.value}"
                    )
                ]
            )

        all_class_names = np.array(
            [[item.name for item in f.glob("*")]
             for f in folders]
        )

        datasets = [
            self.prepare_for_training(
                tf.data.Dataset.list_files(f"{folders[i]}/*/*").map(
                    lambda path: self.process_path(path, all_class_names[i])
                )
            )
            for i in range(len(folders))
        ]

        return datasets
