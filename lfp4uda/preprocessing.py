from enum import Enum
import numpy as np
import os
import pathlib
import re
import tensorflow as tf

IMG_WIDTH = 256
IMG_HEIGHT = 256
BATCH_SIZE = 32


class Preprocessing:
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

    def import_folder(self, *, main_folder_path: str, folder: str = None):

        if folder is None:

            folders = np.array(
                [
                    pathlib.Path(f"{main_folder_path}/{member}")
                    for member in os.listdir(main_folder_path)
                    if os.path.isdir(f"{main_folder_path}/{member}")
                ]
            )

        else:
            folders = np.array([pathlib.Path(f"{main_folder_path}/{folder}")])

        all_class_names = np.array(
            [[item.name for item in f.glob("*")]
             for f in folders]
        )
        print(folders)
        print(all_class_names)

        datasets = [
            self.prepare_for_training(
                tf.data.Dataset.list_files(f"{folders[i]}/*/*").map(
                    lambda path: self.process_path(path, all_class_names[i])
                )
            )
            for i in range(len(folders))
        ]

        return datasets


class OfficeHome(Preprocessing):
    def import_folder(self, *, folder: str = None):
        return super().import_folder(
            main_folder_path=f"{os.path.dirname(os.path.realpath(__file__))}/../office-home"
        )


class Office31(Preprocessing):
    def import_folder(self, *, folder: str = None):
        main_folder_path = f"{os.path.dirname(os.path.realpath(__file__))}/../office-31"
        if folder is None:

            folders = np.array(
                [
                    pathlib.Path(f"{main_folder_path}/{member}/images")
                    for member in os.listdir(main_folder_path)
                    if os.path.isdir(f"{main_folder_path}/{member}")
                ]
            )

        else:
            folders = np.array([pathlib.Path(f"{main_folder_path}/{folder}/images")])

        all_class_names = np.array(
            [[item.name for item in f.glob("*")]
             for f in folders]
        )
        print(folders)
        print(all_class_names)

        datasets = [
            self.prepare_for_training(
                tf.data.Dataset.list_files(f"{folders[i]}/*/*").map(
                    lambda path: self.process_path(path, all_class_names[i])
                )
            )
            for i in range(len(folders))
        ]

        return datasets
