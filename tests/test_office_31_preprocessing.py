from lfp4uda.office_31_preprocessing import Office31Folders, Office31
import numpy as np
import pytest
import tensorflow as tf


class TestOffice31Preprocessing:

    def testimport(self):

        datasets = Office31().import_folder()
        assert len(datasets) == 3
