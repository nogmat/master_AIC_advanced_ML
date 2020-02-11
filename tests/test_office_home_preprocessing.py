from lfp4uda.office_home_preprocessing import OfficeHomeFolders, OfficeHome
import numpy as np
import pytest
import tensorflow as tf


class TestOfficeHomePreprocessing:

    def testimport(self):

        datasets = OfficeHome().import_folder(folder=OfficeHomeFolders.Clipart)
        assert len(datasets) == 1
