"""
Script to load data from ubyte files
"""
import idx2numpy
import numpy as np

from dataset import CustomDatasetFromArrays


class DataLoader:
    """
    Dataloader with random sampler
    to split the data into training and validation
    """

    def __init__(self, config=None):
        self.config = config
        self.train_images = idx2numpy.convert_from_file(self.config.train_images_path)
        self.valid_images = idx2numpy.convert_from_file(self.config.valid_images_path)
        self.train_labels = idx2numpy.convert_from_file(self.config.train_labels_path).astype(np.long)
        self.valid_labels = idx2numpy.convert_from_file(self.config.valid_labels_path).astype(np.long)

    def get_loaders(self, triplet=False):
        """
        :param shuffle_dataset(bool): data shuffle enabled / disabled
        :return: dataset, train_sampler, validation_sampler
        """
        train_dataset = CustomDatasetFromArrays(self.train_images, self.train_labels, transform=True, triplet=triplet,
                                                train=True, eval=False)
        valid_dataset = CustomDatasetFromArrays(self.valid_images, self.valid_labels, transform=True, triplet=triplet,
                                                train=False, eval=True)
        return train_dataset, valid_dataset
