"""
Script to load data from ubyte files
"""
import idx2numpy
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from dataset import CustomDatasetFromArrays

class DataLoader:
    """
    Dataloader with random sampler
    to split the data into training and validation
    """
    def __init__(self, config=None):
        self.config = config
        self.images = idx2numpy.convert_from_file(self.config.images_path)
        self.labels = idx2numpy.convert_from_file(self.config.labels_path).astype(np.long)
        self.dataset_size = len(self.images)
        assert self.dataset_size == len(self.labels)

    def get_loaders(self, shuffle_dataset=False):
        """
        :param shuffle_dataset(bool): data shuffle enabled / disabled
        :return: dataset, train_sampler, validation_sampler
        """
        dataset = CustomDatasetFromArrays(self.images, self.labels, True)
        indices = list(range(self.dataset_size))
        split = int(np.floor(self.config.validation_split * self.dataset_size))
        if shuffle_dataset:
            np.random.seed(self.config.random_seed)
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        return dataset, train_sampler, valid_sampler
