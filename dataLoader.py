import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler

from dataset import CustomDatasetFromArrays

torch.set_grad_enabled(True)  # On by default, leave it here for clarity
import idx2numpy


class dataLoader(object):
  def __init__(self, config=None, normalize=False):
    self.config = config
    self.images = idx2numpy.convert_from_file(self.config.images_path)
    self.labels = idx2numpy.convert_from_file(self.config.labels_path).astype(
      np.long)
    self.dataset_size = len(self.images)
    assert self.dataset_size == len(self.labels)

  def getLoaders(self, shuffle_dataset=False):
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
