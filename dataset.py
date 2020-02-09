from PIL import Image
from torch.utils.data.dataset import Dataset  # For custom datasets
from torchvision.transforms import transforms
import numpy as np

class CustomDatasetFromArrays(Dataset):
  def __init__(self, img_arr, labels_arr, transform=False ):
    """
    Args:
        img_arr (np.float): contains image data
        labels_arr (np.long): contains image labels [0..9]
        transform: pytorch transforms for transforms and tensor conversion
    """
    # Transforms
    self.transform = transform
    self.images = img_arr
    self.labels = labels_arr
    self.img_data_len = len(self.images)
    self.labels_data_len = len(self.labels)
    assert self.labels_data_len == self.img_data_len

  def __getitem__(self, idx):
    img = Image.fromarray(self.images[idx])
    label = self.labels[idx]
    if self.transform:
      img = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))]
)(img)


    return img, label

  def __len__(self):
    return self.img_data_len