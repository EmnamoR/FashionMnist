from PIL import Image
from torch.utils.data.dataset import Dataset  # For custom datasets
from torchvision.transforms import transforms
import numpy as np


class CustomDatasetFromArrays(Dataset):
    """
    Fashion Mnist Custom Dataset
    If triplet :
    For each sample (anchor) randomly chooses a positive and negative samples
    Creates fixed triplets for testing
    """

    def __init__(self, img_arr, labels_arr, transform=False, triplet=False, train=False, eval=False):
        """
        Args:
            img_arr (np.float): contains image data
            labels_arr (np.long): contains image labels [0..9]
            transform: pytorch transforms for transforms and tensor conversion
        """
        self.triplet = triplet
        self.eval = eval
        self.train = train
        self.transform = transform
        self.images = img_arr
        self.labels = labels_arr
        self.img_data_len = len(self.images)
        self.labels_data_len = len(self.labels)
        self.labels_set = set(self.labels)
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                 for label in self.labels_set}
        if self.triplet:
            if self.eval:
                random_state = np.random.RandomState(29)

                triplets = [[i,
                             random_state.choice(self.label_to_indices[self.labels[i].item()]),
                             random_state.choice(self.label_to_indices[
                                                     np.random.choice(
                                                         list(self.labels_set - set([self.labels[i].item()]))
                                                     )
                                                 ])
                             ]
                            for i in range(len(self.images))]
                self.test_triplets = triplets
    def __getitem__(self, idx):
        img = Image.fromarray(self.images[idx])
        label = self.labels[idx]
        if self.transform:
            img = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize((0.5,), (0.5,))])(img)
        if self.triplet:
            if self.train:
                img1, label1 = self.images[idx], self.labels[idx].item()
                positive_idx = idx
                while positive_idx == idx:
                    positive_idx = np.random.choice(self.label_to_indices[label1])
                negative_label = np.random.choice(list(self.labels_set - set([label1])))
                negative_idx = np.random.choice(self.label_to_indices[negative_label])
                img2 = self.images[positive_idx]
                img3 = self.images[negative_idx]
            else:
                img1 = self.images[self.test_triplets[idx][0]]
                img2 = self.images[self.test_triplets[idx][1]]
                img3 = self.images[self.test_triplets[idx][2]]
            if self.transform is not None:
                img1 = transforms.Compose([transforms.ToTensor(),
                                           transforms.Normalize((0.5,), (0.5,))])(img1)
                img2 = transforms.Compose([transforms.ToTensor(),
                                           transforms.Normalize((0.5,), (0.5,))])(img2)
                img3 = transforms.Compose([transforms.ToTensor(),
                                           transforms.Normalize((0.5,), (0.5,))])(img3)
            return (img1, img2, img3), []
        else:
            return img, label

    def __len__(self):
        return self.img_data_len
