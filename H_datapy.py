import numpy as np

import torch
from torch.utils.data.dataset import Dataset


class HData(Dataset):
    def __init__(self, dataset, tranform=None):
        self.data = dataset[0]
        self.trans = tranform
        self.labels = dataset[1]

    def __getitem__(self, index):
        img = torch.from_numpy(np.asarray(self.data[index, :, :, :]))
        label = torch.from_numpy(np.asarray(self.labels[index, :, :]))
        return img, label

    def __len__(self):
        return len(self.labels)

    def __labels__(self):
        return self.labels
