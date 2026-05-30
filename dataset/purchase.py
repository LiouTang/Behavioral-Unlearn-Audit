import os
import urllib
import tarfile

import numpy as np
import torch
from torch.utils.data import Dataset

from .idx_dataset import IdxDataset


class Purchase100(Dataset):
    def __init__(self, root):
        if not os.path.exists(os.path.join(root, f"purchase100.npz")):
            url = "https://github.com/xehartnort/Purchase100-Texas100-datasets/blob/master/purchase100.npz?raw=true"
            urllib.request.urlretrieve(url, os.path.join(root, f"purchase100.npz"))
        file = np.load(os.path.join(root, f"purchase100.npz"))
        self.data, self.targets = np.array(file["features"], dtype=np.float32), np.array(file["labels"], dtype=np.int64)
        self.targets = np.where(self.targets)[1]
        assert np.all((self.data == 0) | (self.data == 1))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]), int(self.targets[idx])


class IdxPurchase100(IdxDataset):
    def __init__(self, root, seed: int=None):
        super().__init__(root=root, seed=seed)
        self.full_set    = Purchase100(root=self.root)
        self.valid_set   = self.full_set
        self.num_classes = np.unique(self.full_set.targets).shape[0]
