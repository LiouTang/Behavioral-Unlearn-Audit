import os
import copy

import numpy as np
import torch
from torch.utils.data import Dataset

import utils

class IdxDataset(Dataset):
    def __init__(self, root, seed: int=None):
        self.root = root
        self.rng = np.random.default_rng(seed)
        self.full_set, self.valid_set = None, None
        self.num_classes = None

    def __len__(self):
        return len(self.full_set)

    def get_subset(self, idx: np.ndarray) -> Dataset:
        new_set = copy.deepcopy(self.full_set)
        new_set.data    = np.array(self.full_set.data)[idx]
        new_set.targets = np.array(self.full_set.targets)[idx]
        return new_set

    def train_unlearn_split(self, N: int, un_size: int):
        class_0_idx = np.where(np.array(self.full_set.targets) == 0)[0]
        self.un_idx = self.rng.choice(class_0_idx, un_size, replace=False) # sample unlearned set from one class

        available = utils.set_minus(np.arange(len(self.full_set)), self.un_idx)

        train_idx_across_N = []
        for i in range(N):
            idx = self.rng.choice(available, int(len(available) / 2), replace=False)
            train_idx_across_N.append(idx)

        self.idx_across_N = np.array(train_idx_across_N)
        return self.idx_across_N, self.un_idx

