import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision import transforms

from .idx_dataset import IdxDataset

class IdxCIFAR10(IdxDataset):
    def __init__(self, root, img_size: int=32, seed: int=None):
        super().__init__(root, seed)
        t = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            transforms.Resize(img_size, antialias=True),
        ])
        self.full_set   = CIFAR10(root=self.root, download=True, train=True, transform=t)
        self.valid_set  = CIFAR10(root=self.root, download=True, train=False, transform=t)
        self.num_classes = np.unique(self.full_set.targets).shape[0]

class IdxCIFAR100(IdxDataset):
    def __init__(self, root, img_size: int=32, seed: int=None):
        super().__init__(root, seed)
        t = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
            transforms.Resize(img_size, antialias=True),
        ])
        self.full_set   = CIFAR100(root=self.root, download=True, train=True, transform=t)
        self.valid_set  = CIFAR100(root=self.root, download=True, train=False, transform=t)
        self.num_classes = np.unique(self.full_set.targets).shape[0]