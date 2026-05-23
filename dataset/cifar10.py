import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
from torchvision import transforms

from .idx_dataset import IdxDataset

class IdxCIFAR10(IdxDataset):
    def __init__(self, root, img_size: int=32, seed: int=None):
        super().__init__(root, seed)
        t = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.Resize(img_size, antialias=True),
        ])
        self.full_set = CIFAR10(root=self.root, download=True, train=True, transform=t)
        self.valid_set = CIFAR10(root=self.root, download=True, train=False, transform=t)
        self.num_classes = len(self.full_set.classes)
