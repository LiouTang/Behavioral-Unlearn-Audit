import os
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class UnlearnMethod(ABC):
    def __init__(self, model, loaders: dict[str, DataLoader], loss_fn, eta: float, device: str) -> None:
        assert "un" in loaders and "rt" in loaders
        self.model = model
        self.loaders = loaders
        self.loss_fn = loss_fn
        self.eta = eta
        self.device = device

    @abstractmethod
    def get_unlearned(self) -> nn.Module:
        raise NotImplementedError