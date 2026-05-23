import os
import numpy as np
import torch
from torch.utils.data import Dataset

from .idx_dataset import IdxDataset


class ToyData(Dataset):
    classes = [str(i) for i in range(4)]

    def __init__(self, root, n_class: int = 250, c: float = 1.6, seed: int = None):
        if os.path.exists(os.path.join(root, f"toydata_{n_class}_{seed}.npz")):
            data = np.load(os.path.join(root, f"toydata_{n_class}_{seed}.npz"))
            self.data    = data['data']
            self.targets = data['targets']
        else:
            rng = np.random.default_rng(seed)
            centers = [(+c, +c), (-c, +c), (-c, -c), (+c, -c)]
            pts, labels = [], []
            for k, (cx, cy) in enumerate(centers):
                Z = rng.normal(loc=[cx, cy], scale=0.9, size=(n_class, 2))
                pts.append(Z)
                labels.extend([k] * n_class)
            X, y = np.vstack(pts), np.array(labels)
            perm = rng.permutation(len(y))

            self.data    = X[perm].astype(np.float32)
            self.targets = y[perm]

            with open(os.path.join(root, f"toydata_{n_class}_{seed}.npz"), 'wb') as f:
                np.savez(f, data=self.data, targets=self.targets)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]), int(self.targets[idx])


class IdxToyData(IdxDataset):
    def __init__(self, root, n_class: int = 250, c: float = 1.6, seed: int = None):
        super().__init__(root=root, seed=seed)
        self.full_set    = ToyData(root=self.root, n_class=n_class, c=c, seed=seed)
        self.num_classes = len(self.full_set.classes)
