from .idx_dataset import IdxDataset
from .cifar10 import IdxCIFAR10

def get_dataset(name, root, *args, **kwargs) -> IdxDataset:
    return eval("Idx" + name)(root, *args, **kwargs)