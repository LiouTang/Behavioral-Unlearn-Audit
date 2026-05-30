from .idx_dataset import IdxDataset
from .cifar import IdxCIFAR10, IdxCIFAR100

def get_dataset(name, root, *args, **kwargs) -> IdxDataset:
    return eval("Idx" + name)(root, *args, **kwargs)