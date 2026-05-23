import numpy as np
import torch


# set functionalities for np arrays
def set_inter(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.array(list(set(a) & set(b)))
def set_union(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.array(list(set(a) | set(b)))
def set_minus(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.array(list(set(a) - set(b)))

def torch_cat(a, b, dim) -> torch.Tensor:
    if a is None:
        return b
    else:
        return torch.cat([a, b], dim=dim)

