import os
import copy

import numpy as np
from sklearn.covariance import LedoitWolf
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.func import stack_module_state, functional_call

class Ensemble(nn.Module):
    """
    Vectorized forward over n models sharing one architecture but with distinct weights.

    Formally, given f_i : X -> Y for i = 1, ..., n with identical structure,
        Ensemble(models)(x) = [f_1(x), ..., f_n(x)]
    stacked along a new leading axis, so the output has shape (n, *f_i(x).shape).

    Eval-only: weights are detached and stored as buffers; no gradients flow.
    """

    def __init__(self, models: nn.ModuleList):
        super().__init__()
        for m in models:
            m.eval()

        params, buffers = stack_module_state(models)

        self.n_models = len(models)
        self._param_keys = list(params.keys())
        self._buffer_keys = list(buffers.keys())

        # register_buffer forbids '.' in names; encode it.
        for k, v in params.items():
            self.register_buffer(self._enc("p", k), v.detach().clone())
        for k, v in buffers.items():
            self.register_buffer(self._enc("b", k), v.detach().clone())

        # Architectural skeleton with no real storage — used only for its structure.
        # Stored via object.__setattr__ to keep it out of the nn.Module registry,
        # so .to(device) / _apply never tries to move it off the meta device.
        base = copy.deepcopy(models[0]).to("meta")
        base.eval()
        object.__setattr__(self, "base", base)

    @staticmethod
    def _enc(prefix: str, key: str) -> str:
        return f"_{prefix}_{key.replace('.', '__DOT__')}"

    def _collect(self):
        params = {k: getattr(self, self._enc("p", k)) for k in self._param_keys}
        buffers = {k: getattr(self, self._enc("b", k)) for k in self._buffer_keys}
        return params, buffers

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        params, buffers = self._collect()
        def one_model(p, b, x_):
            return functional_call(self.base, (p, b), (x_,))
        return torch.vmap(one_model, in_dims=(0, 0, None))(params, buffers, x)
