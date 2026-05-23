import numpy as np
from sklearn.covariance import LedoitWolf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .misc import torch_cat


@torch.inference_mode()
def get_logit(model: nn.Module, query_loader: DataLoader, device):

    logits = None
    for i, (input, target) in enumerate(query_loader):
        input, target = input.to(device), target.to(device)
        output = model(input) # (batch_size, num_classes)
        logits = torch_cat(logits, output.gather(1, target.view(-1, 1)).squeeze(1), dim=0)

    return logits.cpu().numpy()

def fit_gaussian(X, cov_mode="ledoit", eps=1e-6):
    """
    Fit a transcript-level Gaussian N(mu, Sigma).

    cov_mode:
      "diag"   : product of per-query Gaussian marginals
      "ledoit" : full covariance with Ledoit-Wolf shrinkage
    """
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(axis=0)

    if cov_mode == "diag":
        var = X.var(axis=0) + eps
        return {"mode": "diag", "mu": mu, "var": var}

    lw = LedoitWolf(store_precision=True).fit(X)
    cov = lw.covariance_ + eps * np.eye(X.shape[1])
    sign, logdet = np.linalg.slogdet(cov)

    return {
        "mode": "full",
        "mu": lw.location_,
        "cov": cov,
        "prec": np.linalg.inv(cov),
        "logdet": logdet,
    }

def logpdf_gaussian(X, G):
    X = np.asarray(X, dtype=np.float64)
    diff = X - G["mu"]
    d = X.shape[1]

    if G["mode"] == "diag":
        var = G["var"]
        return -0.5 * (np.sum(diff * diff / var, axis=1) + np.sum(np.log(var)) + d * np.log(2.0 * np.pi))

    q = np.einsum("ij,jk,ik->i", diff, G["prec"], diff)
    return -0.5 * (q + G["logdet"] + d * np.log(2.0 * np.pi))

def sample_gaussian(G, n, rng: np.random.Generator):
    if G["mode"] == "diag":
        return rng.normal(G["mu"], np.sqrt(G["var"]), size=(n, len(G["mu"])))
    return rng.multivariate_normal(G["mu"], G["cov"], size=n)

def lrt_overlap_errors(G_pos, G_neg, n_mc=200_000, seed=None):
    """
    Positive class is detected when:
        log p_pos(tau) - log p_neg(tau) >= threshold.

    With equal priors/equal costs, threshold=0 is Bayes-optimal.
    """
    rng = np.random.default_rng(seed)

    X_pos = sample_gaussian(G_pos, n_mc, rng)
    X_neg = sample_gaussian(G_neg, n_mc, rng)

    S_pos = logpdf_gaussian(X_pos, G_pos) - logpdf_gaussian(X_pos, G_neg)
    S_neg = logpdf_gaussian(X_neg, G_pos) - logpdf_gaussian(X_neg, G_neg)

    tpr = np.mean(S_pos >= 0.0)
    fpr = np.mean(S_neg >= 0.0)
    fnr = 1.0 - tpr

    return {
        "fnr": float(fnr),
        "fpr": float(fpr),
        "tpr": float(tpr),
        "J": float(tpr - fpr),   # total-variation estimate under fitted model
    }

def LiR_test(pos, neg, cov_mode="ledoit", n_mc=200_000):
    # pos = get_logit(pos_list, query_loader, device)
    # neg = get_logit(neg_list, query_loader, device)
    G_pos = fit_gaussian(pos, cov_mode=cov_mode)
    G_neg = fit_gaussian(neg, cov_mode=cov_mode)
    return lrt_overlap_errors(G_pos=G_pos, G_neg=G_neg, n_mc=n_mc)