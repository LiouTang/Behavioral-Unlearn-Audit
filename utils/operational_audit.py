import numpy as np
import torch

from .likelihood_ratio_test import fit_gaussian, logpdf_gaussian


@torch.inference_mode()
def get_probabilities(model, query_loader, device):
    model.eval()
    probabilities = []
    for inputs, _ in query_loader:
        probabilities.append(model(inputs.to(device)).softmax(dim=1).cpu())
    return torch.cat(probabilities, dim=0).numpy()


def split_model_runs(member_mask, rng):
    """Keep both update endpoints of each run in the same partition."""
    member_mask = np.asarray(member_mask, dtype=bool)
    partitions = {name: [] for name in ("fit", "calibration", "evaluation")}
    for is_member in (False, True):
        indices = rng.permutation(np.flatnonzero(member_mask == is_member))
        n = len(indices)
        groups = np.split(indices, [n // 2, 3 * n // 4])
        for name, group in zip(partitions, groups):
            partitions[name].append(group)
    return {name: np.concatenate(groups) for name, groups in partitions.items()}


def fit_lrt(pos, neg):
    return {
        "G_pos": fit_gaussian(pos, cov_mode="diag"),
        "G_neg": fit_gaussian(neg, cov_mode="diag"),
    }


def lrt_scores(test, X):
    return logpdf_gaussian(X, test["G_pos"]) - logpdf_gaussian(X, test["G_neg"])


def calibrate_threshold(pos_scores, neg_scores):
    """Minimize calibration balanced error; predict positive at score >= t."""
    pos_scores = np.sort(pos_scores)
    neg_scores = np.sort(neg_scores)
    thresholds = np.concatenate(([-np.inf], np.unique(np.concatenate(
        (pos_scores, neg_scores)
    )), [np.inf]))
    fnr = np.searchsorted(pos_scores, thresholds, side="left") / len(pos_scores)
    fpr = 1.0 - np.searchsorted(neg_scores, thresholds, side="left") / len(neg_scores)
    balanced_error = 0.5 * (fnr + fpr)
    best = thresholds[balanced_error == balanced_error.min()]
    return float(best[np.argmin(np.abs(best))])


def empirical_rates(pos_scores, neg_scores, threshold):
    tpr = float(np.mean(np.asarray(pos_scores) >= threshold))
    fpr = float(np.mean(np.asarray(neg_scores) >= threshold))
    fnr = 1.0 - tpr
    return {
        "fnr": fnr,
        "fpr": fpr,
        "tpr": tpr,
        "J": tpr - fpr,
        "balanced_accuracy": 0.5 * (tpr + 1.0 - fpr),
    }
