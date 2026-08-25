import math
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
        "J": float(tpr - fpr),  # total-variation estimate under fitted model
    }

def LiR_test(pos, neg, cov_mode="ledoit", n_mc=200_000):
    # pos = get_logit(pos_list, query_loader, device)
    # neg = get_logit(neg_list, query_loader, device)
    G_pos = fit_gaussian(pos, cov_mode=cov_mode)
    G_neg = fit_gaussian(neg, cov_mode=cov_mode)
    return lrt_overlap_errors(G_pos=G_pos, G_neg=G_neg, n_mc=n_mc)


def _std_normal_cdf(x):
    """Standard normal CDF without an additional scipy dependency."""
    return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))

def _projection_geometry(p, c, eps=1e-12):
    """
    Euclidean projection of p onto span(c).

    Returns gamma = |<p,c>| / ||c||^2 and the parallel/perpendicular
    component norms. Sign is intentionally discarded in gamma, matching
    the paper's alignment coefficient.
    """
    c = np.asarray(c, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)

    c2 = float(c @ c)
    if c2 <= eps:
        return {
            "gamma": np.nan,
            "p_parallel_norm": np.nan,
            "p_perp_norm": np.nan,
        }

    coeff = float((p @ c) / c2)
    p_parallel = coeff * c
    p_perp = p - p_parallel

    return {
        "gamma": abs(coeff),
        "p_parallel_norm": float(np.linalg.norm(p_parallel)),
        "p_perp_norm": float(np.linalg.norm(p_perp)),
    }


def theory_reference_geometry(hon_list, dis_list, member_mask, query_idx, eps=1e-6):
    """
    Compute the non-convex behavioral geometry and a shared-diagonal-
    covariance Gaussian reference for one query sequence Q.

    This function DOES NOT add observation noise. It uses:
      1. the mean behavioral shifts exposed by the saved surrogate models;
      2. within-cell surrogate variability to define a common diagonal
         covariance reference model.

    Parameters
    ----------
    hon_list : array, shape [N, num_available_queries]
        Honest-unlearning target-class logits for each surrogate model.

    dis_list : array, shape [N, num_available_queries]
        Dishonest-unlearning target-class logits for each surrogate model.

    member_mask : bool array, shape [N]
        True when z* belongs to the retained set for that surrogate.

    query_idx : integer array-like
        Indices of the T behavioral queries in Q.

    eps : float
        Numerical floor for variance/projection denominators.

    Returns
    -------
    dict
        Raw behavioral geometry:
          c_norm, p_norm, gamma, p_parallel_norm, p_perp_norm

        Diagnostics:
          c_balanced_norm, gamma_balanced,
          interaction_norm, interaction_ratio

        Shared-covariance reference geometry:
          d_comp_ref, d_cur_ref, gamma_ref,
          p_parallel_ref_norm, p_perp_ref_norm

        Reference distinguishability:
          e_ref, beta_ref, beta_lb_ref, beta_lb_simple_ref

    Notes
    -----
    The operational audit fits separate diagonal covariances to the two
    hypotheses. The reference quantities below deliberately impose one
    pooled diagonal covariance, so beta_ref/beta_lb_ref are clean
    Gaussian-geometric reference values rather than aliases for beta_op.
    """
    H = np.asarray(hon_list, dtype=np.float64)[:, query_idx]
    D = np.asarray(dis_list, dtype=np.float64)[:, query_idx]

    if H.ndim == 1:
        H = H[:, None]
    if D.ndim == 1:
        D = D[:, None]

    member_mask = np.asarray(member_mask, dtype=bool)
    if member_mask.ndim != 1 or len(member_mask) != len(H):
        raise ValueError(
            "member_mask must be a 1-D boolean array with one entry "
            "per surrogate model."
        )

    nonmember_mask = ~member_mask

    HM = H[member_mask]
    HN = H[nonmember_mask]
    DM = D[member_mask]
    DN = D[nonmember_mask]

    n_mem = len(HM)
    n_non = len(HN)

    if min(n_mem, n_non) < 2:
        raise ValueError(
            "Need at least two member and two nonmember surrogate models "
            f"for the reference geometry; got n_mem={n_mem}, "
            f"n_non={n_non}."
        )

    # ------------------------------------------------------------------
    # Four cell centroids: honesty x membership.
    # ------------------------------------------------------------------
    mu_hm = HM.mean(axis=0)
    mu_hn = HN.mean(axis=0)
    mu_dm = DM.mean(axis=0)
    mu_dn = DN.mean(axis=0)

    # Conditional compliance directions.
    c_mem = mu_hm - mu_dm
    c_non = mu_hn - mu_dn

    # Conditional membership directions.
    p_hon = mu_hm - mu_hn
    p_dis = mu_dm - mu_dn

    # ------------------------------------------------------------------
    # Operational mean shifts.
    #
    # c_op is exactly the mean shift in the honest-vs-dishonest
    # populations passed to LiR_test.
    #
    # p_op equals the member-vs-nonmember mean shift because the original
    # code contributes one honest and one dishonest endpoint per surrogate
    # to the corresponding membership population.
    # ------------------------------------------------------------------
    c_op = H.mean(axis=0) - D.mean(axis=0)
    p_op = 0.5 * (p_hon + p_dis)

    c_norm = float(np.linalg.norm(c_op))
    p_norm = float(np.linalg.norm(p_op))

    raw_geom = _projection_geometry(p_op, c_op, eps=eps)

    # ------------------------------------------------------------------
    # Balanced 2 x 2 main-effect diagnostic.
    #
    # c_balanced removes any effect of unequal member/nonmember counts.
    # The interaction is zero when the compliance displacement is the
    # same under membership and nonmembership.
    # ------------------------------------------------------------------
    c_balanced = 0.5 * (c_mem + c_non)
    interaction = 0.5 * (c_mem - c_non)

    c_balanced_norm = float(np.linalg.norm(c_balanced))
    interaction_norm = float(np.linalg.norm(interaction))
    interaction_ratio = interaction_norm / (
        c_balanced_norm + p_norm + eps
    )

    balanced_geom = _projection_geometry(
        p_op,
        c_balanced,
        eps=eps,
    )

    # ------------------------------------------------------------------
    # Shared diagonal residual covariance.
    #
    # Remove the four cell means first, then pool the residual variance.
    # This captures surrogate/training variability without injecting
    # artificial observation noise.
    # ------------------------------------------------------------------
    residual_ss = np.zeros(H.shape[1], dtype=np.float64)
    residual_dof = 0

    for X in (HM, HN, DM, DN):
        mu = X.mean(axis=0)
        residual_ss += np.square(X - mu).sum(axis=0)
        residual_dof += X.shape[0] - 1

    if residual_dof <= 0:
        raise ValueError("Cannot estimate the pooled reference covariance.")

    pooled_var = residual_ss / residual_dof
    pooled_var = np.maximum(pooled_var, eps)
    pooled_std = np.sqrt(pooled_var)

    # Mahalanobis/whitened geometry under the common covariance.
    c_w = c_op / pooled_std
    p_w = p_op / pooled_std

    d_comp_ref = float(np.linalg.norm(c_w))
    d_cur_ref = float(np.linalg.norm(p_w))

    ref_geom = _projection_geometry(p_w, c_w, eps=eps)

    # ------------------------------------------------------------------
    # Equal-common-covariance Gaussian reference:
    #
    #   TV(N(mu0,Sigma), N(mu1,Sigma))
    #       = 2 Phi(||Sigma^{-1/2}(mu1-mu0)|| / 2) - 1.
    #
    # e_ref = alpha + alpha' = 1 - TV for the Bayes test.
    # ------------------------------------------------------------------
    comp_tv_ref = 2.0 * _std_normal_cdf(d_comp_ref / 2.0) - 1.0
    e_ref = 1.0 - comp_tv_ref

    beta_ref = 2.0 * _std_normal_cdf(d_cur_ref / 2.0) - 1.0

    gamma_ref = ref_geom["gamma"]
    if np.isfinite(gamma_ref):
        # The compliance-aligned component has whitened length
        # gamma_ref * ||c_w||.
        beta_lb_ref = (
            2.0
            * _std_normal_cdf(gamma_ref * d_comp_ref / 2.0)
            - 1.0
        )

        # Simpler linear TV lower bound corresponding to
        # min(gamma, 1) * (1 - e) in the shared-covariance reference.
        beta_lb_simple_ref = (
            min(gamma_ref, 1.0) * (1.0 - e_ref)
        )
    else:
        beta_lb_ref = np.nan
        beta_lb_simple_ref = np.nan

    return {
        "n_mem": int(n_mem),
        "n_non": int(n_non),

        # Raw behavioral geometry, directly in target-logit space.
        "c_norm": c_norm,
        "p_norm": p_norm,
        "gamma": raw_geom["gamma"],
        "p_parallel_norm": raw_geom["p_parallel_norm"],
        "p_perp_norm": raw_geom["p_perp_norm"],

        # 2 x 2 balancing/interaction diagnostics.
        "c_balanced_norm": c_balanced_norm,
        "gamma_balanced": balanced_geom["gamma"],
        "interaction_norm": interaction_norm,
        "interaction_ratio": interaction_ratio,

        # Shared-covariance whitened geometry.
        "d_comp_ref": d_comp_ref,
        "d_cur_ref": d_cur_ref,
        "gamma_ref": gamma_ref,
        "p_parallel_ref_norm": ref_geom["p_parallel_norm"],
        "p_perp_ref_norm": ref_geom["p_perp_norm"],

        # Gaussian reference quantities.
        "e_ref": e_ref,
        "beta_ref": beta_ref,
        "beta_lb_ref": beta_lb_ref,
        "beta_lb_simple_ref": beta_lb_simple_ref,
    }
