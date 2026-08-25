"""Utilities for convex-model behavioral auditing experiments.

This module provides synthetic data generation, regularized softmax fitting,
influence-based unlearning, membership endpoints, behavioral geometry,
Gaussian transcript metrics, and query selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import Divider, Size
from numpy.linalg import norm
from scipy.special import ndtr, ndtri

K = 4
LAM = 5e-3
LIM = 4.0


def single_ax_fig(margin=(1.0, 1.0), ax_size=(6.9, 6.9)):
    """Return (fig, ax) using a fixed-size Divider layout."""
    ml, mb = margin
    aw, ah = ax_size
    h = [Size.Fixed(ml), Size.Fixed(aw)]
    v = [Size.Fixed(mb), Size.Fixed(ah)]
    fig = plt.figure(figsize=(8, 8))
    div = Divider(fig, (0, 0, 1, 1), h, v, aspect=False)
    ax = fig.add_axes(div.get_position(), axes_locator=div.new_locator(nx=1, ny=1))
    return fig, ax


# -----------------------------------------------------------------------------
# Data and model
# -----------------------------------------------------------------------------

def make_data(n_per: int = 120, rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a shuffled four-class, two-dimensional Gaussian dataset."""
    if rng is None:
        rng = np.random.default_rng()
    centers = [(1.6, 1.6), (-1.6, 1.6), (-1.6, -1.6), (1.6, -1.6)]
    pts, labels = [], []
    for k, center in enumerate(centers):
        pts.append(rng.normal(loc=center, scale=0.9, size=(n_per, 2)))
        labels.extend([k] * n_per)
    X2 = np.vstack(pts)
    y = np.asarray(labels, dtype=int)
    perm = rng.permutation(len(y))
    return X2[perm], y[perm]


def feature_lift(X2: np.ndarray) -> np.ndarray:
    """Map each point to (x1, x2, x1*x2, |x1|, |x2|, 1)."""
    X2 = np.asarray(X2)
    x1, x2 = X2[:, 0], X2[:, 1]
    return np.stack([x1, x2, x1 * x2, np.abs(x1), np.abs(x2), np.ones_like(x1)], axis=1)


def softmax(Z: np.ndarray) -> np.ndarray:
    Z = Z - np.max(Z, axis=-1, keepdims=True)
    eZ = np.exp(Z)
    return eZ / np.sum(eZ, axis=-1, keepdims=True)


def predict_proba(theta_flat: np.ndarray, X: np.ndarray, n_classes: int = K) -> np.ndarray:
    D = X.shape[1]
    return softmax(X @ theta_flat.reshape(D, n_classes))


def response(theta_flat: np.ndarray, X: np.ndarray, response_class: int = 0) -> np.ndarray:
    """Scalar behavioral response f_theta(x): posterior of one fixed class."""
    return predict_proba(theta_flat, X)[:, response_class]


def _grad(theta_flat: np.ndarray, X: np.ndarray, y: np.ndarray, lam: float = LAM) -> np.ndarray:
    D = X.shape[1]
    theta = theta_flat.reshape(D, K)
    P = softmax(X @ theta)
    Y = np.zeros_like(P)
    Y[np.arange(len(y)), y] = 1.0
    return (X.T @ (P - Y) / len(y) + lam * theta).ravel()


def _hessian(theta_flat: np.ndarray, X: np.ndarray, lam: float = LAM) -> np.ndarray:
    D = X.shape[1]
    theta = theta_flat.reshape(D, K)
    P = softmax(X @ theta)
    H = np.zeros((D * K, D * K), dtype=float)
    for x, p in zip(X, P):
        class_cov = np.diag(p) - np.outer(p, p)
        H += np.kron(np.outer(x, x), class_cov)
    return H / len(X) + lam * np.eye(D * K)


def per_sample_grad_flat(theta_flat: np.ndarray, x: np.ndarray, yi: int) -> np.ndarray:
    D = len(x)
    p = softmax(x[None, :] @ theta_flat.reshape(D, K)).ravel()
    yvec = np.zeros(K)
    yvec[yi] = 1.0
    return np.outer(x, p - yvec).ravel()


def fit(X: np.ndarray, y: np.ndarray, lam: float = LAM, n_iter: int = 100, tol: float = 1e-10) -> np.ndarray:
    """Damped Newton fit of the strongly-convex regularized softmax risk."""
    theta = np.zeros(X.shape[1] * K, dtype=float)
    for _ in range(n_iter):
        H = _hessian(theta, X, lam)
        g = _grad(theta, X, y, lam)
        step = np.linalg.solve(H, g)
        theta_next = theta - step
        if norm(step) < tol:
            theta = theta_next
            break
        theta = theta_next
    return theta


# -----------------------------------------------------------------------------
# Convex unlearning and membership worlds
# -----------------------------------------------------------------------------

def influence_unlearn_delta(
    theta_flat: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    idx_unlearn: Sequence[int],
    lam: float = LAM,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the influence update Delta=(1/n) H^{-1} G_Du.

    Note the positive sign. For cross-entropy loss, removing D_u changes the
    gradient at the old optimum by -(1/n)G_Du, so the Newton correction is
    +(1/n)H^{-1}G_Du.
    """
    n = len(X)
    H = _hessian(theta_flat, X, lam)
    Hinv = np.linalg.inv(H)
    G_Du = np.zeros_like(theta_flat)
    for i in idx_unlearn:
        G_Du += per_sample_grad_flat(theta_flat, X[i], int(y[i]))
    Delta = (Hinv @ G_Du) / n
    return Delta, G_Du, H, Hinv


def dishonest_theta(theta: np.ndarray, delta: np.ndarray, eta: float) -> np.ndarray:
    """Scale an influence update by eta and add it to the fitted parameters."""
    return theta + eta * delta


def membership_world_endpoints(
    X: np.ndarray,
    y: np.ndarray,
    idx_unlearn: Sequence[int],
    idx_zstar: int,
    lam: float = LAM,
) -> Tuple[np.ndarray, np.ndarray]:
    """Construct member and non-member endpoints for one retained target.

    The member endpoint uses the full dataset; the non-member endpoint removes
    z*. Both endpoints apply the corresponding influence update for D_u. The
    target z* must not belong to D_u.
    """
    idx_unlearn = list(map(int, idx_unlearn))
    if idx_zstar in idx_unlearn:
        raise ValueError("z* must be retained, not in D_u")

    theta_mem = fit(X, y, lam=lam)
    delta_mem, _, _, _ = influence_unlearn_delta(theta_mem, X, y, idx_unlearn, lam)
    theta_u_mem = theta_mem + delta_mem

    keep = np.ones(len(X), dtype=bool)
    keep[idx_zstar] = False
    X_non, y_non = X[keep], y[keep]
    idx_u_non = [i if i < idx_zstar else i - 1 for i in idx_unlearn]
    theta_non = fit(X_non, y_non, lam=lam)
    delta_non, _, _, _ = influence_unlearn_delta(theta_non, X_non, y_non, idx_u_non, lam)
    theta_u_non = theta_non + delta_non
    return theta_u_mem, theta_u_non


# -----------------------------------------------------------------------------
# Behavioral Jacobian and exact geometry
# -----------------------------------------------------------------------------

def response_jacobian(theta_flat: np.ndarray, Xq: np.ndarray, response_class: int = 0) -> np.ndarray:
    """J_Q = nabla_theta F_Q(theta) for scalar class-posterior responses."""
    D = Xq.shape[1]
    P = predict_proba(theta_flat, Xq)
    J = np.empty((len(Xq), D * K), dtype=float)
    c = response_class
    for i, (x, p) in enumerate(zip(Xq, P)):
        coeff = -p[c] * p
        coeff = coeff.copy()
        coeff[c] += p[c]
        J[i] = np.outer(x, coeff).ravel()
    return J


@dataclass
class ExactGeometry:
    c: np.ndarray
    p: np.ndarray
    gamma: float
    a_signed: float
    p_perp: np.ndarray
    d_comp: float
    d_cur: float
    p_perp_norm: float
    decomposition_residual: float


def exact_geometry(
    theta_u_hon: np.ndarray,
    theta_u_dis: np.ndarray,
    theta_u_mem: np.ndarray,
    theta_u_non: np.ndarray,
    Xq: np.ndarray,
    response_class: int = 0,
) -> ExactGeometry:
    c = response(theta_u_hon, Xq, response_class) - response(theta_u_dis, Xq, response_class)
    p = response(theta_u_mem, Xq, response_class) - response(theta_u_non, Xq, response_class)
    c2 = float(c @ c)
    if c2 <= 1e-30:
        gamma = np.nan
        a_signed = np.nan
        p_perp = p.copy()
    else:
        a_signed = float((p @ c) / c2)
        gamma = abs(a_signed)
        p_perp = p - a_signed * c
    d_comp = float(norm(c))
    d_cur = float(norm(p))
    p_perp_norm = float(norm(p_perp))
    if np.isfinite(gamma):
        rhs2 = gamma**2 * d_comp**2 + p_perp_norm**2
        decomposition_residual = abs(d_cur**2 - rhs2)
    else:
        decomposition_residual = np.nan
    return ExactGeometry(c, p, gamma, a_signed, p_perp, d_comp, d_cur, p_perp_norm, decomposition_residual)


# -----------------------------------------------------------------------------
# Convex first-order proxy geometry
# -----------------------------------------------------------------------------

@dataclass
class ConvexProxy:
    c_tilde: np.ndarray
    p_tilde: np.ndarray
    rho: float
    kappa: float
    rho_q: float
    lambda_conv: float
    eps_c: float
    eps_p: float
    rel_eps_c: float
    rel_eps_p: float
    U: float
    p_tilde_perp_norm: float
    lambda_formula_residual: float


def _hinv_norm_sq(v: np.ndarray, Hinv: np.ndarray) -> float:
    return max(float(v @ Hinv @ v), 0.0)


def query_independent_geometry(g_z: np.ndarray, G_Du: np.ndarray, Hinv: np.ndarray, eta: float) -> Tuple[float, float]:
    """Compute query-independent geometry without materializing H^{-1/2}."""
    gg = _hinv_norm_sq(g_z, Hinv)
    GG = _hinv_norm_sq(G_Du, Hinv)
    denom = np.sqrt(gg * GG)
    rho = float((g_z @ Hinv @ G_Du) / denom) if denom > 1e-30 else np.nan
    if abs(1.0 - eta) <= 1e-14 or GG <= 1e-30:
        kappa = np.nan
    else:
        kappa = abs(rho) / abs(1.0 - eta) * np.sqrt(gg / GG)
    return rho, float(kappa)


def convex_proxy_geometry(
    theta: np.ndarray,
    delta: np.ndarray,
    G_Du: np.ndarray,
    Hinv: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    idx_zstar: int,
    eta: float,
    Xq: np.ndarray,
    exact: ExactGeometry,
    response_class: int = 0,
) -> ConvexProxy:
    """Compute first-order proxy geometry and realized approximation errors.

    The reported eps_c and eps_p are ||c-c_tilde|| and ||p-p_tilde||.
    """
    n = len(X)
    J = response_jacobian(theta, Xq, response_class)
    g_z = per_sample_grad_flat(theta, X[idx_zstar], int(y[idx_zstar]))

    delta_c = (1.0 - eta) * delta
    delta_p0 = -(Hinv @ g_z) / n
    c_tilde = J @ delta_c
    p_tilde = J @ delta_p0

    rho, kappa = query_independent_geometry(g_z, G_Du, Hinv, eta)

    ug = J @ (Hinv @ g_z)
    uG = J @ (Hinv @ G_Du)
    ng, nG = norm(ug), norm(uG)
    rho_q = float((ug @ uG) / (ng * nG)) if ng > 1e-30 and nG > 1e-30 else np.nan

    c2 = float(c_tilde @ c_tilde)
    if c2 <= 1e-30:
        lambda_conv = np.nan
        p_tilde_perp_norm = float(norm(p_tilde))
    else:
        a = float((p_tilde @ c_tilde) / c2)
        lambda_conv = abs(a)
        p_tilde_perp_norm = float(norm(p_tilde - a * c_tilde))

    if np.isfinite(rho_q) and abs(1.0 - eta) > 1e-14 and nG > 1e-30:
        lambda_formula = abs(rho_q) / abs(1.0 - eta) * (ng / nG)
        lambda_formula_residual = abs(lambda_conv - lambda_formula)
    else:
        lambda_formula_residual = np.nan

    eps_c = float(norm(exact.c - c_tilde))
    eps_p = float(norm(exact.p - p_tilde))
    rel_eps_c = eps_c / max(exact.d_comp, 1e-30)
    rel_eps_p = eps_p / max(exact.d_cur, 1e-30)
    U = float(lambda_conv * eps_c + eps_p) if np.isfinite(lambda_conv) else np.nan

    return ConvexProxy(
        c_tilde=c_tilde,
        p_tilde=p_tilde,
        rho=rho,
        kappa=kappa,
        rho_q=rho_q,
        lambda_conv=float(lambda_conv),
        eps_c=eps_c,
        eps_p=eps_p,
        rel_eps_c=rel_eps_c,
        rel_eps_p=rel_eps_p,
        U=U,
        p_tilde_perp_norm=p_tilde_perp_norm,
        lambda_formula_residual=float(lambda_formula_residual),
    )


# -----------------------------------------------------------------------------
# Gaussian transcript metrics
# -----------------------------------------------------------------------------

def gaussian_tv(distance: float, sigma: float) -> float:
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    return float(2.0 * ndtr(distance / (2.0 * sigma)) - 1.0)


def optimal_compliance_error(d_comp: float, sigma: float) -> float:
    """Bayes-optimal alpha+alpha' for equal-covariance Gaussian endpoints.

    Computed as 2*Phi(-d/(2 sigma)) rather than 1-TV to avoid catastrophic
    cancellation when the compliance distributions are nearly separable.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    return float(2.0 * ndtr(-d_comp / (2.0 * sigma)))


def qe_from_e(e: float) -> float:
    e = float(np.clip(e, 0.0, 1.0))
    return float(ndtri(1.0 - e / 2.0))


def theorem_values(exact: ExactGeometry, proxy: ConvexProxy, sigma: float) -> Dict[str, float]:
    """Compute exact distinguishability and geometric lower bounds."""
    beta = gaussian_tv(exact.d_cur, sigma)
    e = optimal_compliance_error(exact.d_comp, sigma)
    # This equivalent form remains stable when e is below floating-point precision.
    qe = float(exact.d_comp / (2.0 * sigma))

    if np.isfinite(exact.gamma):
        beta_identity = float(2.0 * ndtr(
            np.sqrt(exact.gamma**2 * exact.d_comp**2 + exact.p_perp_norm**2) / (2.0 * sigma)
        ) - 1.0)
        beta_main = float(2.0 * ndtr(exact.gamma * qe) - 1.0)
        beta_linear = float(min(exact.gamma, 1.0) * (1.0 - e))
    else:
        beta_identity = beta_main = beta_linear = np.nan

    if np.isfinite(proxy.lambda_conv) and np.isfinite(proxy.U):
        arg = max(proxy.lambda_conv * qe - proxy.U / (2.0 * sigma), 0.0)
        beta_cert = float(2.0 * ndtr(arg) - 1.0)
        beta_cert_linear = max(
            min(proxy.lambda_conv, 1.0) * (1.0 - e) - proxy.U / (sigma * np.sqrt(2.0 * np.pi)),
            0.0,
        )
    else:
        beta_cert = beta_cert_linear = np.nan

    return dict(
        sigma=float(sigma), e=e, qe=qe, beta=beta,
        beta_identity=beta_identity,
        beta_main_bound=beta_main,
        beta_linear_bound=beta_linear,
        beta_cert_bound=beta_cert,
        beta_cert_linear_bound=float(beta_cert_linear),
        identity_residual=abs(beta - beta_identity) if np.isfinite(beta_identity) else np.nan,
    )


# -----------------------------------------------------------------------------
# Experiment instance and query protocols
# -----------------------------------------------------------------------------

def setup_instance(seed: int, n_per: int = 120, du_per_class: int = 8, lam: float = LAM) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    X2, y = make_data(n_per=n_per, rng=rng)
    X = feature_lift(X2)
    theta = fit(X, y, lam=lam)

    idx_Du = np.concatenate([
        rng.choice(np.where(y == k)[0], size=du_per_class, replace=False)
        for k in range(K)
    ]).astype(int)
    Du_mask = np.zeros(len(y), dtype=bool)
    Du_mask[idx_Du] = True

    Delta, G_Du, H, Hinv = influence_unlearn_delta(theta, X, y, idx_Du, lam)
    retained = ~Du_mask

    # Query-independent rho for every retained candidate z*.
    GG = _hinv_norm_sq(G_Du, Hinv)
    rhos = np.zeros(len(y), dtype=float)
    g_hinv_norms = np.zeros(len(y), dtype=float)
    for i in range(len(y)):
        gi = per_sample_grad_flat(theta, X[i], int(y[i]))
        gg = _hinv_norm_sq(gi, Hinv)
        g_hinv_norms[i] = np.sqrt(gg)
        denom = np.sqrt(gg * GG)
        rhos[i] = (gi @ Hinv @ G_Du) / denom if denom > 1e-30 else 0.0

    return dict(
        rng=rng, X2=X2, y=y, X=X, theta=theta,
        idx_Du=idx_Du, Du_mask=Du_mask, retained=retained,
        Delta=Delta, G_Du=G_Du, H=H, Hinv=Hinv,
        theta_u_hon=theta + Delta,
        rhos=rhos, g_hinv_norms=g_hinv_norms,
        G_Du_hinv_norm=np.sqrt(GG),
        lam=np.asarray(lam),
    )


def pick_zstars_by_rho(rhos: np.ndarray, retained: np.ndarray) -> List[int]:
    retained_idx = np.where(retained)[0]
    pos = retained_idx[rhos[retained_idx] > 0]
    neg = retained_idx[rhos[retained_idx] < 0]
    near = retained_idx[np.argmin(np.abs(rhos[retained_idx]))]
    if len(pos) == 0 or len(neg) == 0:
        # Fall back to extrema even if all correlations have one sign.
        return [int(retained_idx[np.argmax(rhos[retained_idx])]), int(near), int(retained_idx[np.argmin(rhos[retained_idx])])]
    return [int(pos[np.argmax(rhos[pos])]), int(near), int(neg[np.argmin(rhos[neg])])]


def make_query_grid(lim: float = LIM, n_grid: int = 81) -> Tuple[np.ndarray, np.ndarray]:
    xs = np.linspace(-lim, lim, n_grid)
    XX, YY = np.meshgrid(xs, xs)
    X2q = np.stack([XX.ravel(), YY.ravel()], axis=1)
    return X2q, feature_lift(X2q)


def _quadrant_mask(X2q: np.ndarray, quadrant: int) -> np.ndarray:
    x, y = X2q[:, 0], X2q[:, 1]
    if quadrant == 1:
        return (x >= 0) & (y >= 0)
    if quadrant == 2:
        return (x < 0) & (y >= 0)
    if quadrant == 3:
        return (x < 0) & (y < 0)
    if quadrant == 4:
        return (x >= 0) & (y < 0)
    raise ValueError("quadrant must be 1..4")


def select_query_indices(
    protocol: str,
    T: int,
    X2_candidates: np.ndarray,
    X_candidates: np.ndarray,
    theta_u_hon: np.ndarray,
    theta_u_dis: np.ndarray,
    response_class: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Select a fixed query sequence Q without using z* or D_r membership.

    Supported protocols:
      random             - uniform candidate queries;
      compliance         - top |single-query compliance signal|;
      q1/q2/q3/q4        - top compliance queries restricted to a quadrant.

    Compliance-based protocols depend on the endpoint pair and D_u, but not on
    the retained target z*.
    """
    if T <= 0:
        raise ValueError("T must be positive")
    if T > len(X_candidates):
        raise ValueError("T exceeds candidate query pool")

    if protocol == "random":
        return np.sort(rng.choice(len(X_candidates), size=T, replace=False))

    comp = np.abs(
        response(theta_u_hon, X_candidates, response_class)
        - response(theta_u_dis, X_candidates, response_class)
    )
    eligible = np.ones(len(X_candidates), dtype=bool)
    if protocol.startswith("q") and len(protocol) == 2 and protocol[1].isdigit():
        eligible = _quadrant_mask(X2_candidates, int(protocol[1]))
    elif protocol != "compliance":
        raise ValueError(f"unknown query protocol: {protocol}")

    idx = np.where(eligible)[0]
    if T > len(idx):
        raise ValueError(f"T={T} exceeds eligible candidates for {protocol}: {len(idx)}")
    order = idx[np.argsort(comp[idx])[::-1]]
    return np.sort(order[:T])


def row_from_geometry(
    *,
    seed: int,
    z_role: str,
    z_idx: int,
    eta: float,
    protocol: str,
    T: int,
    response_class: int,
    inst: Dict[str, np.ndarray],
    exact: ExactGeometry,
    proxy: ConvexProxy,
) -> Dict[str, float]:
    return dict(
        seed=seed,
        z_role=z_role,
        z_idx=int(z_idx),
        z_rho=float(inst["rhos"][z_idx]),
        eta=float(eta),
        query_protocol=protocol,
        T=int(T),
        response_class=int(response_class),
        kappa=float(proxy.kappa),
        rho_q=float(proxy.rho_q),
        lambda_conv=float(proxy.lambda_conv),
        gamma=float(exact.gamma),
        a_signed=float(exact.a_signed),
        d_comp=float(exact.d_comp),
        d_cur=float(exact.d_cur),
        p_perp_norm=float(exact.p_perp_norm),
        p_tilde_perp_norm=float(proxy.p_tilde_perp_norm),
        eps_c=float(proxy.eps_c),
        eps_p=float(proxy.eps_p),
        rel_eps_c=float(proxy.rel_eps_c),
        rel_eps_p=float(proxy.rel_eps_p),
        U=float(proxy.U),
        decomposition_residual=float(exact.decomposition_residual),
        lambda_formula_residual=float(proxy.lambda_formula_residual),
    )
