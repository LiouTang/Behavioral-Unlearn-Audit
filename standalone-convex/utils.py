import numpy as np
from numpy.linalg import inv, norm, cholesky
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import Divider, Size


K = 4       # number of classes / quadrants
LAM = 5e-3  # default L2 regularization
LIM = 4.0   # default 2-D grid half-extent

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
cmap_okabe_ito = matplotlib.colors.ListedColormap(
    ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
)
matplotlib.rcParams["axes.prop_cycle"] = matplotlib.cycler(color=cmap_okabe_ito.colors)
cmap_T = LinearSegmentedColormap.from_list("white_red", ["#ffffff", "#0072B2"], N=256)


class Sci1dp(ticker.ScalarFormatter):
    """ScalarFormatter locked to 1 decimal place in the mantissa (a.b × 10^k)."""
    def _set_format(self):
        self.format = "%1.1f"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def make_data(n_per=120, rng=None):
    """Per-class Gaussian blob centered in each quadrant; returns (X2, y) shuffled."""
    if rng is None:
        rng = np.random.default_rng()
    pts, labels = [], []
    for k, (cx, cy) in enumerate([(1.6, 1.6), (-1.6, 1.6), (-1.6, -1.6), (1.6, -1.6)]):
        Z = rng.normal(loc=[cx, cy], scale=0.9, size=(n_per, 2))
        pts.append(Z)
        labels.extend([k] * n_per)
    X2, y = np.vstack(pts), np.array(labels)
    perm = rng.permutation(len(y))
    return X2[perm], y[perm]


def feature_lift(X2):
    """phi(x) = (x1, x2, x1*x2, |x1|, |x2|, 1) — D = 6."""
    x1, x2 = X2[:, 0], X2[:, 1]
    return np.stack([x1, x2, x1 * x2, np.abs(x1), np.abs(x2), np.ones_like(x1)], axis=1)


# ---------------------------------------------------------------------------
# Softmax model: forward, gradients, Hessian (internal helpers prefixed _)
# ---------------------------------------------------------------------------
def softmax(Z):
    Z = Z - Z.max(axis=-1, keepdims=True)
    eZ = np.exp(Z)
    return eZ / eZ.sum(axis=-1, keepdims=True)


def predict_proba(theta_flat, X, D):
    return softmax(X @ theta_flat.reshape(D, K))


def _grad(theta_flat, X, y, D, lam=LAM):
    theta = theta_flat.reshape(D, K)
    n = X.shape[0]
    P = softmax(X @ theta)
    Y = np.zeros_like(P); Y[np.arange(n), y] = 1.0
    return (X.T @ (P - Y) / n + lam * theta).flatten()


def _hessian(theta_flat, X, D, lam=LAM):
    theta = theta_flat.reshape(D, K)
    n = X.shape[0]
    P = softmax(X @ theta)
    H = np.zeros((D * K, D * K))
    for i in range(n):
        p = P[i]
        H += np.kron(np.outer(X[i], X[i]), np.diag(p) - np.outer(p, p))
    return H / n + lam * np.eye(D * K)


def per_sample_grad_flat(theta_flat, x, yi, D):
    p = softmax(x[None, :] @ theta_flat.reshape(D, K)).ravel()
    yvec = np.zeros(K); yvec[yi] = 1.0
    return np.outer(x, p - yvec).flatten()


def fit(X, y, lam=LAM, n_iter=80, tol=1e-9):
    D = X.shape[1]
    theta = np.zeros(D * K)
    for _ in range(n_iter):
        step = np.linalg.solve(_hessian(theta, X, D, lam), _grad(theta, X, y, D, lam))
        theta -= step
        if norm(step) < tol:
            break
    return theta


# ---------------------------------------------------------------------------
# Unlearning
# ---------------------------------------------------------------------------
def newton_delta(theta_flat, X, y, idx_unlearn, lam=LAM):
    """Delta = -(1/n) H^{-1} sum_{z in D_u} grad L(theta, z).  Honest theta_u = theta + Delta."""
    D, n = X.shape[1], X.shape[0]
    Hinv = inv(_hessian(theta_flat, X, D, lam))
    G_Du = np.zeros(D * K)
    for i in idx_unlearn:
        G_Du += per_sample_grad_flat(theta_flat, X[i], y[i], D)
    return -(1.0 / n) * Hinv @ G_Du, G_Du, Hinv


def dishonest_theta(theta, delta, eta):
    """theta_u*(eta) = theta + eta * Delta."""
    return theta + eta * delta


def models_mem_vs_non(X_pool, y_pool, idx_unlearn, idx_zstar, lam=LAM):
    """Curiosity world pair: 'mem' trained with z*, 'non' trained without z* — both honestly unlearn."""
    theta_mem = fit(X_pool, y_pool, lam=lam)
    delta_m, _, _ = newton_delta(theta_mem, X_pool, y_pool, idx_unlearn, lam)
    theta_u_mem = theta_mem + delta_m

    keep = [i for i in range(len(X_pool)) if i != idx_zstar]
    X_no, y_no = X_pool[keep], y_pool[keep]
    new_idx_u = [i if i < idx_zstar else i - 1 for i in idx_unlearn if i != idx_zstar]
    theta_non = fit(X_no, y_no, lam=lam)
    delta_n, _, _ = newton_delta(theta_non, X_no, y_no, new_idx_u, lam)
    return theta_u_mem, theta_non + delta_n


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------
def sig_comp(theta_u_hon, theta_u_dis_eta, X):
    """Compliance signal: half-L1 of posterior difference (multi-class TV)."""
    D = X.shape[1]
    return 0.5 * np.abs(predict_proba(theta_u_hon, X, D) - predict_proba(theta_u_dis_eta, X, D)).sum(axis=1)


def sig_cur(theta_u_mem, theta_u_non, X):
    """Curiosity signal: half-L1 of posterior difference."""
    D = X.shape[1]
    return 0.5 * np.abs(predict_proba(theta_u_mem, X, D) - predict_proba(theta_u_non, X, D)).sum(axis=1)


def signals_on_grid(theta_u_hon, theta_u_dis_eta, theta_u_mem, theta_u_non, lim=LIM, n_grid=80):
    """Evaluate Sig^Comp and Sig^Cur on a 2-D grid; returns XX, YY, s_comp, s_cur."""
    xs = np.linspace(-lim, lim, n_grid)
    XX, YY = np.meshgrid(xs, xs)
    Q = feature_lift(np.stack([XX.ravel(), YY.ravel()], axis=1))
    sc = sig_comp(theta_u_hon, theta_u_dis_eta, Q).reshape(n_grid, n_grid)
    su = sig_cur(theta_u_mem, theta_u_non, Q).reshape(n_grid, n_grid)
    return XX, YY, sc, su


# ---------------------------------------------------------------------------
# Whitened geometry
# ---------------------------------------------------------------------------
def whitening_cholesky(Hinv):
    Hinv_psd = (Hinv + Hinv.T) / 2.0
    return cholesky(Hinv_psd + 1e-9 * np.eye(Hinv.shape[0]))


def per_sample_geometry(theta, X, y, idx_Du, Hinv, lam=LAM):
    """Return G_Du, G_Du_w, G_Du_w_norm, g_norms, rhos (whitened cosine similarities)."""
    n, D = X.shape
    L = whitening_cholesky(Hinv)

    G_Du = np.zeros(D * K)
    for i in idx_Du:
        G_Du += per_sample_grad_flat(theta, X[i], y[i], D)
    G_Du_w = L.T @ G_Du
    G_Du_w_norm = norm(G_Du_w)

    g_norms = np.zeros(n)
    rhos = np.zeros(n)
    for i in range(n):
        gi_w = L.T @ per_sample_grad_flat(theta, X[i], y[i], D)
        g_norms[i] = norm(gi_w)
        if g_norms[i] > 1e-10 and G_Du_w_norm > 1e-10:
            rhos[i] = (gi_w @ G_Du_w) / (g_norms[i] * G_Du_w_norm)
    return G_Du, G_Du_w, G_Du_w_norm, g_norms, rhos


def kappa_theory(rho_z, g_norm_z, G_Du_w_norm, eta=0.0):
    """kappa(z*) = (|rho| / (1-eta)) * ||g_z*||_w / ||G_Du||_w."""
    return (abs(rho_z) / max(1.0 - eta, 1e-12)) * (g_norm_z / max(G_Du_w_norm, 1e-12))


# ---------------------------------------------------------------------------
# Empirical slopes
# ---------------------------------------------------------------------------
def empirical_slope(sc, su, top_frac=0.10):
    """LS slope of Sig^Cur on Sig^Comp at the top-fraction-by-compliance queries."""
    scR, suR = sc.ravel(), su.ravel()
    m = scR > np.quantile(scR, 1.0 - top_frac)
    if m.sum() < 5:
        return np.nan
    return float(np.sum(scR[m] * suR[m]) / max(np.sum(scR[m] ** 2), 1e-12))


def empirical_slope_median(sc, su, top_frac=0.10):
    """Median ratio (robust) of Sig^Cur / Sig^Comp at top-compliance queries."""
    scR, suR = sc.ravel(), su.ravel()
    m = scR > np.quantile(scR, 1.0 - top_frac)
    if m.sum() < 5:
        return np.nan
    return float(np.median(suR[m] / (scR[m] + 1e-12)))


# ---------------------------------------------------------------------------
# Instance setup
# ---------------------------------------------------------------------------
def setup_instance(seed, n_per=120, du_per_class=8, lam=LAM):
    """Build a complete seeded experiment instance: data → model → D_u → geometry.

    Returns a dict with keys: rng, X2, y, X, theta, idx_Du, Du_mask, retained,
    Delta, G_Du, Hinv, theta_u_hon, G_Du_w_norm, g_norms, rhos.
    """
    rng = np.random.default_rng(seed)
    X2, y = make_data(n_per=n_per, rng=rng)
    X = feature_lift(X2)
    n = X.shape[0]

    theta = fit(X, y, lam=lam)

    idx_Du = np.concatenate([
        rng.choice(np.where(y == k)[0], size=du_per_class, replace=False)
        for k in range(K)
    ])
    Du_mask = np.zeros(n, dtype=bool)
    Du_mask[idx_Du] = True

    Delta, G_Du, Hinv = newton_delta(theta, X, y, idx_Du, lam)
    _, _, G_Du_w_norm, g_norms, rhos = per_sample_geometry(theta, X, y, idx_Du, Hinv, lam)

    return dict(
        rng=rng, X2=X2, y=y, X=X, theta=theta,
        idx_Du=idx_Du, Du_mask=Du_mask, retained=~Du_mask,
        Delta=Delta, G_Du=G_Du, Hinv=Hinv,
        theta_u_hon=theta + Delta,
        G_Du_w_norm=G_Du_w_norm, g_norms=g_norms, rhos=rhos,
    )


def top_rho_idx(rhos, retained):
    """Index of the retained sample with the highest |rho|."""
    if retained.dtype != bool:
        mask = np.zeros(len(rhos), dtype=bool)
        mask[retained] = True
        retained = mask
    return int(np.argmax(np.where(retained, np.abs(rhos), -1.0)))


def pick_zstars_by_rho(rhos, retained):
    """Three representative z* indices: highest-positive, near-zero, highest-negative rho.

    `retained` may be a boolean mask or an array of retained indices.
    """
    if retained.dtype != bool:
        mask = np.zeros(len(rhos), dtype=bool)
        mask[retained] = True
        retained = mask
    pos = np.where((rhos > 0) & retained)[0]
    neg = np.where((rhos < 0) & retained)[0]
    valid = np.where(retained)[0]
    return [
        pos[np.argmax(rhos[pos])],
        valid[np.argmin(np.abs(rhos[valid]))],
        neg[np.argmin(rhos[neg])],
    ]


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
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


def plot_heatmap(ax, XX, YY, s_comp, s_cur, X2, idx_Du, z_idx, lim=LIM, vmax=None):
    """Heatmap of Sig^Comp with Sig^Cur contour overlay and data scatter.

    Returns the AxesImage (for attaching a colorbar).
    """
    im = ax.imshow(s_comp, origin="lower", extent=[-lim, lim, -lim, lim],
                   cmap=cmap_T, vmin=0, vmax=(vmax or s_comp.max()), zorder=0)
    levels = np.quantile(s_cur[s_cur > 0], [0.9, 0.95, 0.99])
    c_cur = ax.contour(XX, YY, s_cur, levels=levels, colors="k", linewidths=1.5, zorder=1)
    ax.clabel(c_cur, fmt={l: f"{q:.0%}" for l, q in zip(levels, [0.9, 0.95, 0.99])}, fontsize=7)
    ax.scatter(X2[idx_Du, 0], X2[idx_Du, 1], s=25, c="gray", alpha=0.75, label=r"$D_u$", zorder=3)
    ax.scatter(X2[z_idx, 0], X2[z_idx, 1], s=350, marker="*",
               edgecolors="k", facecolors="gold", linewidths=0.5, label=r"$z^*$", zorder=5)
    ax.axhline(0, c="gray", lw=0.3); ax.axvline(0, c="gray", lw=0.3)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xticks([-4, -2, 0, 2, 4]); ax.set_yticks([-4, -2, 0, 2, 4])
    return im


def plot_signal_scatter(ax, s_comp, s_cur, rho_z, g_norm_z, G_Du_w_norm, eta=0.0):
    """Scatter of Sig^Cur vs Sig^Comp; highlights top-10% compliance queries and kappa line."""
    scR, suR = s_comp.ravel(), s_cur.ravel()
    top10 = scR > np.quantile(scR, 0.90)
    ax.scatter(scR[~top10], suR[~top10], s=5, c="lightgray", rasterized=True)
    ax.scatter([], [], s=50, c="lightgray", label="All queries")
    ax.scatter(scR[top10], suR[top10], s=10, c="#0072B2", alpha=0.25, rasterized=True)
    ax.scatter([], [], s=50, c="#0072B2", alpha=0.75, label="Top-10% compliance queries")
    kappa = kappa_theory(rho_z, g_norm_z, G_Du_w_norm, eta)
    xs = np.linspace(0, scR.max(), 50)
    ax.plot(xs, kappa * xs, "k--", lw=1.5, label=f"Theoretical $\\kappa(\\eta)$={kappa:.2f}")
    ax.set_xlim(0, scR.max()); ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", fontsize=24, framealpha=0.5)
    ax.grid(True, which="both", ls="--", lw=0.5, alpha=0.3)
    for axis in (ax.xaxis, ax.yaxis):
        fmt = Sci1dp(useMathText=True)
        fmt.set_powerlimits((0, 0))
        axis.set_major_formatter(fmt)
