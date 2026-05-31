import os
import argparse

import numpy as np
import pandas as pd


import matplotlib
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import Divider, Size
from matplotlib.patches import Ellipse
from matplotlib import cm, colors

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



def plot_conditional_shift(
    df,
    ax,
    x: str = "a",
    y: str = "b",
    t: str = "T",
    nsig: float = 1.0,
    ellipse_stride: int = 2,
    cmap="plasma",
    scatter_kwargs: dict | None = None,
    ellipse_kwargs: dict | None = None,
    trajectory_kwargs: dict | None = None,
    centroid_kwargs: dict | None = None,
):
    """
    Parameters
    ----------
    df : pandas.DataFrame with columns named by `x`, `y`, `t`.
    ax : matplotlib Axes to draw onto.
    x, y, t : column names.
    nsig : ellipse radius in std-devs (2.0 -> ~86% Gaussian mass).
    ellipse_stride : draw every k-th ellipse (1 = all).
    cmap : colormap name or Colormap.
    *_kwargs : style overrides for each layer (scatter / ellipse / line / centroid).

    Returns
    -------
    dict with keys:
        'bin_centers' : ndarray of t bin midpoints used,
        'mu'          : (K, 2) array of empirical conditional means,
        'Sigma'       : list of (2, 2) empirical covariances,
    """
    # --- extract & clean ---
    sub = df[[x, y, t]].dropna()
    a = sub[x].to_numpy()
    b = sub[y].to_numpy()
    T = sub[t].to_numpy()

    cmap = plt.get_cmap(cmap)
    norm = colors.Normalize(vmin=float(T.min()), vmax=float(T.max() * 3/2))

    # --- style defaults (user overrides win) ---
    sk = {"s": 30, "alpha": 0.25, "edgecolors": "none", "rasterized": True,
          **(scatter_kwargs or {})}
    ek = {"facecolor": "none", "lw": 2, **(ellipse_kwargs or {})}
    tk = {"color": "k", "lw": 2, "zorder": 4,
          **(trajectory_kwargs or {})}
    ck = {"s": 180, "edgecolor": "w", "lw": 1.0, "zorder": 5, "marker": "^",
          **(centroid_kwargs or {})}

    # --- scatter ---
    ax.scatter(a, b, c=T, cmap=cmap, norm=norm, **sk)

    # --- bin t, compute empirical mu_k and Sigma_k ---
    centers, mus, covs = [], [], []
    for ti, g in sub.groupby(t, sort=True):
        # if len(g) < min_per_bin:
        #     continue
        centers.append(float(ti))
        pts_k = g[[x, y]].to_numpy()
        mus.append(pts_k.mean(axis=0))
        covs.append(np.cov(pts_k.T))

    centers = np.asarray(centers)
    mus = np.asarray(mus).reshape(-1, 2)

    # --- mu(t) trajectory + centroid markers ---
    if len(centers) >= 2:
        ax.plot(mus[:, 0], mus[:, 1], "-", **tk)
        ax.scatter(mus[:, 0], mus[:, 1], c=centers, cmap=cmap, norm=norm, **ck)

    # --- Sigma(t) ellipses ---
    for c, m, S in list(zip(centers, mus, covs))[::ellipse_stride]:
        vals, vecs = np.linalg.eigh(S)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        angle = float(np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0])))
        w, h = 2 * nsig * np.sqrt(np.maximum(vals, 0.0))
        ellipse_style = {"edgecolor": "w", "zorder": 3, **ek}
        ellipse_style["lw"] = 5
        ax.add_patch(Ellipse(xy=m, width=w, height=h, angle=angle, **ellipse_style))
        ellipse_style = {"edgecolor": cmap(norm(c)), "zorder": 4, **ek}
        ax.add_patch(Ellipse(xy=m, width=w, height=h, angle=angle, **ellipse_style))


    for t in np.unique(T):
        if t % 10 == 0:
            ax.scatter([], [], s=60, color=cmap(norm(t)), label=f"T={t}")
    ax.legend(loc="upper right", framealpha=0.25, fontsize=24, facecolor="lightgray")

    ax.set_xlabel(r"Compliance error $\alpha + \alpha^\prime$", fontsize=24)
    ax.set_ylabel(r"Curiosity advantage $\beta$", fontsize=24)

    out = {"bin_centers": centers, "mu": mus, "Sigma": covs}
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default=None, required=True)
    args = parser.parse_args()
    # go through all sub-folders with name that start with "results_":
    for folder in os.listdir(args.base_dir):
        # if not .zip
        if folder.startswith("results_") and not folder.endswith(".zip"):
            fig, ax = single_ax_fig(margin=(1.0, 1.0), ax_size=(6.9, 6.9))
            df = pd.read_csv(os.path.join(args.base_dir, folder, "audit_results.csv"))
            df["a_a"] = df["alpha"] + df["alpha_prime"]
            plot_conditional_shift(df, ax, x="a_a", y="beta", t="T", cmap="plasma")
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            ax.grid(True, which="both", linestyle="--", alpha=0.3)

            with open(os.path.join(args.base_dir, folder, "env_audit.pkl"), "rb") as f:
                env_audit = pd.read_pickle(f)
            plt.savefig(os.path.join(args.base_dir, f"{env_audit['args'].unlearn}_{env_audit['args'].eta}.pdf"), dpi=300, bbox_inches=None)
            # plt.show()
            # exit(0)


if __name__ == "__main__":
    main()