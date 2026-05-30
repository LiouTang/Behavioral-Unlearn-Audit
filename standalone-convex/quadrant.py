import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",      type=int, default=7)
    parser.add_argument("--n_class",   type=int, default=120)
    parser.add_argument("--n_u_class", type=int, default=8)
    parser.add_argument("--out_dir",   type=str, default="results")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    inst = utils.setup_instance(args.seed, n_per=args.n_class, du_per_class=args.n_u_class)
    X2, y, X = inst["X2"], inst["y"], inst["X"]
    n, D = X.shape
    rhos, g_norms = inst["rhos"], inst["g_norms"]
    retained = inst["retained"]
    idx_Du = inst["idx_Du"]

    P_train = utils.predict_proba(inst["theta"], X, D)
    print(f"n={n}, D={D}, K={utils.K}")
    print(f"train accuracy: {(P_train.argmax(axis=1) == y).mean():.3f}")

    # Three representative z*: highest |rho|, near-median |rho|, lowest |rho|
    high_idx = int(np.argmax(np.where(retained, (rhos), -1.0)))

    z_idx = high_idx
    pred_ratio = abs(rhos[z_idx]) * g_norms[z_idx] / inst["G_Du_w_norm"]
    print(f"\nPredicted Cur/Comp ratio at high-|ρ| z*: {pred_ratio:.4f}")

    # Decision regions of theta_u_hon on a 500×500 grid
    xs = np.linspace(-utils.LIM, utils.LIM, 500)
    XX, YY = np.meshgrid(xs, xs)
    Q = utils.feature_lift(np.stack([XX.ravel(), YY.ravel()], axis=1))
    pred_hon = utils.predict_proba(inst["theta_u_hon"], Q, D).argmax(axis=1).reshape(XX.shape)

    cs = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]
    cmap_oi = LinearSegmentedColormap.from_list("okabe_ito", cs)

    fig, ax = utils.single_ax_fig(margin=(0.5, 0.5), ax_size=(7.4, 7.4))
    ax.imshow(pred_hon, origin="lower",
              extent=[-utils.LIM, utils.LIM, -utils.LIM, utils.LIM],
              cmap=cmap_oi, alpha=0.3)
    ax.scatter(X2[:, 0], X2[:, 1], s=30, c=[cs[k] for k in y])
    ax.scatter([], [], s=50, c="gray", label=r"$D$")
    ax.scatter(X2[idx_Du, 0], X2[idx_Du, 1], s=200, edgecolors="black",
               facecolors="none", linewidths=1.3, label=r"$D_u$")
    ax.scatter(X2[z_idx, 0], X2[z_idx, 1], s=600, marker="*",
               edgecolors="k", facecolors="gold", linewidths=0.7, label=r"$z^*$", zorder=10)
    ax.axhline(0, c="k", lw=0.4); ax.axvline(0, c="k", lw=0.4)
    ax.set_xlim(-utils.LIM, utils.LIM); ax.set_ylim(-utils.LIM, utils.LIM)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=24)

    plt.savefig(os.path.join(args.out_dir, "decision_space.pdf"), dpi=300, bbox_inches=None)
    plt.close(fig)


if __name__ == "__main__":
    main()
