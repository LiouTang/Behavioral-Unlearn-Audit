import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
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
    retained = inst["retained"]

    kappas = np.array([
        utils.kappa_theory(inst["rhos"][i], inst["g_norms"][i], inst["G_Du_w_norm"])
        if retained[i] else 0.0
        for i in range(len(inst["rhos"]))
    ])
    kappas_r = kappas[retained]

    fig, ax = utils.single_ax_fig(margin=(1.0, 1.0))

    cmap_p = plt.cm.plasma
    ax.hist(kappas_r, bins=20, color="#0072B2", edgecolor="k", alpha=0.25,
            linewidth=0.5, density=True)

    median = np.median(kappas_r)
    p90    = np.quantile(kappas_r, 0.90)
    p99    = np.quantile(kappas_r, 0.99)
    ax.axvline(median, c=cmap_p(2 / 3), ls="--", lw=4.5, label=f"median = {median:.3f}")
    ax.axvline(p90,    c=cmap_p(1 / 3), ls="--", lw=4.5, label=rf"$\kappa$ @ 90% = {p90:.3f}")
    ax.axvline(p99,    c=cmap_p(0 / 3), ls="--", lw=4.5, label=rf"$\kappa$ @ 99% = {p99:.3f}")

    ax.set_xlabel(r"$\kappa(z^*)$", fontdict={"fontsize": 24})
    ax.set_ylabel(r"$\log \Pr(z^* \mid \kappa)$", fontdict={"fontsize": 24})
    ax.set_xlim(0.0, kappas_r.max() * 1.01)
    ax.set_yscale("log")
    ax.legend(fontsize=24, loc="upper right", framealpha=0.5)
    ax.grid(ls="--", alpha=0.3)

    plt.savefig(os.path.join(args.out_dir, "kappa_distribution.pdf"), dpi=300, bbox_inches=None)
    plt.close()


if __name__ == "__main__":
    main()
