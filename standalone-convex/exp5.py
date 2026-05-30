
import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import utils
from utils import K

DU_PER_CLASS_VALUES = np.arange(1, 40).tolist()


def run_one(seed, du_per_class, n_class):
    """Return geometry stats for one (seed, du_per_class) combination."""
    inst = utils.setup_instance(seed * 1000 + du_per_class, n_per=n_class,
                                du_per_class=du_per_class)
    retained = inst["retained"]
    kappas = np.full(len(inst["rhos"]), np.nan)
    for i in np.where(retained)[0]:
        kappas[i] = utils.kappa_theory(inst["rhos"][i], inst["g_norms"][i], inst["G_Du_w_norm"])
    return dict(
        du_size=int(inst["Du_mask"].sum()),
        G_Du_w_norm=float(inst["G_Du_w_norm"]),
        kappa_mean=float(np.nanmean(kappas[retained])),
        kappa_p90=float(np.nanquantile(kappas[retained], 0.90)),
        kappa_p99=float(np.nanquantile(kappas[retained], 0.99)),
        kappa_max=float(np.nanmax(kappas[retained])),
        rho_med=float(np.median(np.abs(inst["rhos"][retained]))),
        rho_p90=float(np.quantile(np.abs(inst["rhos"][retained]), 0.90)),
        g_norm_med=float(np.median(inst["g_norms"][retained])),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_seeds",   type=int, default=30)
    parser.add_argument("--n_class",   type=int, default=120)
    parser.add_argument("--out_dir",   type=str, default="results")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    from tqdm import tqdm
    results = []
    for du_per_class in tqdm(DU_PER_CLASS_VALUES):
        for s in range(args.n_seeds):
            r = run_one(s, du_per_class, args.n_class)
            r["seed"] = s
            r["du_per_class"] = du_per_class
            r["du_size"] = r["du_size"] / (K * args.n_class)
            results.append(r)

    df = pd.DataFrame(results)
    du_sizes = sorted(df["du_size"].unique())

    fig, ax = utils.single_ax_fig(margin=(1.0, 1.0))
    ax.set_yscale("log")

    cmap_p = plt.cm.plasma
    sns.lineplot(data=df, x="du_size", y="kappa_mean",
                 errorbar="sd", marker="o", markersize=8, lw=2,
                 color=cmap_p(2 / 3), label=r"mean ± 1 std over $D$", ax=ax)
    sns.lineplot(data=df, x="du_size", y="kappa_p90",
                 estimator=np.mean, errorbar="sd", marker="s", markersize=7, lw=2,
                 color=cmap_p(1 / 3), label=r"$\kappa$ @ 90%", ax=ax)
    sns.lineplot(data=df, x="du_size", y="kappa_p99",
                 estimator=np.mean, errorbar="sd", marker="^", markersize=7, lw=2,
                 color=cmap_p(0), label=r"$\kappa$ @ 99%", ax=ax)

    first = True
    for col in ("kappa_mean", "kappa_p90", "kappa_p99"):
        grp = df.groupby("du_size")[col].mean()
        z = 1.0 / grp.index.values
        c = (grp.values @ z) / (z @ z)
        ax.plot(du_sizes, c / np.array(du_sizes), "--", color="k", lw=2,
                label=r"$\propto 1/|D_u|$" if first else None)
        first = False

    ax.set_xlabel(r"$|D_u| / |D|$", fontsize=24)
    ax.set_ylabel(r"$\kappa(D_u)$", fontsize=24)
    ax.legend(fontsize=24, loc="upper right", framealpha=0.5)
    ax.grid(ls="--", alpha=0.3, which="major")

    plt.savefig(os.path.join(args.out_dir, "kappa_du_sizes.pdf"), dpi=300, bbox_inches=None)
    plt.close()


if __name__ == "__main__":
    main()
