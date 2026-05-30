import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import utils

N_GRID = 80
ETAS = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99])


def run_one_seed(seed, n_class, n_u_class):
    """Return per-eta results for a single seeded (seed, z*) pair."""
    inst = utils.setup_instance(seed, n_per=n_class, du_per_class=n_u_class, lam=utils.LAM)
    zs = utils.pick_zstars_by_rho(inst["rhos"], inst["retained"])
    z_idx = zs[0]
    rho_z = inst["rhos"][z_idx]
    g_norm_z = inst["g_norms"][z_idx]

    theta_u_mem, theta_u_non = utils.models_mem_vs_non(
        inst["X"], inst["y"], inst["idx_Du"].tolist(), z_idx)

    rows = []
    for eta in ETAS:
        theta_u_dis = utils.dishonest_theta(inst["theta"], inst["Delta"], eta)
        _, _, sc, su = utils.signals_on_grid(
            inst["theta_u_hon"], theta_u_dis, theta_u_mem, theta_u_non,
            lim=utils.LIM, n_grid=N_GRID)
        loc = np.unravel_index(np.argmax(sc), sc.shape)
        rows.append((
            float(eta),
            utils.empirical_slope(sc, su, top_frac=0.10),
            utils.kappa_theory(rho_z, g_norm_z, inst["G_Du_w_norm"], eta=eta),
            float(sc[loc]),
            float(su[loc]),
        ))
    return rows, rho_z


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_seeds",   type=int, default=30)
    parser.add_argument("--n_class",   type=int, default=120)
    parser.add_argument("--n_u_class", type=int, default=8)
    parser.add_argument("--out_dir",   type=str, default="results")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    records = []
    rho_zs = []
    for s in range(args.n_seeds):
        print(f"[seed {s + 1}/{args.n_seeds}]", flush=True)
        rows, rho_z = run_one_seed(s, args.n_class, args.n_u_class)
        rho_zs.append(rho_z)
        for eta, slope_emp, slope_th, s_comp, s_cur in rows:
            records.append(dict(seed=s, eta=eta, slope_emp=slope_emp,
                                slope_th=slope_th, s_comp=s_comp, s_cur=s_cur))

    print(f"|rho_z| across seeds: mean={np.mean(np.abs(rho_zs)):.3f} "
          f"min={np.min(np.abs(rho_zs)):.3f} max={np.max(np.abs(rho_zs)):.3f}")

    df = pd.DataFrame(records).dropna(subset=["slope_emp"])
    th_line = df.groupby("eta")["slope_th"].median()

    fig, ax = utils.single_ax_fig(margin=(1.0, 1.0))
    sns.lineplot(data=df, x="eta", y="slope_emp", errorbar="sd",
                 marker="o", linewidth=2, color="#0072B2", label=r"Empirical $\kappa$", ax=ax)
    ax.plot(th_line.index, th_line.values, "k--", lw=2,
            label=r"Theoretical $\kappa = \frac{|\rho(z^*)|}{1-\eta}"
                  r"\frac{\|g_{z^*}\|}{\|G_{D_u}\|}$")
    ax.set_xlabel(r"$\eta$", fontsize=24)
    ax.set_ylabel(r"$\kappa(\eta)$", fontsize=24)
    ax.set_yscale("log")
    ax.set_xlim(0.0, 1.0)
    ax.legend(fontsize=24, loc="upper left")
    ax.grid(True, which="major", ls="--", lw=0.5, alpha=0.3)

    plt.savefig(os.path.join(args.out_dir, "kappa_eta.pdf"), dpi=300, bbox_inches=None)
    plt.close()


if __name__ == "__main__":
    main()
