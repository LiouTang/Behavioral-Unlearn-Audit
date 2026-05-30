import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
import utils

N_GRID = 80


def run_one_seed(seed, n_class, n_u_class, n_zstar, lam):
    """Return list of (z_idx, rho, g_norm, kappa_theory, kappa_emp)."""
    inst = utils.setup_instance(seed, n_per=n_class, du_per_class=n_u_class, lam=lam)
    X, y = inst["X"], inst["y"]
    theta, Delta = inst["theta"], inst["Delta"]
    idx_Du = inst["idx_Du"]
    G_Du_w_norm, g_norms, rhos = inst["G_Du_w_norm"], inst["g_norms"], inst["rhos"]
    retained = inst["retained"]
    theta_u_hon = inst["theta_u_hon"]
    theta_u_dis_eta0 = utils.dishonest_theta(theta, Delta, 0.0)

    # Sample z* candidates stratified by |rho| to cover the full alignment range
    retained_idx = np.where(retained)[0]
    order = np.argsort(np.abs(rhos[retained_idx]))
    n_avail = len(retained_idx)
    if n_avail <= n_zstar:
        sel = retained_idx
    else:
        sel = retained_idx[order[np.linspace(0, n_avail - 1, n_zstar).astype(int)]]

    results = []
    for z_idx in sel:
        theta_u_mem, theta_u_non = utils.models_mem_vs_non(
            X, y, idx_Du.tolist(), int(z_idx), lam)
        _, _, sc, su = utils.signals_on_grid(
            theta_u_hon, theta_u_dis_eta0, theta_u_mem, theta_u_non,
            lim=utils.LIM, n_grid=N_GRID)
        kappa_th = utils.kappa_theory(rhos[z_idx], g_norms[z_idx], G_Du_w_norm, eta=0.0)
        kappa_ls = utils.empirical_slope(sc, su, top_frac=0.10)
        results.append((int(z_idx), float(rhos[z_idx]), float(g_norms[z_idx]),
                        float(kappa_th), float(kappa_ls)))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_seeds",          type=int, default=30)
    parser.add_argument("--n_zstar_per_seed", type=int, default=30)
    parser.add_argument("--n_class",          type=int, default=120)
    parser.add_argument("--n_u_class",        type=int, default=8)
    parser.add_argument("--out_dir",          type=str, default="results")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    all_rows = []
    for s in range(args.n_seeds):
        print(f"[seed {s + 1}/{args.n_seeds}] running...", flush=True)
        rows = run_one_seed(s, args.n_class, args.n_u_class, args.n_zstar_per_seed, utils.LAM)
        for r in rows:
            all_rows.append((s,) + r)

    arr = np.array(all_rows, dtype=object)
    seeds    = arr[:, 0].astype(int)
    rhos     = arr[:, 2].astype(float)
    kappa_th = arr[:, 4].astype(float)
    kappa_ls = arr[:, 5].astype(float)

    ok = ~(np.isnan(kappa_ls) | np.isnan(kappa_th))
    seeds, rhos, kappa_th, kappa_ls = seeds[ok], rhos[ok], kappa_th[ok], kappa_ls[ok]

    print(f"\ntotal (seed, z*) pairs: {len(kappa_th)}")
    print(f"rho range: [{rhos.min():+.3f}, {rhos.max():+.3f}]")
    print(f"kappa_th range: [{kappa_th.min():.4f}, {kappa_th.max():.4f}]")
    print(f"kappa_ls range: [{kappa_ls.min():.4f}, {kappa_ls.max():.4f}]")

    _, ax = utils.single_ax_fig(margin=(0.8, 0.8), ax_size=(6.9, 6.9))

    axis_max = -1
    geometric = np.abs(rhos) > 0.05
    cmap = plt.cm.plasma
    for s in range(args.n_seeds):
        m = (seeds == s) & geometric
        if m.sum() == 0:
            continue
        ax.scatter(kappa_th[m], kappa_ls[m], s=50, alpha=0.6,
                   color=cmap(s / max(args.n_seeds - 1, 1)),
                   label=f"seed {s}" if s % 10 == 0 else None)
        if kappa_ls[m].max() > axis_max:
            axis_max = kappa_ls[m].max()
    near_zero = ~geometric
    if near_zero.sum() > 0:
        ax.scatter(kappa_th[near_zero], kappa_ls[near_zero], s=25, alpha=0.5,
                   c="lightgray") #, label=r"$|\rho| \leq 0.05$"

    ax.plot([0, axis_max], [0, axis_max], "k:")

    if geometric.sum() > 2:
        m_fit = (np.sum(kappa_th[geometric] * kappa_ls[geometric])
                 / np.sum(kappa_th[geometric] ** 2))
        ss_res = np.sum((kappa_ls[geometric] - m_fit * kappa_th[geometric]) ** 2)
        ss_tot = np.sum((kappa_ls[geometric] - kappa_ls[geometric].mean()) ** 2)
        r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
        ax.plot([0, axis_max], [0, m_fit * axis_max], "crimson", lw=2.5,
                label=rf"fit: $\kappa_{{emp}} \approx {m_fit:.2f}\kappa_{{th}}$ ($R^2={r2:.3f}$)")

    ax.set_xlabel(r"Theoretical $\kappa(z^*)$", fontsize=24)
    ax.set_ylabel(r"Empirical $\kappa(z^*)$", fontsize=24)
    ax.legend(fontsize=24, loc="upper left", framealpha=0.5)
    ax.grid(True, ls="--", lw=0.5, alpha=0.3)
    ax.set_xlim(0, axis_max * 1.1)
    ax.set_ylim(0, axis_max * 1.1)

    plt.savefig(os.path.join(args.out_dir, "kappa_seeds.pdf"), dpi=300, bbox_inches=None)
    plt.close()


if __name__ == "__main__":
    main()
