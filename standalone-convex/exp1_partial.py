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
    X2, y, X = inst["X2"], inst["y"], inst["X"]
    theta, Delta = inst["theta"], inst["Delta"]
    idx_Du = inst["idx_Du"]
    G_Du_w_norm, g_norms, rhos = inst["G_Du_w_norm"], inst["g_norms"], inst["rhos"]

    zs = utils.pick_zstars_by_rho(rhos, inst["retained"])
    z = zs[0]  # pick single z* with highest positive |rho|
    rho_z = rhos[z]
    eta = 0.0

    theta_u_mem, theta_u_non = utils.models_mem_vs_non(X, y, idx_Du.tolist(), z)
    theta_u_dis = utils.dishonest_theta(theta, Delta, eta)
    XX, YY, s_comp, s_cur = utils.signals_on_grid(
        inst["theta_u_hon"], theta_u_dis, theta_u_mem, theta_u_non,
        lim=utils.LIM, n_grid=120)

    # --- heatmap panel ---
    fig1, ax = utils.single_ax_fig(margin=(0.5, 0.5), ax_size=(7.4, 7.4))
    utils.plot_heatmap(ax, XX, YY, s_comp, s_cur, X2, idx_Du, z, lim=utils.LIM)
    plt.savefig(os.path.join(args.out_dir, "sigs_heatmap_select.pdf"), dpi=300, bbox_inches=None)
    plt.close(fig1)

    # --- scatter panel ---
    fig2, ax = utils.single_ax_fig(margin=(0.8, 0.8), ax_size=(6.9, 6.9))
    utils.plot_signal_scatter(ax, s_comp, s_cur, rho_z, g_norms[z], G_Du_w_norm, eta)
    ax.set_ylabel(r"$\mathsf{Sig}^{\mathsf{Cur}}$", fontsize=24)
    ax.set_xlabel(r"$\mathsf{Sig}^{\mathsf{Comp}}_{\eta}$", fontsize=24)
    plt.savefig(os.path.join(args.out_dir, "sigs_kappa_select.pdf"), dpi=300, bbox_inches=None)
    plt.close(fig2)


if __name__ == "__main__":
    main()
