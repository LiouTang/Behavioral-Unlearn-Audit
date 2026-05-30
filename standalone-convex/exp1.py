import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import Divider, Size

import utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",      type=int,   default=7)
    parser.add_argument("--n_class",   type=int,   default=120)
    parser.add_argument("--n_u_class", type=int,   default=8)
    parser.add_argument("--out_dir",   type=str,   default="figs")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    inst = utils.setup_instance(args.seed, n_per=args.n_class, du_per_class=args.n_u_class)
    X2, y, X = inst["X2"], inst["y"], inst["X"]
    theta, Delta = inst["theta"], inst["Delta"]
    idx_Du = inst["idx_Du"]
    G_Du_w_norm, g_norms, rhos = inst["G_Du_w_norm"], inst["g_norms"], inst["rhos"]

    zs = utils.pick_zstars_by_rho(rhos, inst["retained"])
    eta_vals = [0.0, 0.5, 0.9]

    # Pre-compute signals for all (z*, eta) pairs; track global vmax for shared colorbar scale
    vmax_comp = 0.0
    cache = {}
    for z_idx in zs:
        theta_u_mem, theta_u_non = utils.models_mem_vs_non(X, y, idx_Du.tolist(), z_idx)
        for eta in eta_vals:
            theta_u_dis = utils.dishonest_theta(theta, Delta, eta)
            XX, YY, s_comp, s_cur = utils.signals_on_grid(
                inst["theta_u_hon"], theta_u_dis, theta_u_mem, theta_u_non,
                lim=utils.LIM, n_grid=120)
            cache[(z_idx, eta)] = (XX, YY, s_comp, s_cur)
            vmax_comp = max(vmax_comp, s_comp.max())

    # Divider column / row index lists for a 3×3 grid
    ax_nx     = [1, 3, 5]
    ax_ny_top = [5, 3, 1]  # top row → ny=5, bottom row → ny=1

    # -------------------------------------------------------------------------
    # Figure 1: heatmap + contour
    # -------------------------------------------------------------------------
    ax_w, ax_h = 5, 5
    hgap, vgap = 0.5, 0.5
    cbar_pad, cbar_w = 0.08, 0.20
    left_m, right_m, bottom_m, top_m = 0.80, 1.00, 0.45, 0.45

    h1 = [Size.Fixed(left_m),
          Size.Fixed(ax_w), Size.Fixed(hgap),
          Size.Fixed(ax_w), Size.Fixed(hgap),
          Size.Fixed(ax_w), Size.Fixed(cbar_pad), Size.Fixed(cbar_w),
          Size.Fixed(right_m)]
    v1 = [Size.Fixed(bottom_m),
          Size.Fixed(ax_h), Size.Fixed(vgap),
          Size.Fixed(ax_h), Size.Fixed(vgap),
          Size.Fixed(ax_h), Size.Fixed(top_m)]

    fig1 = plt.figure(figsize=(sum(s.fixed_size for s in h1), sum(s.fixed_size for s in v1)))
    div1 = Divider(fig1, (0, 0, 1, 1), h1, v1, aspect=False)

    axes1 = np.empty((len(zs), len(eta_vals)), dtype=object)
    for i, ny in enumerate(ax_ny_top):
        for j, nx in enumerate(ax_nx):
            axes1[i, j] = fig1.add_axes(div1.get_position(),
                                         axes_locator=div1.new_locator(nx=nx, ny=ny))
    caxes1 = [fig1.add_axes(div1.get_position(),
                              axes_locator=div1.new_locator(nx=7, ny=ny))
              for ny in ax_ny_top]

    row_ims = [None] * len(zs)
    for i, z_idx in enumerate(zs):
        rho_z = rhos[z_idx]
        for c, eta in enumerate(eta_vals):
            ax = axes1[i, c]
            XX, YY, s_comp, s_cur = cache[(z_idx, eta)]
            row_ims[i] = utils.plot_heatmap(ax, XX, YY, s_comp, s_cur, X2, idx_Du, z_idx,
                                             lim=utils.LIM, vmax=vmax_comp)
            if i == 0:
                ax.set_title(f"$\\eta={eta:.2f}$", fontsize=20)
            if c == 0:
                ax.set_ylabel(rf"$\rho(z^*)={rho_z:+.2f}$", fontsize=20)
            if c == 2:
                ax.legend(loc="upper right", fontsize=18, framealpha=0.5)
        cb = fig1.colorbar(row_ims[i], cax=caxes1[i])
        if i == 1:
            cb.set_label(r"$\mathsf{Sig}^{\mathsf{Comp}}_{\eta}$", fontsize=20)

    plt.savefig(os.path.join(args.out_dir, "sigs_heatmap.pdf"), dpi=300, bbox_inches=None)
    plt.close(fig1)

    # -------------------------------------------------------------------------
    # Figure 2: Sig^Cur vs Sig^Comp scatter with kappa line
    # -------------------------------------------------------------------------
    h2 = [Size.Fixed(0.85),
          Size.Fixed(ax_w), Size.Fixed(hgap),
          Size.Fixed(ax_w), Size.Fixed(hgap),
          Size.Fixed(ax_w), Size.Fixed(0.50)]
    v2 = [Size.Fixed(0.65),
          Size.Fixed(ax_h), Size.Fixed(vgap),
          Size.Fixed(ax_h), Size.Fixed(vgap),
          Size.Fixed(ax_h), Size.Fixed(0.45)]

    fig2 = plt.figure(figsize=(sum(s.fixed_size for s in h2), sum(s.fixed_size for s in v2)))
    div2 = Divider(fig2, (0, 0, 1, 1), h2, v2, aspect=False)

    axes2 = np.empty((len(zs), len(eta_vals)), dtype=object)
    for i, ny in enumerate(ax_ny_top):
        for j, nx in enumerate(ax_nx):
            axes2[i, j] = fig2.add_axes(div2.get_position(),
                                         axes_locator=div2.new_locator(nx=nx, ny=ny))

    for i, z_idx in enumerate(zs):
        rho_z = rhos[z_idx]
        for c, eta in enumerate(eta_vals):
            ax = axes2[i, c]
            _, _, s_comp, s_cur = cache[(z_idx, eta)]
            utils.plot_signal_scatter(ax, s_comp, s_cur, rho_z, g_norms[z_idx], G_Du_w_norm, eta)
            ax = axes2[i, c]
            if i == 0:
                ax.set_title(f"$\\eta={eta:.2f}$", fontsize=20)
            if c == 0:
                ax.set_ylabel(rf"$\rho(z^*)={rho_z:+.2f}$", fontsize=20)
            if i == 1 and c == 2:
                ax.text(1.01, 0.5, r"$\mathsf{Sig}^{\mathsf{Cur}}$",
                        transform=ax.transAxes, fontsize=20, rotation=90, va="center", ha="left")
            if i == len(zs) - 1 and c == 1:
                ax.set_xlabel(r"$\mathsf{Sig}^{\mathsf{Comp}}_{\eta}$", fontsize=20)

    plt.savefig(os.path.join(args.out_dir, "sigs_kappa.pdf"), dpi=300, bbox_inches=None)
    plt.close(fig2)


if __name__ == "__main__":
    main()
