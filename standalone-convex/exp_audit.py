import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
import utils
from utils import K

M_TRIALS     = 3000
T_VALUES     = [10, 50, 100, 500]
SIGMA_VALUES = np.geomspace(0.005, 0.5, 20)
ETAS_TO_TEST = [0.1, 0.5, 0.9]


def _simulate_lrt(signal_t, sigma, M, sign=1):
    """Monte Carlo LR-test statistics under H0 (noise only) and H1 (signal + noise).

    sign=+1: H1 adds signal; sign=-1: H1 subtracts signal (compliance case).
    Returns (Tstat_under_H0, Tstat_under_H1).
    """
    s = signal_t
    eps0 = np.random.normal(0, sigma, (M, len(s)))
    eps1 = np.random.normal(0, sigma, (M, len(s)))
    Tstat0 = np.sum(sign * s[None, :] * eps0, axis=1)
    Tstat1 = np.sum(sign * s[None, :] * (sign * s[None, :] + eps1), axis=1)
    return Tstat0, Tstat1


def alpha_curves(Tstat_0, Tstat_1):
    """Sweep LR threshold → arrays of (alpha_prime, alpha).

    alpha_prime = P(Tstat > c | H0) — false positive
    alpha       = P(Tstat <= c | H1) — false negative (soundness error)
    """
    all_vals = np.concatenate([Tstat_0, Tstat_1])
    cs = np.sort(np.concatenate([[all_vals.min() - 1],
                                  np.unique(all_vals),
                                  [all_vals.max() + 1]]))
    ap = np.array([(Tstat_0 > c).mean() for c in cs])
    a  = np.array([(Tstat_1 <= c).mean() for c in cs])
    return ap, a


def curiosity_advantage(Tstat_0, Tstat_1):
    """beta = TV(P_mem, P_non) = 2 * (best_classifier_accuracy - 0.5)."""
    cs = np.unique(np.concatenate([Tstat_0, Tstat_1]))
    best = 0.5
    for c in cs:
        acc = 0.5 * (Tstat_1 > c).mean() + 0.5 * (Tstat_0 <= c).mean()
        best = max(best, acc, 1 - acc)
    return 2 * (best - 0.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",      type=int, default=7)
    parser.add_argument("--n_class",   type=int, default=120)
    parser.add_argument("--n_u_class", type=int, default=8)
    parser.add_argument("--out_dir",   type=str, default="results")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    inst = utils.setup_instance(args.seed, n_per=args.n_class, du_per_class=args.n_u_class)
    X, y = inst["X"], inst["y"]
    theta, Delta = inst["theta"], inst["Delta"]
    idx_Du = inst["idx_Du"]
    G_Du_w_norm, g_norms, rhos = inst["G_Du_w_norm"], inst["g_norms"], inst["rhos"]

    z = utils.pick_zstars_by_rho(rhos, inst["retained"])[0]
    rho_z, g_n = rhos[z], g_norms[z]
    kappa_z = abs(rho_z) * g_n / G_Du_w_norm
    print(f"z*={z}, rho={rho_z:+.3f}, ||g||_w={g_n:.4f}, "
          f"||G_Du||_w={G_Du_w_norm:.4f}, kappa(eta=0)={kappa_z:.4f}")

    theta_u_mem, theta_u_non = utils.models_mem_vs_non(X, y, idx_Du.tolist(), z)

    # Scalar signals at a 120×120 grid (class-0 posterior difference)
    xs = np.linspace(-utils.LIM, utils.LIM, 120)
    XX, YY = np.meshgrid(xs, xs)
    Q = utils.feature_lift(np.stack([XX.ravel(), YY.ravel()], axis=1))
    Dq = Q.shape[1]
    cls = 0
    f_hon = utils.predict_proba(inst["theta_u_hon"], Q, Dq)[:, cls]
    f_mem = utils.predict_proba(theta_u_mem, Q, Dq)[:, cls]
    f_non = utils.predict_proba(theta_u_non, Q, Dq)[:, cls]

    np.random.seed(args.seed)

    T_max = max(T_VALUES)
    cmap_eta = plt.cm.plasma

    for eta in ETAS_TO_TEST:
        f_dis = utils.predict_proba(utils.dishonest_theta(theta, Delta, eta), Q, Dq)[:, cls]
        sig_comp_v = f_hon - f_dis   # signed compliance signal
        sig_cur_v  = f_mem - f_non   # signed curiosity signal (eta-independent)

        top_idx = np.argsort(np.abs(sig_comp_v))[::-1][:T_max]
        s_comp_top = sig_comp_v[top_idx]
        s_cur_top  = sig_cur_v[top_idx]

        pareto_pts = []
        for T in T_VALUES:
            s_c = s_comp_top[:T]
            s_u = s_cur_top[:T]
            for sigma in SIGMA_VALUES:
                Tc0, Tc1 = _simulate_lrt(s_c, sigma, M_TRIALS, sign=-1)
                ap_arr, a_arr = alpha_curves(Tc0, Tc1)
                k_best = int(np.argmin(a_arr + ap_arr))
                Tu0, Tu1 = _simulate_lrt(s_u, sigma, M_TRIALS, sign=+1)
                beta_emp = float(curiosity_advantage(Tu0, Tu1))
                pareto_pts.append((T, sigma, float(a_arr[k_best]), float(ap_arr[k_best]), beta_emp))

        pareto_pts = np.array(pareto_pts)
        kappa_eta = kappa_z / max(1.0 - eta, 1e-12)

        fig, ax = utils.single_ax_fig(margin=(1.0, 1.0))
        Ts = pareto_pts[:, 0]
        for T in T_VALUES:
            m = Ts == T
            ax.scatter(pareto_pts[m, 2] + pareto_pts[m, 3],
                       pareto_pts[m, 4],
                       s=100, alpha=0.75,
                       color=cmap_eta(T_VALUES.index(T) / max(len(T_VALUES) - 1, 1)),
                       label=f"T={T}")
        xs_th = np.linspace(0, 1, 60)
        beta_floor = np.maximum(min(kappa_eta, 1.0) * (1.0 - xs_th), 0)
        ax.plot(xs_th, beta_floor, "k--", lw=2,
                label=rf"Theoretical floor:" + "\n" +
                      rf"$\kappa (1-\alpha-\alpha^\prime)$, $\kappa={kappa_eta:.2f}$")
        ax.fill_between(xs_th, 0, beta_floor, alpha=0.10, color="black", label="Impossibility region")
        ax.set_xlabel(r"Compliance error $\alpha + \alpha^\prime$", fontsize=24)
        ax.set_ylabel(r"Curiosity advantage $\beta$", fontsize=24)
        ax.set_xlim(0.0, 1.0); ax.set_ylim(0.0, 1.0)
        ax.legend(fontsize=16, loc="lower left", framealpha=0.5)
        ax.grid(linestyle="--", alpha=0.3)

        plt.savefig(os.path.join(args.out_dir, f"pareto_{eta}.pdf"), dpi=300, bbox_inches=None)
        plt.close()


if __name__ == "__main__":
    main()
