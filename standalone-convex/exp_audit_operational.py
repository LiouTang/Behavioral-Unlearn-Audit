"""Run Monte Carlo compliance and membership tests for the convex setting.

The experiment samples noisy transcripts under four endpoint worlds:

  compliance: honest vs dishonest endpoint,
  curiosity:  z* member vs non-member endpoint.

For each (eta, T, sigma), a plug-in equal-covariance Gaussian verifier/MIA is
fit from independent calibration transcripts and evaluated on independent
held-out transcripts.  The classifier never receives the exact c_Q or p_Q
vectors.  We report held-out estimates of

  alpha       = P[predict honest | dishonest],
  alpha_prime = P[predict dishonest | honest],
  e_hat       = alpha + alpha_prime,
  beta_MIA    = 2 * attack_accuracy - 1 = 1 - (FNR + FPR).

Exact endpoint geometry supplies analytic baselines and geometric lower bounds
for comparison with the held-out measurements.

Outputs:
  operational_trials.csv
  operational_summary.csv
  pareto_*.pdf
  operational_attack_vs_exact.pdf
  operational_bound_confirmation.pdf
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ndtr, ndtri

import utils


DEFAULT_ETAS = [0.1, 0.5, 0.9]
DEFAULT_T = [10, 50, 100, 500]
DEFAULT_SIGMAS = np.geomspace(0.003, 2.0, 18).tolist()


def parse_float_list(s):
    return [float(x) for x in s.split(",") if x.strip()]


def parse_int_list(s):
    return [int(x) for x in s.split(",") if x.strip()]


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _direct_trial_batch(mu1, mu0, sigmas, n_cal, n_test, rng):
    """Fit/evaluate plug-in LRTs for all sigmas using direct T-D trials.

    We draw actual standard-normal calibration and held-out transcript noise once
    and rescale it for each sigma (common random numbers).  For every sigma the
    marginal samples are exactly the requested N(mu, sigma^2 I) transcripts.
    The batching only avoids regenerating the same T-D noise arrays in Python.

    Returns arrays (fnr, fpr) for class 1 and class 0 respectively.
    """
    mu1 = np.asarray(mu1, dtype=float)
    mu0 = np.asarray(mu0, dtype=float)
    sigmas = np.asarray(sigmas, dtype=float)
    T = len(mu1)

    # Full T-dimensional calibration trials; only their empirical means are
    # retained when fitting the plug-in equal-covariance Gaussian LRT.
    Z1_cal = rng.standard_normal((n_cal, T))
    Z0_cal = rng.standard_normal((n_cal, T))
    z1_bar = Z1_cal.mean(axis=0)
    z0_bar = Z0_cal.mean(axis=0)

    # One fitted direction per sigma, columns shape T x S.
    M1 = mu1[:, None] + z1_bar[:, None] * sigmas[None, :]
    M0 = mu0[:, None] + z0_bar[:, None] * sigmas[None, :]
    W = M1 - M0
    B = 0.5 * (np.sum(M1 * M1, axis=0) - np.sum(M0 * M0, axis=0))

    # Independent full held-out transcripts.  Compute all sigma-specific LRT
    # scores in two BLAS matrix multiplies.
    Z1 = rng.standard_normal((n_test, T))
    Z0 = rng.standard_normal((n_test, T))
    base1 = mu1 @ W - B
    base0 = mu0 @ W - B
    S1 = base1[None, :] + (Z1 @ W) * sigmas[None, :]
    S0 = base0[None, :] + (Z0 @ W) * sigmas[None, :]
    fnr = np.mean(S1 <= 0.0, axis=0)
    fpr = np.mean(S0 > 0.0, axis=0)
    return fnr, fpr


def theorem_floor_from_e(e, gamma):
    """Compute the alignment lower bound for an error value and fixed gamma_Q."""
    e = float(np.clip(e, 0.0, 1.0))
    if not np.isfinite(gamma):
        return np.nan
    if e <= 0.0:
        return 1.0 if gamma > 0 else 0.0
    if e >= 1.0:
        return 0.0
    qe = float(ndtri(1.0 - e / 2.0))
    return float(2.0 * ndtr(gamma * qe) - 1.0)


def _aggregate(rows):
    groups = defaultdict(list)
    for r in rows:
        key = (r["eta"], r["T"], r["sigma"])
        groups[key].append(r)
    out = []
    for (eta, T, sigma), rr in sorted(groups.items()):
        def mean(k):
            return float(np.mean([x[k] for x in rr]))
        def sd(k):
            return float(np.std([x[k] for x in rr], ddof=1)) if len(rr) > 1 else 0.0
        out.append(dict(
            eta=eta, T=T, sigma=sigma, repeats=len(rr),
            gamma=mean("gamma"), d_comp=mean("d_comp"), d_cur=mean("d_cur"),
            alpha=mean("alpha"), alpha_prime=mean("alpha_prime"),
            e_emp=mean("e_emp"), e_emp_sd=sd("e_emp"),
            beta_mia=mean("beta_mia"), beta_mia_sd=sd("beta_mia"),
            e_exact=mean("e_exact"), beta_exact=mean("beta_exact"),
            beta_bound_exact_e=mean("beta_bound_exact_e"),
            beta_bound_emp_e=mean("beta_bound_emp_e"),
        ))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n_class", type=int, default=120)
    ap.add_argument("--n_u_class", type=int, default=8)
    ap.add_argument("--n_grid", type=int, default=81)
    ap.add_argument("--response_class", type=int, default=0)
    ap.add_argument("--eta", type=parse_float_list, default=DEFAULT_ETAS)
    ap.add_argument("--T", type=parse_int_list, default=DEFAULT_T)
    ap.add_argument("--sigma", type=parse_float_list, default=DEFAULT_SIGMAS)
    ap.add_argument("--protocol", type=str, default="compliance")
    ap.add_argument("--z_role", choices=["positive-rho", "near-zero-rho", "negative-rho"], default="positive-rho")
    ap.add_argument("--n_cal", type=int, default=1500, help="calibration transcripts per world")
    ap.add_argument("--n_test", type=int, default=6000, help="held-out transcripts per world")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out_dir", type=str, default="results_operational")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    inst = utils.setup_instance(args.seed, args.n_class, args.n_u_class)
    X2q, Xq_all = utils.make_query_grid(n_grid=args.n_grid)
    z_all = utils.pick_zstars_by_rho(inst["rhos"], inst["retained"])
    role_to_z = dict(zip(["positive-rho", "near-zero-rho", "negative-rho"], z_all))
    z_idx = role_to_z[args.z_role]
    theta_mem, theta_non = utils.membership_world_endpoints(
        inst["X"], inst["y"], inst["idx_Du"], z_idx
    )

    rows = []
    geom = {}
    # Q is fixed before observation-noise trials and never uses z* in selection.
    for eta in args.eta:
        if abs(eta - 1.0) < 1e-12:
            continue
        theta_dis = utils.dishonest_theta(inst["theta"], inst["Delta"], eta)
        for T in args.T:
            q_rng = np.random.default_rng(args.seed + 100003 * T + int(round(eta * 10000)))
            idx_q = utils.select_query_indices(
                args.protocol, T, X2q, Xq_all,
                inst["theta_u_hon"], theta_dis, args.response_class, q_rng,
            )
            Xq = Xq_all[idx_q]
            ex = utils.exact_geometry(
                inst["theta_u_hon"], theta_dis, theta_mem, theta_non,
                Xq, args.response_class,
            )
            geom[(eta, T)] = ex

            # Actual transcript means.  These are not passed to the fitted
            # verifier/MIA except as centers from which independent calibration
            # and held-out observations are generated.
            mu_h = utils.response(inst["theta_u_hon"], Xq, args.response_class)
            mu_d = utils.response(theta_dis, Xq, args.response_class)
            mu_m = utils.response(theta_mem, Xq, args.response_class)
            mu_n = utils.response(theta_non, Xq, args.response_class)

            # Compute analytic baselines from the same endpoint geometry.
            proxy = utils.convex_proxy_geometry(
                inst["theta"], inst["Delta"], inst["G_Du"], inst["Hinv"],
                inst["X"], inst["y"], z_idx, eta, Xq, ex, args.response_class,
            )
            exact_by_sigma = [utils.theorem_values(ex, proxy, sigma) for sigma in args.sigma]

            for rep in range(args.repeats):
                base = args.seed * 1000003 + int(round(eta * 1000)) * 10007 + T * 101 + rep
                rng_c = np.random.default_rng(base + 11)
                rng_p = np.random.default_rng(base + 17)

                # Evaluate full-transcript tests for all noise scales.
                alpha_prime_arr, alpha_arr = _direct_trial_batch(
                    mu_h, mu_d, args.sigma, args.n_cal, args.n_test, rng_c
                )
                fn_arr, fp_arr = _direct_trial_batch(
                    mu_m, mu_n, args.sigma, args.n_cal, args.n_test, rng_p
                )

                for sigma_idx, sigma in enumerate(args.sigma):
                    exact_vals = exact_by_sigma[sigma_idx]
                    alpha_prime = float(alpha_prime_arr[sigma_idx])
                    alpha = float(alpha_arr[sigma_idx])
                    e_emp = alpha + alpha_prime
                    fn_mia = float(fn_arr[sigma_idx])
                    fp_mia = float(fp_arr[sigma_idx])
                    beta_raw = float(1.0 - fn_mia - fp_mia)
                    beta_mia = max(beta_raw, 0.0)

                    e_se = float(np.sqrt(
                        alpha * (1.0 - alpha) / args.n_test
                        + alpha_prime * (1.0 - alpha_prime) / args.n_test
                    ))
                    beta_se = float(np.sqrt(
                        fn_mia * (1.0 - fn_mia) / args.n_test
                        + fp_mia * (1.0 - fp_mia) / args.n_test
                    ))

                    rows.append(dict(
                        seed=args.seed, repeat=rep, z_role=args.z_role, z_idx=z_idx,
                        z_rho=float(inst["rhos"][z_idx]), eta=float(eta), T=int(T),
                        sigma=float(sigma), query_protocol=args.protocol,
                        n_cal=args.n_cal, n_test=args.n_test,
                        gamma=float(ex.gamma), d_comp=float(ex.d_comp), d_cur=float(ex.d_cur),
                        p_perp_norm=float(ex.p_perp_norm),
                        alpha=alpha, alpha_prime=alpha_prime, e_emp=float(e_emp), e_emp_se=e_se,
                        mia_fnr=fn_mia, mia_fpr=fp_mia,
                        beta_mia_raw=beta_raw, beta_mia=beta_mia, beta_mia_se=beta_se,
                        e_exact=float(exact_vals["e"]), beta_exact=float(exact_vals["beta"]),
                        beta_bound_exact_e=float(exact_vals["beta_main_bound"]),
                        beta_bound_emp_e=theorem_floor_from_e(e_emp, ex.gamma),
                    ))

    trial_path = os.path.join(args.out_dir, "operational_trials.csv")
    write_csv(trial_path, rows)
    summary = _aggregate(rows)
    write_csv(os.path.join(args.out_dir, "operational_summary.csv"), summary)

    # Plot empirical Pareto points with a common lower bound for each eta.
    cmap = plt.cm.plasma
    t_colors = {T: cmap(i / max(len(args.T) - 1, 1)) for i, T in enumerate(args.T)}
    e_grid = np.linspace(1e-5, 0.9999, 400)

    for eta in args.eta:
        fig, ax = utils.single_ax_fig()
        erows = [r for r in rows if r["eta"] == eta]
        for T in args.T:
            rr = [r for r in erows if r["T"] == T]
            ax.scatter(
                [r["e_emp"] for r in rr], [r["beta_mia"] for r in rr],
                s=80, alpha=0.55, color=t_colors[T], label=f"T={T}", rasterized=True,
            )
        gamma_min = min(geom[(eta, T)].gamma for T in args.T)
        floor = np.array([theorem_floor_from_e(e, gamma_min) for e in e_grid])
        ax.plot(e_grid, floor, "k--", linewidth=1.4,
                label=rf"common lower bound, $\gamma_{{\min}}={gamma_min:.2f}$")
        ax.fill_between(e_grid, 0.0, floor, color="0.8", alpha=0.35, label="excluded region")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel(
            r"Compliance error $\alpha+\alpha^\prime$ ",
            fontsize=24,
        )
        ax.set_ylabel(
            r"Curiosity advantage $\beta_{\rm MIA}$",
            fontsize=24,
        )
        ax.grid(alpha=0.25, linestyle="--")
        ax.legend(fontsize=16, loc="lower left")
        stem = f"pareto_{eta:g}"
        fig.savefig(
            os.path.join(args.out_dir, f"{stem}.pdf"),
            dpi=300,
            bbox_inches=None,
        )
        plt.close(fig)

    # Compare the operational membership test with exact transcript TV.
    fig, ax = utils.single_ax_fig()
    for T in args.T:
        rr = [r for r in rows if r["T"] == T]
        ax.scatter([r["beta_exact"] for r in rr], [r["beta_mia"] for r in rr],
                   s=80, alpha=0.45, color=t_colors[T], label=f"T={T}", rasterized=True)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Bayes optimum")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel(r"exact transcript distinguishability $\beta$", fontsize=24)
    ax.set_ylabel(r"held-out operational $\widehat\beta_{\rm MIA}$", fontsize=24)
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(fontsize=16)
    fig.savefig(
        os.path.join(args.out_dir, "operational_attack_vs_exact.pdf"),
        dpi=300,
        bbox_inches=None,
    )
    plt.close(fig)

    # Compare operational measurements with the geometric lower bound.
    fig, ax = utils.single_ax_fig()
    for T in args.T:
        rr = [r for r in rows if r["T"] == T]
        ax.scatter([r["beta_bound_emp_e"] for r in rr], [r["beta_mia"] for r in rr],
                   s=80, alpha=0.45, color=t_colors[T], label=f"T={T}", rasterized=True)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="lower-bound boundary")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel(
        r"lower bound from held-out $\widehat e$ and exact $\gamma_Q$",
        fontsize=24,
    )
    ax.set_ylabel(r"held-out operational $\widehat\beta_{\rm MIA}$", fontsize=24)
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(fontsize=16)
    fig.savefig(
        os.path.join(args.out_dir, "operational_bound_confirmation.pdf"),
        dpi=300,
        bbox_inches=None,
    )
    plt.close(fig)

    beta_mae = float(np.mean([abs(r["beta_mia"] - r["beta_exact"]) for r in rows]))
    e_mae = float(np.mean([abs(r["e_emp"] - r["e_exact"]) for r in rows]))
    below = [r for r in rows if r["beta_mia"] + 1.96 * r["beta_mia_se"] < r["beta_bound_emp_e"]]
    print(f"z*={z_idx}, rho={inst['rhos'][z_idx]:+.4f}; wrote {len(rows)} direct trial measurements")
    print(f"mean |e_emp-e_exact| = {e_mae:.4f}")
    print(f"mean |beta_MIA-beta_exact| = {beta_mae:.4f}")
    print(f"points whose beta_MIA 95% upper sampling CI is below bound(empirical e): {len(below)}/{len(rows)}")
    print(f"outputs: {args.out_dir}")


if __name__ == "__main__":
    main()
