"""Measure query-conditioned geometry and its first-order convex proxy.

The experiment varies query protocols, query counts, update scales, and retained
targets. It records exact and proxy geometry, plots their agreement, illustrates
query conditioning, and summarizes realized approximation errors.

Outputs:
  geometry_results.csv
  gamma_vs_lambda.pdf
  query_conditioning_example.pdf
  proxy_error.pdf
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

import utils


DEFAULT_ETAS = [0.0, 0.5, 0.9, 1.1, 1.5]
DEFAULT_T = [10, 50, 200]
DEFAULT_PROTOCOLS = ["random", "compliance", "q1", "q2", "q3", "q4"]


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n_class", type=int, default=120)
    ap.add_argument("--n_u_class", type=int, default=8)
    ap.add_argument("--n_grid", type=int, default=81)
    ap.add_argument("--response_class", type=int, default=0)
    ap.add_argument("--etas", type=parse_float_list, default=DEFAULT_ETAS)
    ap.add_argument("--T", type=parse_int_list, default=DEFAULT_T)
    ap.add_argument("--protocols", type=lambda s: [x.strip() for x in s.split(",")], default=DEFAULT_PROTOCOLS)
    ap.add_argument("--out_dir", type=str, default="results_geometry")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    inst = utils.setup_instance(args.seed, args.n_class, args.n_u_class)
    X2q, Xq_all = utils.make_query_grid(n_grid=args.n_grid)
    z_indices = utils.pick_zstars_by_rho(inst["rhos"], inst["retained"])
    z_roles = ["positive-rho", "near-zero-rho", "negative-rho"]

    # Membership endpoints depend on z*, but not on eta or Q.
    curiosity = {}
    for role, z_idx in zip(z_roles, z_indices):
        curiosity[z_idx] = utils.membership_world_endpoints(
            inst["X"], inst["y"], inst["idx_Du"], z_idx
        )
        print(f"{role}: z*={z_idx}, rho={inst['rhos'][z_idx]:+.4f}")

    rows = []
    rng_master = np.random.default_rng(args.seed + 99173)

    for eta in args.etas:
        if abs(eta - 1.0) < 1e-12:
            print("Skipping eta=1: compliance vector is identically zero.")
            continue
        theta_u_dis = utils.dishonest_theta(inst["theta"], inst["Delta"], eta)

        for T in args.T:
            for protocol in args.protocols:
                # Reuse the same target-independent queries for all z*.
                q_rng = np.random.default_rng(rng_master.integers(0, 2**32 - 1))
                idx_q = utils.select_query_indices(
                    protocol, T, X2q, Xq_all,
                    inst["theta_u_hon"], theta_u_dis,
                    args.response_class, q_rng,
                )
                Xq = Xq_all[idx_q]

                for role, z_idx in zip(z_roles, z_indices):
                    theta_u_mem, theta_u_non = curiosity[z_idx]
                    exact = utils.exact_geometry(
                        inst["theta_u_hon"], theta_u_dis,
                        theta_u_mem, theta_u_non,
                        Xq, args.response_class,
                    )
                    proxy = utils.convex_proxy_geometry(
                        inst["theta"], inst["Delta"], inst["G_Du"], inst["Hinv"],
                        inst["X"], inst["y"], z_idx, eta, Xq, exact,
                        args.response_class,
                    )
                    rows.append(utils.row_from_geometry(
                        seed=args.seed, z_role=role, z_idx=z_idx, eta=eta,
                        protocol=protocol, T=T, response_class=args.response_class,
                        inst=inst, exact=exact, proxy=proxy,
                    ))

    csv_path = os.path.join(args.out_dir, "geometry_results.csv")
    write_csv(csv_path, rows)

    # Plot exact alignment against its convex proxy.
    fig, ax = utils.single_ax_fig()
    protocols = list(dict.fromkeys(r["query_protocol"] for r in rows))
    for protocol in protocols:
        rr = [r for r in rows if r["query_protocol"] == protocol and np.isfinite(r["gamma"]) and np.isfinite(r["lambda_conv"])]
        if rr:
            ax.scatter([r["lambda_conv"] for r in rr], [r["gamma"] for r in rr], s=24, alpha=0.65, label=protocol)
    finite = [v for r in rows for v in (r["lambda_conv"], r["gamma"]) if np.isfinite(v)]
    if finite:
        mx = max(finite)
        ax.plot([0, mx], [0, mx], "--", linewidth=1.2, label="y=x")
    ax.set_xlabel(r"Convex proxy $\lambda_Q^{\mathrm{conv}}$", fontsize=24)
    ax.set_ylabel(r"Exact behavioral $\gamma_Q$", fontsize=24)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=24, ncol=2)
    fig.savefig(
        os.path.join(args.out_dir, "gamma_vs_lambda.pdf"),
        dpi=300,
        bbox_inches=None,
    )
    plt.close(fig)

    # Compare query protocols for a fixed target, eta, and query count.
    eta0 = min(args.etas, key=lambda x: abs(x - 0.5) if abs(x - 1.0) > 1e-12 else 99)
    T0 = args.T[len(args.T) // 2]
    role0 = "positive-rho"
    rr = [r for r in rows if r["eta"] == eta0 and r["T"] == T0 and r["z_role"] == role0]
    if rr:
        labels = [r["query_protocol"] for r in rr]
        x = np.arange(len(rr))
        width = 0.34
        fig, ax = utils.single_ax_fig()
        ax.bar(x - width / 2, [r["gamma"] for r in rr], width, label=r"exact $\gamma_Q$")
        ax.bar(x + width / 2, [r["lambda_conv"] for r in rr], width, label=r"proxy $\lambda_Q^{conv}$")
        kappa = rr[0]["kappa"]
        ax.axhline(kappa, linestyle="--", linewidth=1.3, label=r"query-independent $\kappa$")
        ax.set_xticks(x, labels, rotation=25, ha="right")
        ax.set_ylabel("alignment coefficient", fontsize=24)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=24)
        fig.savefig(
            os.path.join(args.out_dir, "query_conditioning_example.pdf"),
            dpi=300,
            bbox_inches=None,
        )
        plt.close(fig)

    # Summarize realized first-order errors across eta.
    fig, ax = utils.single_ax_fig()
    by_eta = defaultdict(lambda: [[], []])
    for r in rows:
        by_eta[r["eta"]][0].append(r["rel_eps_c"])
        by_eta[r["eta"]][1].append(r["rel_eps_p"])
    etas = sorted(by_eta)
    ax.plot(etas, [np.nanmedian(by_eta[e][0]) for e in etas], marker="o", label=r"median $\epsilon_c/\|c_Q\|$")
    ax.plot(etas, [np.nanmedian(by_eta[e][1]) for e in etas], marker="s", label=r"median $\epsilon_p/\|p_Q\|$")
    ax.set_xlabel(r"honesty scale $\eta$", fontsize=24)
    ax.set_ylabel("realized relative proxy error", fontsize=24)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=24)
    fig.savefig(
        os.path.join(args.out_dir, "proxy_error.pdf"),
        dpi=300,
        bbox_inches=None,
    )
    plt.close(fig)

    max_decomp = max(abs(r["decomposition_residual"]) for r in rows if np.isfinite(r["decomposition_residual"]))
    max_lam_res = max(abs(r["lambda_formula_residual"]) for r in rows if np.isfinite(r["lambda_formula_residual"]))
    print(f"Wrote {len(rows)} rows to {csv_path}")
    print(f"max decomposition residual: {max_decomp:.3e}")
    print(f"max lambda formula residual: {max_lam_res:.3e}")


if __name__ == "__main__":
    main()
