"""Measure how behavioral alignment changes across controlled parameter sweeps.

The experiment varies the update scale, retained target, and deletion-set size.
It compares exact alignment with its convex proxy and reuses fitted non-member
models across deletion-set sizes to avoid redundant computation.
"""

from __future__ import annotations

import argparse, csv, os
import matplotlib.pyplot as plt
import numpy as np
import utils

ETAS = np.array(
    [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.99]
)


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def instance_with_du(base, idx_du):
    idx_du = np.asarray(idx_du, dtype=int)
    Delta, G, H, Hinv = utils.influence_unlearn_delta(base["theta"], base["X"], base["y"], idx_du, float(base["lam"]))
    mask = np.zeros(len(base["y"]), bool)
    mask[idx_du] = True
    GG = max(float(G @ Hinv @ G), 0.0)
    rhos = np.zeros(len(base["y"]))
    gnorm = np.zeros(len(base["y"]))
    for i in range(len(base["y"])):
        gi = utils.per_sample_grad_flat(base["theta"], base["X"][i], int(base["y"][i]))
        gg = max(float(gi @ Hinv @ gi), 0.0)
        gnorm[i] = np.sqrt(gg)
        den = np.sqrt(gg * GG)
        rhos[i] = float(gi @ Hinv @ G) / den if den > 1e-30 else 0.0
    return dict(
        rng=base["rng"],
        X2=base["X2"],
        y=base["y"],
        X=base["X"],
        theta=base["theta"],
        lam=base["lam"],
        idx_Du=idx_du,
        Du_mask=mask,
        retained=~mask,
        Delta=Delta,
        G_Du=G,
        H=H,
        Hinv=Hinv,
        theta_u_hon=base["theta"] + Delta,
        rhos=rhos,
        g_hinv_norms=gnorm,
        G_Du_hinv_norm=np.sqrt(GG),
    )


def precompute_non_world(base, z):
    """Fit the nonmember training world once and cache its Hessian/sample gradients."""
    keep = np.ones(len(base["X"]), bool)
    keep[z] = False
    Xn, yn = base["X"][keep], base["y"][keep]
    th = utils.fit(Xn, yn, lam=float(base["lam"]))
    H = utils._hessian(th, Xn, float(base["lam"]))
    Hinv = np.linalg.inv(H)
    grads = np.vstack([utils.per_sample_grad_flat(th, Xn[i], int(yn[i])) for i in range(len(yn))])
    return th, Hinv, grads


def non_endpoint_from_cache(cache, idx_du, z):
    th, Hinv, grads = cache
    iu = np.array([i if i < z else i - 1 for i in idx_du], dtype=int)
    delta = (Hinv @ grads[iu].sum(axis=0)) / len(grads)
    return th + delta


def fixed_compliance_Q(inst, eta, T, X2q, Xq_all, response_class, seed):
    dis = utils.dishonest_theta(inst["theta"], inst["Delta"], eta)
    iq = utils.select_query_indices(
        "compliance", T, X2q, Xq_all, inst["theta_u_hon"], dis, response_class,
        np.random.default_rng(seed),
    )
    return Xq_all[iq]


def panel_eta(seed, n_class, du_per_class, T, n_grid, response_class, eta_ref):
    inst = utils.setup_instance(seed, n_class, du_per_class)
    z = utils.pick_zstars_by_rho(inst["rhos"], inst["retained"])[0]
    mem, non = utils.membership_world_endpoints(inst["X"], inst["y"], inst["idx_Du"], z, float(inst["lam"]))
    X2q, Xall = utils.make_query_grid(n_grid=n_grid)
    Xq = fixed_compliance_Q(inst, eta_ref, T, X2q, Xall, response_class, seed + 111)
    rows = []
    for eta in ETAS:
        dis = utils.dishonest_theta(inst["theta"], inst["Delta"], float(eta))
        ex = utils.exact_geometry(
            inst["theta_u_hon"], dis, mem, non, Xq, response_class
        )
        pr = utils.convex_proxy_geometry(
            inst["theta"], inst["Delta"], inst["G_Du"], inst["Hinv"], inst["X"], inst["y"],
            z, float(eta), Xq, ex, response_class,
        )
        rows.append(
            dict(
                seed=seed,
                eta=float(eta),
                z_idx=z,
                rho=float(inst["rhos"][z]),
                gamma=float(ex.gamma),
                lambda_conv=float(pr.lambda_conv),
                kappa=float(pr.kappa),
                rho_q=float(pr.rho_q),
                d_comp=float(ex.d_comp),
                p_perp=float(ex.p_perp_norm),
            )
        )
    return rows


def choose_stratified(indices, score, n):
    indices = np.asarray(indices, dtype=int)
    if n <= 0 or n >= len(indices):
        return indices
    order = indices[np.argsort(score[indices])]
    pos = np.unique(np.linspace(0, len(order) - 1, n).round().astype(int))
    return order[pos]


def panel_targets(
    seed, n_class, du_per_class, T, n_grid, response_class, eta, max_targets
):
    inst = utils.setup_instance(seed, n_class, du_per_class)
    X2q, Xall = utils.make_query_grid(n_grid=n_grid)
    Xq = fixed_compliance_Q(inst, eta, T, X2q, Xall, response_class, seed + 222)
    dis = utils.dishonest_theta(inst["theta"], inst["Delta"], eta)
    targets = choose_stratified(
        np.where(inst["retained"])[0], inst["rhos"], max_targets
    )
    rows = []
    for j, z in enumerate(targets):
        cache = precompute_non_world(inst, int(z))
        non = non_endpoint_from_cache(cache, inst["idx_Du"], int(z))
        mem = inst["theta_u_hon"]
        ex = utils.exact_geometry(
            inst["theta_u_hon"], dis, mem, non, Xq, response_class
        )
        pr = utils.convex_proxy_geometry(
            inst["theta"], inst["Delta"], inst["G_Du"], inst["Hinv"], inst["X"], inst["y"],
            int(z), eta, Xq, ex, response_class,
        )
        rows.append(
            dict(
                z_idx=int(z),
                rho=float(inst["rhos"][z]),
                gamma=float(ex.gamma),
                lambda_conv=float(pr.lambda_conv),
                rho_q=float(pr.rho_q),
                kappa=float(pr.kappa),
                d_cur=float(ex.d_cur),
                p_perp=float(ex.p_perp_norm),
            )
        )
        if (j + 1) % 50 == 0:
            print(f"  target sweep {j+1}/{len(targets)}", flush=True)
    return rows


def panel_du(n_seeds, n_class, T, n_grid, response_class, eta, du_values, n_targets):
    X2q, Xall = utils.make_query_grid(n_grid=n_grid)
    rows = []
    maxm = max(du_values)
    for s in range(n_seeds):
        base = utils.setup_instance(1000 + s, n_class, min(8, maxm))
        rng = np.random.default_rng(800000 + s)
        # Use nested, class-balanced deletion sets.
        orders = {
            k: rng.permutation(np.where(base["y"] == k)[0]) for k in range(utils.K)
        }
        max_du = np.concatenate([orders[k][:maxm] for k in range(utils.K)])
        safe = np.setdiff1d(np.arange(len(base["y"])), max_du)
        # Select a common target pool spanning the largest-set rho values.
        inst_max = instance_with_du(base, max_du)
        targets = choose_stratified(safe, inst_max["rhos"], n_targets)
        caches = {int(z): precompute_non_world(base, int(z)) for z in targets}
        for m in du_values:
            idx_du = np.concatenate([orders[k][:m] for k in range(utils.K)])
            inst = instance_with_du(base, idx_du)
            Xq = fixed_compliance_Q(
                inst, eta, T, X2q, Xall, response_class, s * 10000 + m
            )
            dis = utils.dishonest_theta(inst["theta"], inst["Delta"], eta)
            for z in targets:
                z = int(z)
                non = non_endpoint_from_cache(caches[z], idx_du, z)
                mem = inst["theta_u_hon"]
                ex = utils.exact_geometry(
                    inst["theta_u_hon"], dis, mem, non, Xq, response_class
                )
                pr = utils.convex_proxy_geometry(
                    inst["theta"], inst["Delta"], inst["G_Du"], inst["Hinv"], inst["X"], inst["y"],
                    z, eta, Xq, ex, response_class,
                )
                rows.append(
                    dict(
                        seed=s,
                        du_per_class=m,
                        du_size=len(idx_du),
                        du_frac=len(idx_du) / len(base["y"]),
                        z_idx=z,
                        gamma=float(ex.gamma),
                        lambda_conv=float(pr.lambda_conv),
                        kappa=float(pr.kappa),
                        rho=float(pr.rho),
                        rho_q=float(pr.rho_q),
                        d_comp=float(ex.d_comp),
                        p_perp=float(ex.p_perp_norm),
                    )
                )
        print(f"  D_u sweep seed {s+1}/{n_seeds}", flush=True)
    return rows


def grouped_quantiles(rows, xkey, ykey):
    out = []
    for x in sorted(set(r[xkey] for r in rows)):
        v = np.array([r[ykey] for r in rows if r[xkey] == x and np.isfinite(r[ykey])])
        out.append(
            dict(
                x=x,
                mean=float(v.mean()),
                median=float(np.median(v)),
                std=float(v.std()),
                p10=float(np.quantile(v, 0.1)),
                p90=float(np.quantile(v, 0.9)),
                p99=float(np.quantile(v, 0.99)),
            )
        )
    return out


def plot_all(er, tr, dr, out):
    outputs = []

    # Plot the update-scale sweep.
    fig, a = utils.single_ax_fig()
    x = np.array([r["eta"] for r in er])
    g = np.array([r["gamma"] for r in er])
    l = np.array([r["lambda_conv"] for r in er])
    a.plot(x, g, marker="o", ms=3.5, lw=1.8, label=r"exact $\gamma_Q$")
    a.plot(x, l, "--", lw=1.8, label=r"proxy $\lambda_Q^{\rm conv}$")
    a.set_yscale("log")
    a.set_xlabel(r"honesty level $\eta$", fontsize=24)
    a.set_ylabel(r"alignment coefficient", fontsize=24)
    a.grid(alpha=0.25)
    a.legend(fontsize=24)
    pdf = os.path.join(out, "gamma_tuning_eta.pdf")
    fig.savefig(pdf, dpi=300, bbox_inches=None)
    plt.close(fig)
    outputs.append(pdf)

    # Plot the retained-target distribution.
    fig, a = utils.single_ax_fig()
    g = np.array([r["gamma"] for r in tr])
    upper = max(np.quantile(g, 0.995) * 1.05, 1e-6)
    bins = np.linspace(0, upper, 22)
    a.hist(g, bins=bins, density=True, alpha=0.3, edgecolor="black", linewidth=0.4)
    med, p90, p99 = np.quantile(g, [0.5, 0.9, 0.99])
    for val, lab in [
        (med, f"median={med:.3f}"),
        (p90, rf"$\gamma_Q$ @ 90%={p90:.3f}"),
        (p99, rf"$\gamma_Q$ @ 99%={p99:.3f}"),
    ]:
        a.axvline(val, ls="--", lw=1.8, label=lab)
    a.set_yscale("log")
    a.set_xlabel(r"$\gamma_Q(z^*)$", fontsize=24)
    a.set_ylabel("density", fontsize=24)
    a.grid(alpha=0.25)
    a.legend(fontsize=24)
    pdf = os.path.join(out, "gamma_tuning_targets.pdf")
    fig.savefig(pdf, dpi=300, bbox_inches=None)
    plt.close(fig)
    outputs.append(pdf)

    # Plot the deletion-set-size sweep.
    fig, a = utils.single_ax_fig()
    q = grouped_quantiles(dr, "du_frac", "gamma")
    x = np.array([r["x"] for r in q])
    a.plot(x, [r["median"] for r in q], marker="o", ms=3.5, lw=1.7, label="median")
    a.plot(
        x, [r["p90"] for r in q], marker="s", ms=3.2, lw=1.5, label=r"$\gamma_Q$ @ 90%"
    )
    a.plot(
        x, [r["p99"] for r in q], marker="^", ms=3.2, lw=1.5, label=r"$\gamma_Q$ @ 99%"
    )
    # Fit an inverse-size reference to the 90th percentile.
    y = np.array([r["p90"] for r in q])
    z = 1 / np.maximum(x, 1e-12)
    c = float(y @ z / (z @ z))
    a.plot(x, c / x, "k--", lw=1.3, label=r"fitted $\propto1/|D_u|$")
    a.set_yscale("log")
    a.set_xlabel(r"$|D_u|/|D|$", fontsize=24)
    a.set_ylabel(r"$\gamma_Q(D_u)$", fontsize=24)
    a.grid(alpha=0.25)
    a.legend(fontsize=24)
    pdf = os.path.join(out, "gamma_tuning_du_size.pdf")
    fig.savefig(pdf, dpi=300, bbox_inches=None)
    plt.close(fig)
    outputs.append(pdf)
    return outputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_class", type=int, default=120)
    ap.add_argument("--du_per_class", type=int, default=8)
    ap.add_argument("--T", type=int, default=100)
    ap.add_argument("--n_grid", type=int, default=81)
    ap.add_argument("--response_class", type=int, default=0)
    ap.add_argument("--eta_seed", type=int, default=7)
    ap.add_argument("--eta_ref", type=float, default=0.5)
    ap.add_argument("--target_seed", type=int, default=7)
    ap.add_argument("--target_eta", type=float, default=0.5)
    ap.add_argument("--max_targets", type=int, default=160)
    ap.add_argument("--du_seeds", type=int, default=6)
    ap.add_argument("--du_targets", type=int, default=50)
    ap.add_argument("--du_eta", type=float, default=0.5)
    ap.add_argument("--du_values", type=str, default="1,2,3,4,6,8,10,12,16,20,24,28,32")
    ap.add_argument("--out_dir", default="results_gamma_tuning")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    dvs = [int(x) for x in args.du_values.split(",") if x.strip()]
    print("[1/3] eta", flush=True)
    er = panel_eta(
        args.eta_seed,
        args.n_class,
        args.du_per_class,
        args.T,
        args.n_grid,
        args.response_class,
        args.eta_ref,
    )
    write_csv(os.path.join(args.out_dir, "gamma_eta.csv"), er)
    print("[2/3] targets", flush=True)
    tr = panel_targets(
        args.target_seed,
        args.n_class,
        args.du_per_class,
        args.T,
        args.n_grid,
        args.response_class,
        args.target_eta,
        args.max_targets,
    )
    write_csv(os.path.join(args.out_dir, "gamma_targets.csv"), tr)
    print("[3/3] Du", flush=True)
    dr = panel_du(
        args.du_seeds,
        args.n_class,
        args.T,
        args.n_grid,
        args.response_class,
        args.du_eta,
        dvs,
        args.du_targets,
    )
    write_csv(os.path.join(args.out_dir, "gamma_du_size.csv"), dr)
    plot_paths = plot_all(er, tr, dr, args.out_dir)
    print(
        f"eta gamma/lambda corr={np.corrcoef([r['gamma'] for r in er],[r['lambda_conv'] for r in er])[0,1]:.4f}"
    )
    print(
        f"target n={len(tr)} gamma median/p90/p99={np.quantile([r['gamma'] for r in tr],[.5,.9,.99])}"
    )
    for path in plot_paths:
        print(path)


if __name__ == "__main__":
    main()
