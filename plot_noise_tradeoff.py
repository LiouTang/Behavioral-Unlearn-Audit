"""Plot held-out curiosity and compliance against mediator noise scale."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHODS = (("ga", "GA"), ("salun", "SalUn"))
ETAS = (0.1, 0.5, 0.9)
BLUE = "#0072B2"
ORANGE = "#D55E00"


def read_results(evidence_dir, query_budget, include_calibrated=False):
    frames = []
    for method, label in METHODS:
        for eta in ETAS:
            path = evidence_dir / f"{method}-eta{eta}" / "mediation_summary.csv"
            rows = pd.read_csv(path)
            sweeps = ["baseline", "scale", "budget"] if include_calibrated else ["baseline", "scale"]
            rows = rows[(rows["T"] == query_budget) & rows["sweep"].isin(sweeps)].copy()
            expected = 19 if include_calibrated else 11
            if len(rows) != expected or (rows["sweep"] == "baseline").sum() != 1:
                raise ValueError(f"Expected one baseline and {expected - 1} noise policies at T={query_budget}: {path}")
            rows["method"] = label
            rows["eta"] = eta
            frames.append(rows)
    return pd.concat(frames, ignore_index=True)


def plot_series(ax, rows, metric, color, marker, linestyle):
    x = rows["noise_scale"].to_numpy()
    low = rows[metric + "_ci_low"].to_numpy()
    high = rows[metric + "_ci_high"].to_numpy()
    ax.fill_between(x, low, high, color=color, alpha=0.10, linewidth=0)
    ax.vlines(x, low, high, color=color, alpha=0.65, linewidth=0.9)
    ax.plot(x, rows[metric], color=color, marker=marker, markersize=4,
            linewidth=1.35, linestyle=linestyle)


def make_panels(rows, mediator, output_dir, include_calibrated=False):
    beta_extent = max(abs(rows["membership_advantage_ci_low"].min()),
                      abs(rows["membership_advantage_ci_high"].max()))
    beta_limit = np.ceil((beta_extent + 0.015) / 0.02) * 0.02
    error_limit = np.ceil((rows["e_op_ci_high"].max() + 0.03) / 0.1) * 0.1
    max_noise = rows.loc[rows["mediator"] == mediator, "noise_scale"].max()
    xlabel = (r"Noise standard deviation $\sigma_M$" if mediator == "gaussian"
              else r"Laplace noise scale $\ell$")
    suffix = "-full" if include_calibrated else ""
    for eta in ETAS:
        for _, label in METHODS:
            fig, ax = plt.subplots(figsize=(3.5, 2.6))
            fig.subplots_adjust(left=0.22, right=0.78, top=0.96, bottom=0.24)
            right = ax.twinx()
            subset = rows[(rows["method"] == label) & (rows["eta"] == eta)
                          & rows["mediator"].isin(["none", mediator])].sort_values("noise_scale")
            if (not include_calibrated and not np.allclose(subset["noise_scale"], [0, 0.1, 0.5, 1, 2, 5])) or (include_calibrated and len(subset) != 10):
                raise ValueError(f"Unexpected noise-scale grid for {label}, eta={eta}, {mediator}")
            plot_series(ax, subset, "membership_advantage", BLUE, "o", "--")
            plot_series(right, subset, "e_op", ORANGE, "s", "-")
            ax.set(xlim=(-0.12, 5.12), ylim=(-beta_limit, beta_limit),
                   xticks=[0, 1, 2, 3, 4, 5], xlabel=xlabel)
            if include_calibrated:
                ax.set_xscale("symlog", linthresh=0.5, linscale=0.7)
                ticks = [0, 0.5, 2, 10, 50] if mediator == "gaussian" else [0, 0.5, 5, 50, 500]
                ax.set(xlim=(-0.05, max_noise * 1.15), xticks=ticks)
                ax.set_xticklabels([f"{tick:g}" for tick in ticks])
                ax.minorticks_off()
            right.set(ylim=(0, error_limit))
            ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=4, steps=[1, 2, 5, 10], symmetric=True))
            right.yaxis.set_major_locator(plt.MaxNLocator(nbins=4))
            ax.tick_params(axis="y", colors=BLUE, pad=2)
            right.tick_params(axis="y", colors=ORANGE, pad=2)
            ax.tick_params(axis="x", pad=2)
            ax.spines["left"].set_color(BLUE)
            right.spines["right"].set_color(ORANGE)
            ax.axhline(0, color=BLUE, alpha=0.2, linewidth=0.7)
            ax.grid(axis="x", color="0.9", linewidth=0.6)
            ax.set_axisbelow(True)
            ax.set_ylabel(r"Curiosity advantage $\widehat{\beta}$", color=BLUE, labelpad=4)
            right.set_ylabel(r"Compliance error $\alpha+\alpha'$", color=ORANGE, labelpad=4)
            stem = output_dir / f"{mediator}-{label}-eta{eta:g}{suffix}"
            fig.savefig(f"{stem}.pdf")
            plt.close(fig)
            subset.assign(plotted_mediator=mediator).to_csv(f"{stem}.csv", index=False)
            print(f"Saved {stem}; beta limits +/-{beta_limit:g}, error limits [0,{error_limit:g}]")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--T", type=int, default=30)
    parser.add_argument("--include-calibrated", action="store_true",
                        help="Also plot budget-calibrated policies at their actual noise scales on a symlog x-axis.")
    args = parser.parse_args()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12.5,
                         "axes.labelsize": 12.5,
                         "xtick.labelsize": 12, "ytick.labelsize": 12,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    rows = read_results(args.evidence_dir, args.T, args.include_calibrated)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for mediator in ("gaussian", "laplace"):
        make_panels(rows, mediator, args.output_dir, args.include_calibrated)


if __name__ == "__main__":
    main()
