import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {"none": "black", "gaussian": "tab:blue", "laplace": "tab:orange"}
NAMES = {"none": "Unmediated baseline", "gaussian": "Gaussian", "laplace": "Laplace"}


def draw_estimates(ax, rows, x, metric, label, color, linestyle="-"):
    rows = rows.sort_values(x)
    # Draw the intervals directly: percentile intervals need not be centered
    # on the estimate, and empirical performance is not clipped to a ceiling.
    ax.vlines(rows[x], rows[metric + "_ci_low"], rows[metric + "_ci_high"],
              color=color, alpha=0.65, linewidth=1.3)
    ax.plot(rows[x], rows[metric], marker="o", markersize=4,
            linestyle=linestyle, color=color, label=label)


def save_figure(fig, directory, name):
    fig.tight_layout()
    fig.savefig(os.path.join(directory, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(directory, name + ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def certificate_plot(rows, T, directory, metric, ylabel, name, bound, bound_label):
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for (mediator, sweep), group in rows.groupby(["mediator", "sweep"], sort=False):
        label = NAMES[mediator]
        if sweep != "baseline":
            label += " (" + ("scale sweep" if sweep == "scale" else "budget sweep") + ")"
        draw_estimates(ax, group, "b_cert", metric, label, COLORS[mediator],
                       "--" if sweep == "budget" else "-")
    b = np.linspace(0.0, 1.0, 200)
    ax.plot(b, bound(b), ":", color="tab:red", label=bound_label)
    ax.set(xlabel="Certified domain-wide TV upper bound", ylabel=ylabel,
           title=f"T = {T}; 95% held-out model bootstrap intervals", xlim=(-0.02, 1.02))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    save_figure(fig, directory, name)


def plot_results(result_dir, T=None):
    summary = pd.read_csv(os.path.join(result_dir, "mediation_summary.csv"))
    T = int(summary["T"].max()) if T is None else T
    rows = summary[summary["T"] == T]
    directory = os.path.join(result_dir, "figures", f"T{T:03d}")
    os.makedirs(directory, exist_ok=True)

    certificate_plot(
        rows, T, directory, "compliance_balanced_accuracy", "Audit balanced accuracy",
        "utility_vs_certificate", lambda b: (1 + b) / 2, "Utility ceiling (1 + b) / 2",
    )
    certificate_plot(
        rows, T, directory, "e_op", "Audit error sum", "error_vs_certificate",
        lambda b: 1 - b, "Error floor 1 - b",
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharex=True, sharey=True)
    for ax, metric, title in zip(axes, (
        "membership_advantage", "membership_advantage_reference", "membership_advantage_alternative",
    ), ("Equal update mixture", "Reference endpoints", "Alternative endpoints")):
        for (mediator, sweep), group in rows.groupby(["mediator", "sweep"], sort=False):
            label = NAMES[mediator] + (" / " + sweep if sweep != "baseline" else "")
            draw_estimates(ax, group, "b_cert", metric, label, COLORS[mediator],
                           "--" if sweep == "budget" else "-")
        ax.plot([0, 1], [0, 1], ":", color="tab:red", label="Population advantage upper bound")
        ax.set(xlabel="Certified domain-wide TV upper bound", title=title, xlim=(-0.02, 1.02))
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Measured membership advantage (TPR - FPR)")
    axes[-1].legend(fontsize=7)
    fig.suptitle(f"T = {T}; 95% held-out model bootstrap intervals")
    save_figure(fig, directory, "membership_vs_certificate")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, metric, ylabel in zip(axes, ("compliance_balanced_accuracy", "utility_loss"),
                                 ("Audit balanced accuracy", "Paired utility loss from baseline")):
        for mediator, group in rows[rows["sweep"] == "scale"].groupby("mediator", sort=False):
            scale_unit = "standard deviation" if mediator == "gaussian" else "Laplace scale"
            draw_estimates(ax, group, "noise_scale", metric,
                           NAMES[mediator] + " (" + scale_unit + ")", COLORS[mediator])
        draw_estimates(ax, rows[rows["sweep"] == "baseline"], "noise_scale", metric,
                       NAMES["none"], COLORS["none"])
        ax.set(xlabel="Mediator scale parameter", ylabel=ylabel)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle(f"T = {T}; Laplace standard deviation = sqrt(2) × scale")
    save_figure(fig, directory, "utility_vs_noise_scale")

    budgets = np.sort(summary.loc[summary["sweep"] == "budget", "requested_budget"].unique())
    if len(budgets):
        ncols = min(2, len(budgets))
        nrows = (len(budgets) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4.5 * nrows), squeeze=False)
        for ax, budget in zip(axes.flat, budgets):
            budget_rows = summary[(summary["sweep"] == "budget") & (summary["requested_budget"] == budget)]
            for mediator, group in budget_rows.groupby("mediator", sort=False):
                draw_estimates(ax, group, "T", "compliance_balanced_accuracy", NAMES[mediator], COLORS[mediator])
            draw_estimates(ax, summary[summary["sweep"] == "baseline"], "T", "compliance_balanced_accuracy",
                           NAMES["none"], COLORS["none"])
            ax.axhline((1 + budget) / 2, linestyle=":", color="tab:red", label="Requested-budget ceiling")
            ax.set(xlabel="Query budget T", ylabel="Audit balanced accuracy", title=f"Certified budget b ≤ {budget:g}")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
        for ax in axes.flat[len(budgets):]:
            ax.set_visible(False)
        fig.suptitle("Fixed certified budgets; 95% held-out model bootstrap intervals")
        save_figure(fig, directory, "utility_vs_queries_fixed_budget")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", required=True)
    parser.add_argument("--T", type=int, default=None, help="Query budget for certificate/scale plots; default: largest saved T.")
    args = parser.parse_args()
    plot_results(args.result_dir, args.T)


if __name__ == "__main__":
    main()
