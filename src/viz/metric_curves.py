import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def _ordinal_axis(ax, t_vals):
    """Set ordinal x-axis with depth labels."""
    positions = list(range(len(t_vals)))
    labels = [r"$\infty$" if t == "inf" or t == float("inf") else str(t) for t in t_vals]
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8)
    return positions


def _quantile_bands(ax, xs, df, metric, color):
    """Draw nested quantile bands: 5–95 (outer, light) and 25–75 (inner, dark)."""
    ax.fill_between(xs, df[f"{metric}_q5"], df[f"{metric}_q95"],
                    alpha=0.12, color=color, label="5–95%")
    ax.fill_between(xs, df[f"{metric}_q25"], df[f"{metric}_q75"],
                    alpha=0.28, color=color, label="25–75%")


def plot_mixed_curves(
    df: pd.DataFrame,       # e1_mixed.csv
    save_path: Path,
    baseline: dict,         # {"M1_mean": float, "M2_mean": float} from e0_baseline.csv
) -> None:
    """Figure 2 (E1): two panels (M1 left, M2 right), mean + nested quantile bands."""
    t_vals = df["t"].tolist()
    xs = list(range(len(t_vals)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    for ax, metric, label in [(ax1, "M1", r"$M_1$ ($L^2$)"), (ax2, "M2", r"$M_2$ ($1-\cos$)")]:
        mean = df[f"{metric}_mean"].values
        (line,) = ax.plot(xs, mean, marker="o", markersize=3, label="mean")
        _quantile_bands(ax, xs, df, metric, line.get_color())
        ax.axhline(baseline[f"{metric}_mean"], color="gray", linestyle="--",
                   linewidth=1, label=r"baseline ($t=0$)")
        _ordinal_axis(ax, t_vals)
        ax.set_xlabel("depth $t$")
        ax.set_ylabel(label)
        ax.set_title(f"{label} over depth (Mixed)")
        ax.legend(fontsize=8)

    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path.with_suffix(".png"), dpi=150)
    plt.close(fig)


def plot_full_vs_mixed(
    df_mixed: pd.DataFrame,  # e1_mixed.csv
    df_full: pd.DataFrame,   # e4_full.csv
    save_path: Path,
) -> None:
    """Figure 7 (E4): M1 curves for Mixed (solid) and Full (dashed), nested quantile bands."""
    t_vals = df_mixed["t"].tolist()
    xs = list(range(len(t_vals)))

    fig, ax = plt.subplots(figsize=(6, 4))

    for df, style, lbl in [(df_mixed, "-", "Mixed"), (df_full, "--", "Full")]:
        mean = df["M1_mean"].values
        (line,) = ax.plot(xs, mean, linestyle=style, marker="o", markersize=3, label=f"{lbl} mean")
        _quantile_bands(ax, xs, df, "M1", line.get_color())

    _ordinal_axis(ax, t_vals)
    ax.set_xlabel("depth $t$")
    ax.set_ylabel(r"$M_1$ ($L^2$)")
    ax.set_title(r"Full vs Mixed — $M_1$")
    ax.legend(fontsize=9)
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path.with_suffix(".png"), dpi=150)
    plt.close(fig)
