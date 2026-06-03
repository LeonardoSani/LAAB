import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_stitching(
    stitching_df: pd.DataFrame,  # e3_stitching.csv: columns t, mse_mean, mse_std
    delta_df: pd.DataFrame,      # e2_delta.csv: columns t, M3_q95 (for secondary axis)
    save_path: Path,
) -> None:
    """Figure 6 (E3): left y-axis MSE ± std at T_5; right y-axis q95(delta_i) over full T.
    Shared ordinal x-axis."""
    # Full T for q95 curve (13 depths)
    t_full = delta_df["t"].tolist()
    xs_full = list(range(len(t_full)))
    full_labels = [r"$\infty$" if str(t) == "inf" else str(t) for t in t_full]

    # T_5 for MSE bars — map to ordinal positions in T_full
    t5_strs = [str(t) for t in stitching_df["t"].tolist()]
    t_full_strs = [str(t) for t in t_full]
    xs_t5 = [t_full_strs.index(s) for s in t5_strs]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    # Left: MSE
    mse_mean = stitching_df["mse_mean"].values
    mse_std = stitching_df["mse_std"].values
    ax1.errorbar(xs_t5, mse_mean, yerr=mse_std, fmt="s-", color="steelblue",
                 capsize=4, label=r"stitching MSE $M_4$")
    ax1.set_ylabel(r"$M_4$ (stitching MSE)", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")

    # Right: q95
    q95 = delta_df["M3_q95"].values
    ax2.plot(xs_full, q95, "o--", color="tomato", markersize=4, label=r"$q_{95}(M_3)$")
    ax2.set_ylabel(r"$q_{95}(M_3)$", color="tomato")
    ax2.tick_params(axis="y", labelcolor="tomato")

    ax1.set_xticks(xs_full)
    ax1.set_xticklabels(full_labels, fontsize=8)
    ax1.set_xlabel("depth $t$")
    ax1.set_title(r"Stitching $M_4$ vs per-anchor $M_3$ tail")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path.with_suffix(".png"), dpi=150)
    plt.close(fig)
