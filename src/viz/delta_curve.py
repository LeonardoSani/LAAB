import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_delta_curve(
    df: pd.DataFrame,        # e2_delta.csv
    save_path: Path,
    ref_t0_mean: float,      # e0 baseline M3 mean
    ref_tinf_mean: float = 0.090,
) -> None:
    """Fig 3: M3 mean + nested quantile bands over depth (Mixed)."""
    t_vals = df["t"].tolist()
    xs = list(range(len(t_vals)))
    mean = df["M3_mean"].values

    fig, ax = plt.subplots(figsize=(7, 4))
    (line,) = ax.plot(xs, mean, marker="o", markersize=3, label="mean $M_3$")
    c = line.get_color()
    ax.fill_between(xs, df["M3_q5"], df["M3_q95"], alpha=0.12, color=c, label="5–95%")
    ax.fill_between(xs, df["M3_q25"], df["M3_q75"], alpha=0.28, color=c, label="25–75%")

    ax.axhline(ref_t0_mean, color="gray", linestyle="--", linewidth=1,
               label=f"baseline ($t=0$): {ref_t0_mean:.4f}")

    labels = [r"$\infty$" if str(t) == "inf" else str(t) for t in t_vals]
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlabel("depth $t$")
    ax.set_ylabel(r"$M_3$")
    ax.set_title(r"Per-anchor mismatch $M_3$ over depth (Mixed)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path.with_suffix(".png"), dpi=150)
    plt.close(fig)
