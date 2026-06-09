"""Mean residual ||Δz|| vs iteration."""
import math
from pathlib import Path

import matplotlib.pyplot as plt

from src.attractors.iteration import ConvergenceStats


def plot_convergence(
    stats_per_label: dict[str, ConvergenceStats],
    save_path: Path,
    tol: float,
):
    """Mean ||Δz|| vs iteration (log-y), threshold at sqrt(tol)."""
    fig, ax = plt.subplots(figsize=(7, 4))

    for label, stats in stats_per_label.items():
        if not stats.residuals:
            print(f"  Warning: no residuals for '{label}'. Pass track_residuals=True.")
            continue
        iters = range(1, len(stats.residuals) + 1)
        ax.plot(list(iters), stats.residuals, label=label, linewidth=1.5)

    thresh = math.sqrt(tol)
    ax.axhline(thresh, color="black", linestyle="--", linewidth=1.0, label=f"threshold √tol = {thresh:.0e}")

    ax.set_yscale("log")
    ax.set_xlabel("Iteration $t$")
    ax.set_ylabel(r"Mean $\|z_{t+1} - z_t\|_2$")
    ax.set_title("Attractor iteration — residual convergence")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path}")
