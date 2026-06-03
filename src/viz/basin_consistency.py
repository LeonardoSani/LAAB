import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_basin_consistency(
    J: np.ndarray,        # (N,) per-anchor cross-seed basin Jaccard
    J_null: np.ndarray,   # (N,) size-matched chance Jaccard
    save_path: Path,
) -> None:
    """E5 (global): are attractor basins shared across independently trained
    seeds? The per-anchor cross-seed basin Jaccard sits at its size-matched
    chance level, so the dynamics are not shared — independently trained fields
    carve the data into different basins."""
    obs, chance = float(J.mean()), float(J_null.mean())

    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    ax.bar(["observed", "chance"], [obs, chance],
           color=["C0", "gray"], width=0.6)
    ax.set_ylabel("cross-seed basin Jaccard")
    ax.set_title(r"Attractor basins are not shared" + "\n" + r"across seeds ($t=\infty$)")
    for xi, v in enumerate([obs, chance]):
        ax.text(xi, v, f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, max(obs, chance) * 1.25)

    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path.with_suffix(".png"), dpi=150)
    plt.close(fig)
