"""§11.1 — Latent space PCA scatter across seeds."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def pca_project(Z: torch.Tensor, fit_Z: torch.Tensor | None = None) -> np.ndarray:
    """
    Project (N, k) embeddings to 2D via PCA.
    fit_Z: if given, PCA is fitted on fit_Z and applied to Z (shared PCA).
           if None, PCA is fitted on Z itself (independent PCA).
    Returns (N, 2) numpy array.
    """
    Z_f = Z.float().cpu()
    fit = fit_Z.float().cpu() if fit_Z is not None else Z_f
    mean = fit.mean(0)
    _, _, Vh = torch.linalg.svd(fit - mean, full_matrices=False)
    components = Vh[:2].T          # (k, 2)
    return ((Z_f - mean) @ components).numpy()


CLASS_COLORS = plt.cm.tab10.colors  # 10 distinct colors


def plot_latent_scatter(
    embeddings_per_seed: dict[int, torch.Tensor],
    labels: torch.Tensor,
    save_dir: Path,
    shared_pca_ref_seed: int | None = 1,
):
    """
    Produce two scatter plots per §11.1:
      1. Independent PCA per seed  -> latent_pca_per_seed.png
      2. Shared PCA (fit on ref seed, applied to all) -> latent_pca_shared.png

    embeddings_per_seed: {seed: (N, k) tensor}
    labels: (N,) integer class labels
    shared_pca_ref_seed: seed used to fit the shared PCA basis
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    seeds = sorted(embeddings_per_seed.keys())
    n = len(seeds)
    labs = labels.cpu().numpy()

    # --- Independent PCA ---
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, s in zip(axes, seeds):
        proj = pca_project(embeddings_per_seed[s])
        for c in range(10):
            mask = labs == c
            ax.scatter(proj[mask, 0], proj[mask, 1], s=4, alpha=0.5, color=CLASS_COLORS[c], label=str(c))
        ax.set_title(f"Seed {s} (independent PCA)")
        ax.set_xticks([]); ax.set_yticks([])
    axes[-1].legend(markerscale=3, bbox_to_anchor=(1.05, 1), loc="upper left", title="Class")
    fig.suptitle("Latent space — independent PCA per seed", y=1.01)
    fig.tight_layout()
    path = save_dir / "latent_pca_per_seed.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")

    # --- Shared PCA (fit on reference seed) ---
    if shared_pca_ref_seed is not None and shared_pca_ref_seed in embeddings_per_seed:
        ref_Z = embeddings_per_seed[shared_pca_ref_seed]
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
        if n == 1:
            axes = [axes]
        for ax, s in zip(axes, seeds):
            proj = pca_project(embeddings_per_seed[s], fit_Z=ref_Z)
            for c in range(10):
                mask = labs == c
                ax.scatter(proj[mask, 0], proj[mask, 1], s=4, alpha=0.5, color=CLASS_COLORS[c], label=str(c))
            ax.set_title(f"Seed {s} (PCA fit on seed {shared_pca_ref_seed})")
            ax.set_xticks([]); ax.set_yticks([])
        axes[-1].legend(markerscale=3, bbox_to_anchor=(1.05, 1), loc="upper left", title="Class")
        fig.suptitle(f"Latent space — shared PCA (fit on seed {shared_pca_ref_seed})", y=1.01)
        fig.tight_layout()
        path = save_dir / "latent_pca_shared.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {path}")
