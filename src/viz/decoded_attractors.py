import math

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path


@torch.no_grad()
def plot_decoded_attractors(
    decoder: nn.Module,
    attractors: torch.Tensor,   # (N, k) attractor latent vectors
    X_a: torch.Tensor,          # (N, 1, 28, 28) source images
    save_path: Path,
    n: int = 64,
    device: torch.device = torch.device("cpu"),
) -> None:
    """
    Plot decoded attractors alongside their source images.

    Top row block: original X_a samples.
    Bottom row block: D_s(attractor_i) for same indices.

    A memorization regime shows near-identical decoded images regardless of input.
    Healthy dynamics show diverse, digit-like reconstructions that differ from source.
    """
    n = min(n, attractors.shape[0])
    cols = min(16, n)
    rows_per_block = math.ceil(n / cols)

    decoder.eval()
    decoder.to(device)
    z = attractors[:n].to(device)
    decoded = decoder(z).cpu().clamp(0, 1)  # (n, 1, 28, 28)

    src = X_a[:n].clamp(0, 1)  # (n, 1, 28, 28)

    # Total figure: two blocks (source / decoded), each rows_per_block rows
    total_rows = rows_per_block * 2 + 1  # +1 for gap label row
    fig, axes = plt.subplots(total_rows, cols, figsize=(cols * 0.9, total_rows * 0.9))
    for ax in axes.ravel():
        ax.axis("off")

    def _place(block_row_offset, images):
        for idx in range(n):
            r = idx // cols + block_row_offset
            c = idx % cols
            axes[r, c].imshow(images[idx, 0].numpy(), cmap="gray", vmin=0, vmax=1)

    _place(0, src)
    _place(rows_per_block + 1, decoded)

    # Labels
    fig.text(0.01, 1 - (rows_per_block / 2) / total_rows, "Source $X_a$",
             va="center", fontsize=9, fontweight="bold")
    fig.text(0.01, (rows_per_block / 2) / total_rows, r"$D_s(\phi_s^\infty)$",
             va="center", fontsize=9, fontweight="bold")

    fig.suptitle("Decoded attractors vs source (memorization check)", fontsize=10)
    fig.tight_layout(rect=[0.04, 0, 1, 0.97])

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path.with_suffix(".png"), dpi=150)
    plt.close(fig)
