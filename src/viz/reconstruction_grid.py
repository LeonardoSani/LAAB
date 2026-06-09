import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.relative.cosine_map import relative_cosine


def _pick_one_per_class(test_ds, n_classes: int = 10, seed: int = 0) -> list[int]:
    """One index per digit class, deterministic."""
    rng = np.random.default_rng(seed)
    per_class: dict[int, list[int]] = {c: [] for c in range(n_classes)}
    for idx in range(len(test_ds)):
        _, label = test_ds[idx]
        lbl = int(label)
        if lbl in per_class:
            per_class[lbl].append(idx)
    chosen = []
    for c in range(n_classes):
        pool = per_class[c]
        chosen.append(int(rng.choice(pool)))
    return chosen


def _to_img(t: torch.Tensor) -> np.ndarray:
    """(1,H,W) -> (H,W) numpy in [0,1]."""
    return t.squeeze(0).cpu().float().numpy().clip(0, 1)


@torch.no_grad()
def _decode_mixed(
    encoder: nn.Module,
    anchors: torch.Tensor,       # (N, k)
    rel_decoder: nn.Module,
    images: torch.Tensor,        # (B, 1, 28, 28)
    device: torch.device,
) -> torch.Tensor:               # (B, 1, 28, 28) CPU
    encoder.eval()
    rel_decoder.eval()
    z = encoder(images.to(device))
    rep = relative_cosine(z, anchors.to(device))
    x_hat = rel_decoder(rep)
    return x_hat.cpu()


def _render_grid(
    rows: list[list[np.ndarray]],   # [row][col] -> (H,W)
    col_labels: list[str],
    row_labels: list[str],
    title: str,
    save_path: Path,
) -> None:
    n_rows = len(rows)
    n_cols = len(rows[0])
    cell = 1.5
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * cell, n_rows * cell + 0.6))
    for r in range(n_rows):
        for c in range(n_cols):
            ax = axes[r, c]
            ax.imshow(rows[r][c], cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if r == 0:
                ax.set_title(col_labels[c], fontsize=7, pad=2)
        axes[r, 0].set_ylabel(row_labels[r], fontsize=7, rotation=0, labelpad=18, va="center")

    fig.suptitle(title, fontsize=9, y=1.0)
    fig.tight_layout(pad=0.3)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_stitching_grid(
    ae_s1: nn.Module,            # AE seed 1 (ref)
    encoder_s2: nn.Module,       # cross-encoder source
    anchors_s2: dict,            # {t: (N,k)} for seed 2
    rel_decoders: dict,          # {t: RelativeDecoder}
    test_ds,
    device: torch.device,
    save_path: Path,
) -> None:
    """Fig 9: stitching grid, cols orig | AE s=1 | D_r^t(Mixed s=2)."""
    depth_order = [0, 8, 64, 512, float("inf")]
    depth_labels = ["t=0", "t=8", "t=64", "t=512", r"t=∞"]
    col_labels = ["orig", "AE s=1"] + [f"D_r({lbl})\nMixed" for lbl in depth_labels]

    idxs = _pick_one_per_class(test_ds)
    imgs = torch.stack([test_ds[i][0] for i in idxs])
    labels = [int(test_ds[i][1]) for i in idxs]
    row_labels = [str(lbl) for lbl in labels]

    ae_s1.eval()
    with torch.no_grad():
        ae_recon = ae_s1(imgs.to(device)).cpu()

    decoded_per_t = {}
    for t in depth_order:
        decoded_per_t[t] = _decode_mixed(
            encoder_s2, anchors_s2[t], rel_decoders[t], imgs, device
        )

    rows = []
    for r in range(10):
        row = [_to_img(imgs[r]), _to_img(ae_recon[r])]
        for t in depth_order:
            row.append(_to_img(decoded_per_t[t][r]))
        rows.append(row)

    _render_grid(rows, col_labels, row_labels,
                 "Stitching reconstruction — encoder s=2 (cross-encoder error)",
                 save_path)


def plot_reference_grid(
    ae_s1: nn.Module,            # AE seed 1 (ref)
    anchors_s1: dict,            # {t: (N,k)} for seed 1
    rel_decoders: dict,          # {t: RelativeDecoder}
    test_ds,
    device: torch.device,
    save_path: Path,
) -> None:
    """Fig 10: reference grid, cols orig | AE s=1 | D_r^t(Mixed s=1)."""
    depth_order = [0, 8, 64, 512, float("inf")]
    depth_labels = ["t=0", "t=8", "t=64", "t=512", r"t=∞"]
    col_labels = ["orig", "AE s=1"] + [f"D_r({lbl})\nM₁" for lbl in depth_labels]

    idxs = _pick_one_per_class(test_ds)
    imgs = torch.stack([test_ds[i][0] for i in idxs])
    labels = [int(test_ds[i][1]) for i in idxs]
    row_labels = [str(lbl) for lbl in labels]

    ae_s1.eval()
    with torch.no_grad():
        ae_recon = ae_s1(imgs.to(device)).cpu()

    decoded_per_t = {}
    for t in depth_order:
        decoded_per_t[t] = _decode_mixed(
            ae_s1.encoder, anchors_s1[t], rel_decoders[t], imgs, device
        )

    rows = []
    for r in range(10):
        row = [_to_img(imgs[r]), _to_img(ae_recon[r])]
        for t in depth_order:
            row.append(_to_img(decoded_per_t[t][r]))
        rows.append(row)

    _render_grid(rows, col_labels, row_labels,
                 "Reference reconstruction — encoder s=1 (relative rep + decoder error only)",
                 save_path)
