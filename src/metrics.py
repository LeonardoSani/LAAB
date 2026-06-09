"""Metrics M1-M4 and per-anchor badness mu_i. Inputs are relative codes (B, N)."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def M1(r_s: torch.Tensor, r_sp: torch.Tensor) -> torch.Tensor:
    """||r_s - r_sp||_2 -> (B,)."""
    return (r_s - r_sp).norm(p=2, dim=1)


def M2(r_s: torch.Tensor, r_sp: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """1 - cos(r_s, r_sp) -> (B,)."""
    a = F.normalize(r_s, p=2, dim=1, eps=eps)
    b = F.normalize(r_sp, p=2, dim=1, eps=eps)
    return 1.0 - (a * b).sum(dim=1)


def M3(r_s: torch.Tensor, r_sp: torch.Tensor) -> torch.Tensor:
    """Per-anchor |r_s - r_sp| -> (B,N)."""
    return (r_s - r_sp).abs()


@torch.no_grad()
def M4(
    decoder: nn.Module,
    rep_per_seed: dict[int, torch.Tensor],   # {seed: (|D|, N) Mixed rep}
    images: torch.Tensor,                    # (|D|, 1, 28, 28)
    ref_seed: int,
    device: torch.device,
) -> tuple[float, float, dict[int, float]]:
    """Stitching MSE: decode each non-ref seed's rep through the ref decoder.
    Returns (mean, std, {s: MSE})."""
    decoder.eval()
    decoder.to(device)
    x = images.to(device)
    per_seed: dict[int, float] = {}
    for s, R in rep_per_seed.items():
        if s == ref_seed:
            continue
        x_hat = decoder(R.to(device))
        mse = ((x_hat - x) ** 2).flatten(1).mean(1)
        per_seed[s] = float(mse.mean().cpu())
    vals = torch.tensor(list(per_seed.values()))
    return float(vals.mean()), float(vals.std()), per_seed


def per_pair_badness(r_s: torch.Tensor, r_sp: torch.Tensor) -> torch.Tensor:
    """Mean M3 over points for one pair -> (N,)."""
    return M3(r_s, r_sp).mean(dim=0)


def anchor_badness(pair_badness: np.ndarray) -> np.ndarray:
    """mu_i: mean over pairs -> (..., N)."""
    return pair_badness.mean(axis=-1)
