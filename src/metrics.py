"""The four metrics M1-M4 and the per-anchor badness mu_i (Method, eq. M1-M4
and per-anchor badness), in one place.

All operate on relative representations r_s(x) = relative_cosine(Z_s, A_s),
shape (B, N): B test points x, N anchors. M1-M3 are defined pointwise on a
test point and a seed pair, (x, s, s'); M4 is a stitching error over seeds vs a
reference decoder; mu_i aggregates M3 into the per-anchor badness.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --- M1, M2, M3: defined on (x, s, s') ---

def M1(r_s: torch.Tensor, r_sp: torch.Tensor) -> torch.Tensor:
    """M1(x,s,s') = ||r_s(x) - r_sp(x)||_2.  (B,N)x(B,N) -> (B,)."""
    return (r_s - r_sp).norm(p=2, dim=1)


def M2(r_s: torch.Tensor, r_sp: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """M2(x,s,s') = 1 - cos(r_s(x), r_sp(x)).  (B,N)x(B,N) -> (B,)."""
    a = F.normalize(r_s, p=2, dim=1, eps=eps)
    b = F.normalize(r_sp, p=2, dim=1, eps=eps)
    return 1.0 - (a * b).sum(dim=1)


def M3(r_s: torch.Tensor, r_sp: torch.Tensor) -> torch.Tensor:
    """M3(x,s,s')_i = |r_s(x)_i - r_sp(x)_i|, per anchor i.  (B,N)x(B,N) -> (B,N)."""
    return (r_s - r_sp).abs()


# --- M4: stitching MSE, defined on (s) vs a reference relative decoder ---

@torch.no_grad()
def M4(
    decoder: nn.Module,
    rep_per_seed: dict[int, torch.Tensor],   # {seed: (|D|, N) Mixed rep}
    images: torch.Tensor,                    # (|D|, 1, 28, 28)
    ref_seed: int,
    device: torch.device,
) -> tuple[float, float, dict[int, float]]:
    """For each non-reference seed s, decode rep_per_seed[s] through the
    reference relative `decoder`, take per-sample MSE vs `images`, average over
    the dataset. Returns (mean over the S-1 per-seed means, std, {s: mean MSE})."""
    decoder.eval()
    decoder.to(device)
    x = images.to(device)
    per_seed: dict[int, float] = {}
    for s, R in rep_per_seed.items():
        if s == ref_seed:
            continue
        x_hat = decoder(R.to(device))
        mse = ((x_hat - x) ** 2).flatten(1).mean(1)   # (|D|,)
        per_seed[s] = float(mse.mean().cpu())
    vals = torch.tensor(list(per_seed.values()))
    return float(vals.mean()), float(vals.std()), per_seed


# --- per-anchor badness mu_i, built from M3 ---
# Averaging order matters: first over test points x (per pair), then over the
# seed couples (s, s').

def per_pair_badness(r_s: torch.Tensor, r_sp: torch.Tensor) -> torch.Tensor:
    """Mean of M3 over test points for ONE pair: bar_delta_i(s,s').
    (B,N)x(B,N) -> (N,)."""
    return M3(r_s, r_sp).mean(dim=0)


def anchor_badness(pair_badness: np.ndarray) -> np.ndarray:
    """Per-anchor badness mu_i = mean over seed pairs of per_pair_badness.
    Operates on the (N, n_pairs) badness slice (or any (..., N, n_pairs)),
    reducing the trailing pair axis -> (..., N). The reproducible per-anchor
    signal of the Results 'Per-anchor structure' paragraph."""
    return pair_badness.mean(axis=-1)
