from typing import Union

import torch
import torch.nn as nn

DepthLike = Union[int, float]  # int for finite, float('inf') for attractor


@torch.no_grad()
def iterate_with_snapshots(
    encoder: nn.Module,
    decoder: nn.Module,
    z0: torch.Tensor,
    depths: list[DepthLike],
    tol: float = 1e-6,
    max_iter: int = 3000,
) -> dict[DepthLike, torch.Tensor]:
    """
    Iterate z_{t+1} = E(D(z_t)) and return snapshots at each requested depth.

    Finite t: snapshot at exactly t iterations. float('inf'): iterate until
    per-sample convergence (||z_{t+1} - z_t||^2 < tol) or max_iter.
    Snapshots are CPU tensors of shape (z0.shape[0], k).
    Depth 0 is z0 (no iterations applied).
    """
    encoder.eval()
    decoder.eval()

    finite_depths = sorted(d for d in depths if d != float("inf"))
    has_inf = float("inf") in depths
    finite_set = set(finite_depths)
    max_finite = max(finite_depths) if finite_depths else 0
    n_stop = max_iter if has_inf else max_finite

    snapshots: dict[DepthLike, torch.Tensor] = {}
    z = z0.clone()

    if 0 in finite_set:
        snapshots[0] = z.cpu().clone()

    converged = torch.zeros(z.size(0), dtype=torch.bool, device=z.device)

    for t in range(1, n_stop + 1):
        z_next = encoder(decoder(z))

        if t in finite_set:
            snapshots[t] = z_next.cpu().clone()

        if has_inf:
            delta_sq = ((z_next - z) ** 2).sum(dim=1)
            newly_converged = (~converged) & (delta_sq < tol)
            converged |= newly_converged
            if converged.all() and t >= max_finite:
                z = z_next
                break

        z = z_next

    if has_inf:
        snapshots[float("inf")] = z.cpu().clone()

    return snapshots
