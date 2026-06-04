"""The single E o D fixed-point iteration, with depth snapshots and optional
convergence diagnostics.

z_{t+1} = E(D(z_t)) is iterated once; callers pick what to read out:
  - snapshots at requested depths (finite t and/or float('inf')),
  - a ConvergenceStats summary (set return_stats=True).

Convergence is detected per sample (||z_{t+1} - z_t||^2 < tol) but never freezes
a sample: the update is applied to the whole batch every step; `converged` only
governs when the loop may stop. This is the one place the dynamics live —
compute_attractors() in iteration.py is a thin wrapper over the inf-only call.
"""
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from src.depths import DepthLike


@dataclass
class ConvergenceStats:
    n_total: int
    n_converged: int
    mean_iters: float
    max_iters: int
    # Per-iteration mean L2 norm ||z_{t+1} - z_t||_2 across all samples.
    # Populated only when track_residuals=True; empty list otherwise.
    residuals: list[float] = field(default_factory=list)

    def __str__(self):
        pct = 100 * self.n_converged / self.n_total
        return (
            f"Converged: {self.n_converged}/{self.n_total} ({pct:.1f}%) | "
            f"Iters mean={self.mean_iters:.1f} max={self.max_iters}"
        )


@torch.no_grad()
def iterate_with_snapshots(
    encoder: nn.Module,
    decoder: nn.Module,
    z0: torch.Tensor,
    depths: list[DepthLike],
    tol: float = 1e-6,
    max_iter: int = 3000,
    return_stats: bool = False,
    track_residuals: bool = False,
):
    """
    Iterate z_{t+1} = E(D(z_t)) and return snapshots at each requested depth.

    Finite t: snapshot at exactly t iterations. float('inf'): iterate until
    per-sample convergence (||z_{t+1} - z_t||^2 < tol) or max_iter.
    Snapshots are CPU tensors of shape (z0.shape[0], k). Depth 0 is z0.

    return_stats: also return a ConvergenceStats (requires float('inf') in
    depths so the convergence loop runs). track_residuals: record the mean L2
    residual per iteration into the stats.

    Returns the snapshots dict, or (snapshots, stats) when return_stats=True.
    """
    encoder.eval()
    decoder.eval()

    finite_depths = sorted(d for d in depths if d != float("inf"))
    has_inf = float("inf") in depths
    finite_set = set(finite_depths)
    max_finite = max(finite_depths) if finite_depths else 0
    n_stop = max_iter if has_inf else max_finite

    if return_stats and not has_inf:
        raise ValueError("return_stats=True requires float('inf') in depths")

    snapshots: dict[DepthLike, torch.Tensor] = {}
    z = z0.clone()

    if 0 in finite_set:
        snapshots[0] = z.cpu().clone()

    converged = torch.zeros(z.size(0), dtype=torch.bool, device=z.device)
    iters = torch.zeros(z.size(0), dtype=torch.long, device=z.device)
    residuals: list[float] = []

    for t in range(1, n_stop + 1):
        z_next = encoder(decoder(z))

        if t in finite_set:
            snapshots[t] = z_next.cpu().clone()

        if has_inf:
            delta_sq = ((z_next - z) ** 2).sum(dim=1)
            if track_residuals:
                residuals.append(delta_sq.sqrt().mean().item())
            newly_converged = (~converged) & (delta_sq < tol)
            iters[newly_converged] = t
            converged |= newly_converged
            if converged.all() and t >= max_finite:
                z = z_next
                break

        z = z_next

    if has_inf:
        snapshots[float("inf")] = z.cpu().clone()

    if not return_stats:
        return snapshots

    iters[~converged] = max_iter
    stats = ConvergenceStats(
        n_total=z.size(0),
        n_converged=int(converged.sum().item()),
        mean_iters=float(iters.float().mean().item()),
        max_iters=int(iters.max().item()),
        residuals=residuals,
    )
    return snapshots, stats
