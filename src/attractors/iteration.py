"""Attractors via z_{t+1} = E(D(z_t)) to convergence. Thin wrapper over snapshot.py."""
import torch
import torch.nn as nn

from src.attractors.snapshot import ConvergenceStats, iterate_with_snapshots

__all__ = ["ConvergenceStats", "compute_attractors"]


@torch.no_grad()
def compute_attractors(
    encoder: nn.Module,
    decoder: nn.Module,
    z0: torch.Tensor,
    tol: float,
    max_iter: int,
    track_residuals: bool = False,
) -> tuple[torch.Tensor, ConvergenceStats]:
    """Iterate to convergence (||Δz||² < tol) or max_iter.
    Returns (N, k) attractors and ConvergenceStats."""
    snapshots, stats = iterate_with_snapshots(
        encoder, decoder, z0, depths=[float("inf")], tol=tol, max_iter=max_iter,
        return_stats=True, track_residuals=track_residuals)
    return snapshots[float("inf")], stats
