"""Attractor computation: iterate z_{t+1} = E(D(z_t)) to convergence.

Thin wrapper over the single dynamics loop in snapshot.py (the inf-only case).
ConvergenceStats is re-exported here for backward compatibility with callers
that import it from this module.
"""
import torch
import torch.nn as nn

from src.attractors.snapshot import ConvergenceStats, iterate_with_snapshots

__all__ = ["ConvergenceStats", "compute_attractors"]


@torch.no_grad()
def compute_attractors(
    encoder: nn.Module,
    decoder: nn.Module,
    z0: torch.Tensor,
    tol: float = 1e-6,
    max_iter: int = 3000,
    track_residuals: bool = False,
) -> tuple[torch.Tensor, ConvergenceStats]:
    """
    Iterate z_{t+1} = E(D(z_t)) until per-sample convergence
    (||z_{t+1} - z_t||_2^2 < tol) or max_iter (per NLSD Appendix D).

    Returns the (N, k) converged latent points and a ConvergenceStats summary.
    """
    snapshots, stats = iterate_with_snapshots(
        encoder, decoder, z0, depths=[float("inf")], tol=tol, max_iter=max_iter,
        return_stats=True, track_residuals=track_residuals)
    return snapshots[float("inf")], stats
