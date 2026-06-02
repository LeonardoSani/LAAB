from dataclasses import dataclass, field

import torch
import torch.nn as nn


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
def compute_attractors(
    encoder: nn.Module,
    decoder: nn.Module,
    z0: torch.Tensor,
    tol: float = 1e-6,
    max_iter: int = 3000,
    track_residuals: bool = False,
) -> tuple[torch.Tensor, ConvergenceStats]:
    """
    Iterate z_{t+1} = E(D(z_t)) until convergence (per NLSD Appendix D).

    Convergence criterion per sample: ||z_{t+1} - z_t||_2^2 < tol
    Stops early when ALL samples converged or max_iter reached.

    Args:
        encoder:          frozen encoder E_s
        decoder:          frozen decoder D_s
        z0:               (N, k) initial latent points (one per anchor source)
        tol:              per-sample squared L2 threshold (default 1e-6)
        max_iter:         hard cap on iterations (default 3000)
        track_residuals:  if True, record mean L2 norm per iteration in stats.residuals

    Returns:
        attractors:  (N, k) converged latent points
        stats:       ConvergenceStats with diagnostics (and residuals if tracked)
    """
    encoder.eval()
    decoder.eval()

    z = z0.clone()
    converged = torch.zeros(z.size(0), dtype=torch.bool, device=z.device)
    iters     = torch.zeros(z.size(0), dtype=torch.long,  device=z.device)
    residuals: list[float] = []

    for t in range(1, max_iter + 1):
        z_next = encoder(decoder(z))
        delta_sq = ((z_next - z) ** 2).sum(dim=1)   # (N,) squared L2 per sample

        if track_residuals:
            # Mean L2 norm across all (not yet converged) samples
            residuals.append(delta_sq.sqrt().mean().item())

        newly_converged = (~converged) & (delta_sq < tol)
        iters[newly_converged] = t
        converged |= newly_converged
        z = z_next

        if converged.all():
            break

    iters[~converged] = max_iter

    stats = ConvergenceStats(
        n_total=z.size(0),
        n_converged=int(converged.sum().item()),
        mean_iters=float(iters.float().mean().item()),
        max_iters=int(iters.max().item()),
        residuals=residuals,
    )
    return z, stats
