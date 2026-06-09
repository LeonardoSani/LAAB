"""Basin consistency across seeds. x in anchor i's basin iff
cos(phi^inf(x), phi^inf(x_i)) > tau; cross-seed Jaccard J_i tests sharing."""
import numpy as np
import torch
import torch.nn.functional as F


def basin_masks(phi_data: torch.Tensor, phi_refs: torch.Tensor,
                tau: float, chunk: int | None = None) -> torch.Tensor:
    """Membership for one seed: (|D|, R) bool, cos(phi_data, phi_refs) > tau.
    chunk caps columns per matmul block (None = one product)."""
    D = F.normalize(phi_data.float(), p=2, dim=1)
    R = F.normalize(phi_refs.float(), p=2, dim=1)
    if chunk is None:
        return (D @ R.T) > tau
    out = torch.empty((D.shape[0], R.shape[0]), dtype=torch.bool)
    for j0 in range(0, R.shape[0], chunk):
        j1 = min(j0 + chunk, R.shape[0])
        out[:, j0:j1] = (D @ R[j0:j1].T) > tau
    return out


def _expected_jaccard(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    """Size-matched null Jaccard (independent membership)."""
    eint = a * b / n
    den = a + b - eint
    return np.where(den > 0, eint / np.maximum(den, 1e-12), 1.0)


def cross_seed_jaccard(masks: dict, pairs: list) -> dict:
    """Per-anchor basin Jaccard over seed pairs vs size-matched null.
    masks: {seed: (|D|, N) bool}. Returns (N,) J, J_null, size."""
    seeds = list(masks)
    n, N = next(iter(masks.values())).shape
    J = np.zeros(N)
    J_null = np.zeros(N)
    for s, sp in pairs:
        Ma = masks[s].numpy()
        Mb = masks[sp].numpy()
        inter = (Ma & Mb).sum(0).astype(np.float64)
        uni = (Ma | Mb).sum(0).astype(np.float64)
        J += np.where(uni > 0, inter / np.maximum(uni, 1.0), 1.0)
        J_null += _expected_jaccard(Ma.sum(0).astype(np.float64),
                                    Mb.sum(0).astype(np.float64), n)
    sizes = np.mean([masks[s].numpy().sum(0) for s in seeds], axis=0)
    return {"J": J / len(pairs), "J_null": J_null / len(pairs), "size": sizes}


