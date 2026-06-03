"""Analysis of basin-of-attraction consistency across seeds.

A test point x is in the basin of anchor i (under seed s) if its attractor
phi_s^inf(x) is cos-aligned (> tau) with the anchor's own attractor
phi_s^inf(x_i). Basins are sets over the shared test-set index space, so the
membership set of anchor i can be compared across seeds: if the same points
fall into anchor i's basin regardless of seed, the basin geometry is aligned;
if not, the dynamics' basin structure is seed-specific.

Operates on the cached attractor tensors (phi_data, phi_anchors at t=inf).
"""
import numpy as np
import torch
import torch.nn.functional as F


def basin_masks(phi_data: torch.Tensor, phi_anchors: torch.Tensor,
                tau: float = 0.99) -> torch.Tensor:
    """Boolean basin-membership matrix for one seed.

    phi_data (|D|, k), phi_anchors (N, k) -> (|D|, N): entry [x, i] is True iff
    cos(phi_data[x], phi_anchors[i]) > tau, i.e. x falls in anchor i's basin.
    """
    D = F.normalize(phi_data.float(), p=2, dim=1)
    A = F.normalize(phi_anchors.float(), p=2, dim=1)
    return (D @ A.T) > tau


def _expected_jaccard(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    """Chance Jaccard for two random subsets of sizes a, b in a universe of n
    under independent membership: E|inter| = a*b/n, E|union| = a + b - a*b/n."""
    eint = a * b / n
    den = a + b - eint
    return np.where(den > 0, eint / np.maximum(den, 1e-12), 1.0)


def cross_seed_jaccard(masks: dict, pairs: list) -> dict:
    """Per-anchor basin Jaccard across seed pairs, with a size-matched null.

    masks: {seed: (|D|, N) bool}. The size-matched null is the Jaccard expected
    if the two basins held the same number of points but membership were
    independent — it controls for the fact that large (collapsed) basins overlap
    a lot by construction. Returns dict of (N,) arrays:
      J      observed mean Jaccard over pairs,
      J_null analytic chance Jaccard (size-matched),
      size   mean basin size over seeds.
    """
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


def nearest_anchor_labels(phi_data: torch.Tensor,
                          phi_anchors: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Hard basin label per point = argmax cos to an anchor attractor, with the
    max cos (attachment strength). phi_data (|D|, k), phi_anchors (N, k) ->
    labels (|D|,), strength (|D|,)."""
    D = F.normalize(phi_data.float(), p=2, dim=1)
    A = F.normalize(phi_anchors.float(), p=2, dim=1)
    strength, labels = (D @ A.T).max(dim=1)
    return labels.numpy(), strength.numpy()


def label_agreement(labels: dict, pairs: list, n_anchors: int) -> tuple[float, float]:
    """Cross-seed agreement of the nearest-anchor basin label. The anchor index
    is the same image across seeds, so labels are directly comparable without
    matching. Returns (raw_mean, null_mean), the null being the marginal-product
    chance agreement sum_i p_s(i) p_s'(i). labels: {seed: (|D|,) int}."""
    n = len(next(iter(labels.values())))
    raw, null = [], []
    for s, sp in pairs:
        la, lb = labels[s], labels[sp]
        raw.append(float((la == lb).mean()))
        pa = np.bincount(la, minlength=n_anchors) / n
        pb = np.bincount(lb, minlength=n_anchors) / n
        null.append(float((pa * pb).sum()))
    return float(np.mean(raw)), float(np.mean(null))
