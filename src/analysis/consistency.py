"""Per-anchor badness ordering: reliability and digit-class split.
Operates on a (N, n_pairs) badness slice."""
import numpy as np
from scipy.stats import pearsonr, spearmanr

from src.metrics import anchor_badness


def split_half_reliability(slice_: np.ndarray, n_repeats: int = 500,
                           seed: int = 0) -> dict:
    """Split-half reliability of mu_i: split pairs, correlate halves,
    Spearman-Brown to full-set reliability."""
    _, n_pairs = slice_.shape
    half = n_pairs // 2
    rng = np.random.default_rng(seed)
    pear, spear = [], []
    for _ in range(n_repeats):
        perm = rng.permutation(n_pairs)
        a, b = perm[:half], perm[half:2 * half]
        mu_a, mu_b = anchor_badness(slice_[:, a]), anchor_badness(slice_[:, b])
        pear.append(pearsonr(mu_a, mu_b)[0])
        spear.append(spearmanr(mu_a, mu_b)[0])
    r_half = float(np.mean(pear))
    return {
        "pearson_half_mean": r_half,
        "pearson_half_std": float(np.std(pear)),
        "spearman_half_mean": float(np.mean(spear)),
        "spearman_brown_full": 2 * r_half / (1 + r_half),
    }


def disjoint_half_means(slice_: np.ndarray, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """One disjoint pair split for the scatter -> (mu_a, mu_b)."""
    _, n_pairs = slice_.shape
    half = n_pairs // 2
    perm = np.random.default_rng(seed).permutation(n_pairs)
    return anchor_badness(slice_[:, perm[:half]]), anchor_badness(slice_[:, perm[half:2 * half]])


def class_tail_counts(mu: np.ndarray, labels: np.ndarray,
                      tails=(("10", 0.10), ("20", 0.20)),
                      n_classes: int = 10) -> list[dict]:
    """Digit-class composition of the best/worst mu tails (10%/20%)."""
    N = len(mu)
    order = np.argsort(mu)
    sets = {}
    for tag, frac in tails:
        k = int(round(N * frac))
        sets[f"good{tag}"] = set(order[:k].tolist())
        sets[f"bad{tag}"] = set(order[-k:].tolist())
    rows = []
    for c in range(n_classes):
        idx_c = set(np.where(labels == c)[0].tolist())
        row = {"class": c, "n_class": len(idx_c)}
        row.update({name: len(idx_c & s) for name, s in sets.items()})
        rows.append(row)
    return rows
