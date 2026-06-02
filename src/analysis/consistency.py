"""Analysis built on the per-anchor badness M5: is the per-anchor ordering a
reliable signal, and how does it split by digit class?

Operates on a single-depth badness slice of shape (N, n_pairs): row i = anchor
i, column p = per_pair_badness for seed-pair p (i.e. the t=inf slice of the
badness cube).
"""
import numpy as np
from scipy.stats import pearsonr, spearmanr


def split_half_reliability(slice_: np.ndarray, n_repeats: int = 500,
                           seed: int = 0) -> dict:
    """Reliability of mu_i = mean_p per_pair_badness. Per-pair rankings are
    noisy, so repeatedly split the pairs into two disjoint halves, recompute
    mu_i on each, and correlate; Spearman-Brown lifts the half-set correlation
    to the reliability of mu_i from ALL pairs.

    (N, n_pairs) -> dict(pearson_half_mean, pearson_half_std,
                         spearman_half_mean, spearman_brown_full).
    """
    _, n_pairs = slice_.shape
    half = n_pairs // 2
    rng = np.random.default_rng(seed)
    pear, spear = [], []
    for _ in range(n_repeats):
        perm = rng.permutation(n_pairs)
        a, b = perm[:half], perm[half:2 * half]
        mu_a, mu_b = slice_[:, a].mean(1), slice_[:, b].mean(1)
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
    """One deterministic disjoint split of the pairs, for the scatter plot.
    (N, n_pairs) -> (mu_half_a (N,), mu_half_b (N,))."""
    _, n_pairs = slice_.shape
    half = n_pairs // 2
    perm = np.random.default_rng(seed).permutation(n_pairs)
    return slice_[:, perm[:half]].mean(1), slice_[:, perm[half:2 * half]].mean(1)


def class_tail_counts(mu: np.ndarray, labels: np.ndarray,
                      tails=(("10", 0.10), ("20", 0.20)),
                      n_classes: int = 10) -> list[dict]:
    """Digit-class composition of the good/bad badness tails. Ranks anchors by
    mu (ascending = good first), takes the cumulative best/worst fractions, and
    counts per class. Returns rows: {class, n_class, good10, good20, bad10, bad20}."""
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
