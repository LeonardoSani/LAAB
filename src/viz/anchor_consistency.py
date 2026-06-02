import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.analysis.consistency import disjoint_half_means


def plot_anchor_consistency(
    slice_inf: np.ndarray,   # (N, n_pairs) badness at t=inf
    mu: np.ndarray,          # (N,) per-anchor mean badness
    summary: dict,           # split-half stats (from e2_summary.csv)
    save_path: Path,
    baseline: dict = None,   # t=0 per-anchor badness range {min,median,max}
) -> None:
    """Figure 5 (E2): do consistently good/bad anchors exist?
    (a) sorted mu_i with per-pair IQR band — shows the spread of per-anchor
    badness; (b) split-half reliability scatter — mu_i computed on two
    disjoint halves of the seed pairs, showing the ranking is reproducible
    despite weak single-pair consistency (Spearman-Brown upgrades r_half to
    the reliability of the full-set mu_i)."""
    N, n_pairs = slice_inf.shape
    q25, q75 = np.percentile(slice_inf, [25, 75], axis=1)
    order = np.argsort(mu)[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # (a) sorted mu with per-pair IQR band
    ax = axes[0]
    xs = np.arange(N)
    ax.plot(xs, mu[order], color="C0", lw=1.5)
    ax.fill_between(xs, q25[order], q75[order], alpha=0.25, color="C0",
                    label="per-pair IQR")
    ax.axhline(np.median(mu), color="gray", ls=":",
               label=f"median={np.median(mu):.3f}")
    if baseline is not None:
        # t=0 alignment reference: even the best anchor sits far above it
        ax.axhspan(baseline["min"], baseline["max"], color="green", alpha=0.18,
                   label=f"t=0 baseline (≤{baseline['max']:.3f})")
        ax.set_ylim(bottom=0)
    ax.set_xlabel(r"anchor rank (by $\mu_i$ desc)")
    ax.set_ylabel(r"badness $\mu_i$ (mean over pairs)")
    ax.set_title("(a) Per-anchor badness profile")
    ax.legend(fontsize=8)

    # (b) split-half reliability scatter
    ax = axes[1]
    a, b = disjoint_half_means(slice_inf, seed=0)
    ax.scatter(a, b, s=10, alpha=0.5, color="C3")
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel(r"$\mu_i$ (half A of pairs)")
    ax.set_ylabel(r"$\mu_i$ (half B of pairs)")
    ax.set_title(f"(b) Split-half reliability\n"
                 f"$r_{{half}}$={summary['pearson_half_mean']:.2f}, "
                 f"Spearman-Brown={summary['spearman_brown_full']:.2f}")

    fig.suptitle(r"E2 — Are some anchors consistently bad? (Mixed, $t=\infty$)",
                 fontsize=13)
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path.with_suffix(".png"), dpi=150)
    plt.close(fig)
