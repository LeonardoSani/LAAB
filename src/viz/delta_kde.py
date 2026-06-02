import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import gaussian_kde

_KDE_MAX_SAMPLES = 100_000  # gaussian_kde is O(n) per eval point; cap for speed


def _kde_subsample(vals: np.ndarray) -> np.ndarray:
    if len(vals) > _KDE_MAX_SAMPLES:
        rng = np.random.default_rng(0)
        return rng.choice(vals, size=_KDE_MAX_SAMPLES, replace=False)
    return vals


def plot_delta_kde_single(
    delta_vals: torch.Tensor,  # flat pooled M3 values from Pi_delta^0
    save_path: Path,
    log_scale: bool = False,
    annotate_mean: bool = True,
) -> None:
    """Figure 1 (E0): single-panel KDE of pooled delta_i at t=0."""
    vals = delta_vals.numpy().astype(np.float64).ravel()
    kde = gaussian_kde(_kde_subsample(vals), bw_method="scott")
    x = np.linspace(vals.min(), vals.max(), 512)
    y = kde(x)

    fig, ax = plt.subplots(figsize=(5, 4))
    if log_scale:
        ax.semilogy(x, y)
    else:
        ax.plot(x, y)
    ax.fill_between(x, y, alpha=0.3)

    if annotate_mean:
        mu = float(vals.mean())
        ax.axvline(mu, color="red", linestyle="--", label=f"mean={mu:.4f}")
        ax.legend(fontsize=9)

    ax.set_xlabel(r"$\delta_i$")
    ax.set_ylabel("log density" if log_scale else "density")
    ax.set_title(r"$\Pi_\delta^0$ — Base representation ($t=0$)")
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path.with_suffix(".png"), dpi=150)
    plt.close(fig)


def plot_delta_kde_grid(
    delta_per_t: dict,  # {t: flat tensor} for t in {0, 8, 64, 256, 512, inf}
    save_path: Path,
) -> None:
    """Figure 4 (E2): 6-panel KDE grid (shared linear y-axis), 3 rows × 2 cols.
    Panels: t=0 (baseline), 8, 64, 256, 512, inf.
    Each panel: KDE with dashed verticals at q50 and q95. Bandwidth: Scott's rule."""
    t_keys = list(delta_per_t.keys())
    n = len(t_keys)
    ncols = 2
    nrows = (n + 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, nrows * 3.5))
    axes = axes.ravel()

    # Common x-axis range across all panels
    all_vals = [delta_per_t[t].numpy().astype(np.float64).ravel() for t in t_keys]
    x_min = min(v.min() for v in all_vals)
    x_max = max(v.max() for v in all_vals)

    # Precompute KDE curves to derive a common (linear) y-axis range.
    # Linear density keeps the distribution shape visible; a log axis with
    # fill_between fills to ~0 and turns broad distros into solid blocks.
    x = np.linspace(x_min, x_max, 512)
    ys = [gaussian_kde(_kde_subsample(vals), bw_method="scott")(x) for vals in all_vals]
    # Shared y-cap = 2x median per-panel peak. Keeps broad late-t humps fully
    # visible; tall narrow early-t spikes (t=0, 8) clip at the top edge.
    peaks = [y.max() for y in ys]
    y_top = float(np.median(peaks)) * 2.0

    for ax, t, vals, y in zip(axes, t_keys, all_vals, ys):
        ax.plot(x, y)
        ax.fill_between(x, y, alpha=0.3)
        q50 = float(np.quantile(vals, 0.50))
        q95 = float(np.quantile(vals, 0.95))
        ax.axvline(q50, color="orange", linestyle="--", label=f"q50={q50:.3f}")
        ax.axvline(q95, color="red", linestyle="--", label=f"q95={q95:.3f}")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, y_top)
        label = r"$t=\infty$" if t == float("inf") else f"$t={t}$"
        ax.set_title(label)
        ax.set_xlabel(r"$\delta_i$")
        ax.set_ylabel("density")
        ax.legend(fontsize=8)

    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle(r"Per-anchor mismatch $\delta_i$ distribution (Mixed)")
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path.with_suffix(".png"), dpi=150)
    plt.close(fig)
