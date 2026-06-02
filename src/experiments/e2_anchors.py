"""E2: per-anchor mismatch under the attractor dynamics (the consolidated E2).

Outputs:
  e2_delta.csv          M3 mean + quantiles vs depth        (fig3)
  e2_m3_pooled_t*.pt    pooled M3 at select depths          (fig4)
  e2_anchor_badness.npy per-anchor badness cube (N,pairs,T)
  e2_consistency.csv    per-anchor M5 (digit, mu, std)      (fig5)
  e2_summary.csv        split-half reliability of M5        (fig5)
  e2_class_tails.csv    digit-class composition of tails    (fig6)
"""
import csv

import numpy as np
import torch

from src.analysis.consistency import class_tail_counts, split_half_reliability
from src.data.mnist import anchor_class_labels
from src.depths import depth_to_str
from src.experiments.base import Experiment
from src.metrics import M3, per_pair_badness

_KDE_DEPTHS = {float("inf"), 8, 64, 256, 512}
_QS = [0.05, 0.25, 0.50, 0.75, 0.95]
_DELTA_FIELDS = ["t", "M3_mean", "M3_q5", "M3_q25", "M3_q50", "M3_q75", "M3_q95"]


class E2Anchors(Experiment):
    name = "e2"

    def run(self):
        self._ensure_results_dir()
        depth_grid = self.configs.eval.depth_grid
        N = self.configs.ae.latent_dim
        n_pairs, T = len(self.pairs), len(depth_grid)
        print(f"Seeds: {self.seeds} | Pairs: {n_pairs} | Depths: {T} | N={N}")

        cube = np.zeros((N, n_pairs, T), dtype=np.float32)
        delta_rows = []

        for k, t in enumerate(depth_grid):
            reps = {s: self.mixed_rep(s, t) for s in self.seeds}
            m3_parts = []
            for p_idx, (s, sp) in enumerate(self.pairs):
                m3_parts.append(M3(reps[s], reps[sp]).flatten())
                cube[:, p_idx, k] = per_pair_badness(reps[s], reps[sp]).numpy()

            m3_pooled = torch.cat(m3_parts)
            # exact quantiles over the full pool (deterministic; numpy has no
            # element cap, unlike torch.quantile)
            q5, q25, q50, q75, q95 = (float(v) for v in np.quantile(m3_pooled.numpy(), _QS))
            delta_rows.append({
                "t": depth_to_str(t), "M3_mean": float(m3_pooled.mean()),
                "M3_q5": q5, "M3_q25": q25, "M3_q50": q50, "M3_q75": q75, "M3_q95": q95,
            })
            print(f"  t={depth_to_str(t):>6}: M3 mean={delta_rows[-1]['M3_mean']:.4f} q95={q95:.4f}")

            if t in _KDE_DEPTHS:
                torch.save(m3_pooled, self.results_dir / f"e2_m3_pooled_t{depth_to_str(t)}.pt")

        with open(self.results_dir / "e2_delta.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=_DELTA_FIELDS)
            w.writeheader()
            w.writerows(delta_rows)
        np.save(self.results_dir / "e2_anchor_badness.npy", cube)

        self._consistency(cube, depth_grid, N)
        print("Saved e2_delta.csv, e2_anchor_badness.npy, e2_m3_pooled_t*.pt, "
              "e2_consistency.csv, e2_summary.csv, e2_class_tails.csv")

    def _consistency(self, cube, depth_grid, N):
        """M5 split-half reliability + digit-class tails at t=inf."""
        sl = cube[:, :, depth_grid.index(float("inf"))]   # (N, n_pairs)
        mu, sd = sl.mean(1), sl.std(1)
        labels = anchor_class_labels(N, n_classes=10).numpy()
        rel = split_half_reliability(sl)
        print(f"Split-half Pearson r: {rel['pearson_half_mean']:.3f} ± "
              f"{rel['pearson_half_std']:.3f} | Spearman-Brown: "
              f"{rel['spearman_brown_full']:.3f} | mu range "
              f"[{mu.min():.4f}, {mu.max():.4f}]")

        with open(self.results_dir / "e2_consistency.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["anchor", "digit", "mu", "std"])
            for i in range(N):
                w.writerow([i, int(labels[i]), f"{mu[i]:.6f}", f"{sd[i]:.6f}"])

        summary = {
            "n_anchors": N, "n_pairs": len(self.pairs),
            "pearson_half_mean": rel["pearson_half_mean"],
            "pearson_half_std": rel["pearson_half_std"],
            "spearman_half_mean": rel["spearman_half_mean"],
            "spearman_brown_full": rel["spearman_brown_full"],
            "mu_min": float(mu.min()), "mu_median": float(np.median(mu)),
            "mu_max": float(mu.max()),
        }
        with open(self.results_dir / "e2_summary.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["metric", "value"])
            for kk, vv in summary.items():
                w.writerow([kk, vv])

        rows = class_tail_counts(mu, labels)
        with open(self.results_dir / "e2_class_tails.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["class", "n_class",
                                              "good10", "good20", "bad10", "bad20"])
            w.writeheader()
            w.writerows(rows)
