"""E5: basin-of-attraction consistency across seeds.

Tests whether the breakdown of cross-seed alignment is mirrored in the
attractor *basins*. For each anchor, the set of test points that converge to
its attractor is compared across independently trained seeds (Jaccard, with a
size-matched null), giving two readouts:
  - Global: are the basins shared across seeds, or seed-specific?
  - Per-anchor: does basin (in)consistency explain the per-anchor badness mu_i
    (reused from E2's badness cube)?

Outputs:
  e5_basins.csv         summary scalars (basin sizes, Jaccard vs null, label
                        agreement vs null, correlation with mu_i)
  e5_basin_jaccard.npy  per-anchor stack [J, J_null, size, mu]  (4, N) — fig11
"""
import csv

import numpy as np
from scipy.stats import pearsonr, spearmanr

from src.analysis.basins import (basin_masks, cross_seed_jaccard,
                                  label_agreement, nearest_anchor_labels)
from src.experiments.base import Experiment
from src.metrics import anchor_badness

_TAU = 0.99


class E5Basins(Experiment):
    name = "e5"

    def run(self, tau: float = _TAU):
        self._ensure_results_dir()
        t = float("inf")
        N = self.configs.ae.latent_dim
        print(f"Seeds: {self.seeds} | Pairs: {len(self.pairs)} | tau={tau} | t=inf")

        masks, labels = {}, {}
        for s in self.seeds:
            phi_d = self.anchors.phi_data(s, t)
            phi_a = self.anchors.phi_anchors(s, t)
            masks[s] = basin_masks(phi_d, phi_a, tau)
            labels[s], _ = nearest_anchor_labels(phi_d, phi_a)
        n_data = masks[self.seeds[0]].shape[0]

        jac = cross_seed_jaccard(masks, self.pairs)
        J, J_null, size = jac["J"], jac["J_null"], jac["size"]
        raw_ag, null_ag = label_agreement(labels, self.pairs, N)

        mu = self._mu_inf()
        sp_inc = spearmanr(1 - J, mu)
        pe_inc = pearsonr(1 - J, mu)

        summary = {
            "tau": tau, "n_seeds": len(self.seeds), "n_pairs": len(self.pairs),
            "n_data": n_data,
            "basin_size_median": float(np.median(size)),
            "basin_size_mean": float(size.mean()),
            "frac_singleton": float((size <= 1).mean()),
            "jaccard_mean": float(J.mean()),
            "jaccard_median": float(np.median(J)),
            "jaccard_null_mean": float(J_null.mean()),
            "label_agreement_raw": raw_ag,
            "label_agreement_null": null_ag,
            "spearman_incons_mu": float(sp_inc.correlation),
            "spearman_incons_mu_p": float(sp_inc.pvalue),
            "pearson_incons_mu": float(pe_inc[0]),
        }
        with open(self.results_dir / "e5_basins.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["metric", "value"])
            for k, v in summary.items():
                w.writerow([k, v])
        np.save(self.results_dir / "e5_basin_jaccard.npy",
                np.stack([J, J_null, size, mu]).astype(np.float32))

        print(f"  basin size: median={summary['basin_size_median']:.0f} "
              f"mean={summary['basin_size_mean']:.1f} "
              f"singleton={summary['frac_singleton']:.2f}")
        print(f"  Jaccard mean={summary['jaccard_mean']:.3f} "
              f"(null {summary['jaccard_null_mean']:.3f}) | "
              f"label agreement raw={raw_ag:.3f} (null {null_ag:.3f})")
        print(f"  corr(1-J, mu): Spearman={summary['spearman_incons_mu']:+.3f} "
              f"(p={summary['spearman_incons_mu_p']:.2g})")
        print("Saved e5_basins.csv, e5_basin_jaccard.npy")

    def _mu_inf(self) -> np.ndarray:
        """Per-anchor badness mu_i at t=inf, from the E2 badness cube."""
        cube = np.load(self.results_dir / "e2_anchor_badness.npy")  # (N, pairs, T)
        idx = self.configs.eval.depth_grid.index(float("inf"))
        return anchor_badness(cube[:, :, idx])
