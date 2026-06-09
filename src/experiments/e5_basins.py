"""E5: cross-seed basin consistency (the proven cause). Per-anchor Jaccard vs a
size-matched null. -> e5_basins.csv, e5_basin_jaccard.npy ([J,J_null,size,mu])."""
import csv

import numpy as np

from src.analysis.basins import basin_masks, cross_seed_jaccard
from src.experiments.base import Experiment
from src.metrics import anchor_badness


class E5Basins(Experiment):
    name = "e5"

    def run(self, tau: float | None = None):
        return self._run_anchor_basins(tau=tau, write_results=True)

    def run_all_data_check(self, tau: float | None = None):
        return self._run_all_data_check(tau=tau)

    def _run_anchor_basins(self, tau: float | None = None, write_results: bool = True):
        if tau is None:
            tau = self.configs.attractors.tau
        self._ensure_results_dir()
        t = float("inf")
        print(f"Seeds: {self.seeds} | Pairs: {len(self.pairs)} | tau={tau} | t=inf")

        masks = {}
        for s in self.seeds:
            phi_d = self.anchors.phi_data(s, t)
            phi_a = self.anchors.phi_anchors(s, t)
            masks[s] = basin_masks(phi_d, phi_a, tau)
        n_data = masks[self.seeds[0]].shape[0]

        jac = cross_seed_jaccard(masks, self.pairs)
        J, J_null, size = jac["J"], jac["J_null"], jac["size"]

        mu = self._mu_inf()

        summary = {
            "tau": tau, "n_seeds": len(self.seeds), "n_pairs": len(self.pairs),
            "n_data": n_data,
            "basin_size_median": float(np.median(size)),
            "basin_size_mean": float(size.mean()),
            "frac_singleton": float((size <= 1).mean()),
            "jaccard_mean": float(J.mean()),
            "jaccard_median": float(np.median(J)),
            "jaccard_null_mean": float(J_null.mean()),
        }
        if write_results:
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
              f"(null {summary['jaccard_null_mean']:.3f})")
        if write_results:
            print("Saved e5_basins.csv, e5_basin_jaccard.npy")

    def _run_all_data_check(self, tau: float | None = None):
        if tau is None:
            tau = self.configs.attractors.tau
        t = float("inf")
        print(f"Seeds: {self.seeds} | Pairs: {len(self.pairs)} | tau={tau}")

        phi_d = {s: self.anchors.phi_data(s, t) for s in self.seeds}
        phi_a = {s: self.anchors.phi_anchors(s, t) for s in self.seeds}
        n_data = phi_d[self.seeds[0]].shape[0]
        print(f"|D|={n_data}  N_anchors={phi_a[self.seeds[0]].shape[0]}")

        masks_anchor = {s: basin_masks(phi_d[s], phi_a[s], tau) for s in self.seeds}
        self._summarize_masks(masks_anchor, self.pairs, "anchors (N=256)")

        masks_all = {}
        for s in self.seeds:
            M = basin_masks(phi_d[s], phi_d[s], tau)
            M.fill_diagonal_(False)
            masks_all[s] = M
        self._summarize_masks(masks_all, self.pairs, "all of D (diagonal removed)")

    def _summarize_masks(self, masks: dict, pairs: list, label: str):
        jac = cross_seed_jaccard(masks, pairs)
        J, J_null, size = jac["J"], jac["J_null"], jac["size"]
        d = J - J_null
        frac_above = float((J > J_null).mean())
        print(f"\n[{label}]  refs={len(size)}")
        print(f"  basin size:   mean={size.mean():.1f}  median={np.median(size):.0f}"
              f"  singleton_frac={(size <= 1).mean():.3f}")
        print(f"  Jaccard  mean={J.mean():.4f}  median={np.median(J):.4f}")
        print(f"  null     mean={J_null.mean():.4f}  median={np.median(J_null):.4f}")
        print(f"  J-null   mean={d.mean():+.4f}  median={np.median(d):+.4f}")
        print(f"  frac(J>null)={frac_above:.3f}  median(J/null)="
              f"{np.median(J[J_null > 0] / J_null[J_null > 0]):.2f}x")

    def _mu_inf(self) -> np.ndarray:
        """mu_i at t=inf, from the E2 badness cube."""
        cube = np.load(self.results_dir / "e2_anchor_badness.npy")
        idx = self.configs.eval.depth_grid.index(float("inf"))
        return anchor_badness(cube[:, :, idx])
