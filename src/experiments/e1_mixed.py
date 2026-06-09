"""E1: Mixed-alignment breakdown over depth. -> e1_mixed.csv (M1/M2, quantiles)."""
import csv

import torch

from src.depths import depth_to_str
from src.experiments.base import Experiment
from src.metrics import M1, M2

_QS = torch.tensor([0.05, 0.25, 0.75, 0.95])
_FIELDS = ["t", "M1_mean", "M1_q5", "M1_q25", "M1_q75", "M1_q95",
           "M2_mean", "M2_q5", "M2_q25", "M2_q75", "M2_q95"]


class E1Mixed(Experiment):
    name = "e1"

    def run(self, run_decomposition: bool = False):
        self._ensure_results_dir()
        depth_grid = self.configs.eval.depth_grid
        print(f"Seeds: {self.seeds} | Pairs: {len(self.pairs)} | Depths: {len(depth_grid)}")

        rows = []
        for t in depth_grid:
            reps = {s: self.mixed_rep(s, t) for s in self.seeds}
            m1_vals, m2_vals = [], []
            for s, sp in self.pairs:
                m1_vals.append(M1(reps[s], reps[sp]))
                m2_vals.append(M2(reps[s], reps[sp]))
            m1_all, m2_all = torch.cat(m1_vals), torch.cat(m2_vals)
            m1q, m2q = m1_all.quantile(_QS), m2_all.quantile(_QS)
            rows.append({
                "t": depth_to_str(t),
                "M1_mean": float(m1_all.mean()),
                "M1_q5": float(m1q[0]), "M1_q25": float(m1q[1]),
                "M1_q75": float(m1q[2]), "M1_q95": float(m1q[3]),
                "M2_mean": float(m2_all.mean()),
                "M2_q5": float(m2q[0]), "M2_q25": float(m2q[1]),
                "M2_q75": float(m2q[2]), "M2_q95": float(m2q[3]),
            })
            r = rows[-1]
            print(f"  t={r['t']:>6}: M1={r['M1_mean']:.4f} [q5={r['M1_q5']:.4f} "
                  f"q95={r['M1_q95']:.4f}]  M2={r['M2_mean']:.4f} "
                  f"[q5={r['M2_q5']:.4f} q95={r['M2_q95']:.4f}]")

        with open(self.results_dir / "e1_mixed.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=_FIELDS)
            w.writeheader()
            w.writerows(rows)
        print("Saved e1_mixed.csv")

        if run_decomposition:
            print("\nRunning decomposition as requested (--decomposition / -d)")
            self.run_decomposition()
    def run_decomposition(self):
        """M1^2 = norm-difference + direction terms over depth (prints a table)."""
        import itertools

        depths = self.configs.eval.depth_grid
        store = self.anchors
        seeds = [s for s in self.seeds if store.has_data(s, 0)]

        reps = self.reps

        print(f"{'t':>5} {'mean|r|':>8} {'M1^2':>9} {'mean|d|r||':>10} {'M2':>9} {'normdiff%':>9} {'direction%':>10}")
        for t in depths:
            rep_t = {s: reps.mixed(s, t) for s in seeds}
            nd_tot = dir_tot = m1sq_tot = m2_tot = rnorm_tot = ndabs_tot = 0.0
            n = 0
            for s, s2 in itertools.combinations(seeds, 2):
                r, rp = rep_t[s], rep_t[s2]
                nr, nrp = r.norm(dim=1), rp.norm(dim=1)
                cos = (r * rp).sum(1) / (nr * nrp).clamp_min(1e-12)
                nd_tot += (nr - nrp).pow(2).sum().item()
                ndabs_tot += (nr - nrp).abs().sum().item()
                dir_tot += (2 * nr * nrp * (1 - cos)).sum().item()
                m1sq_tot += (r - rp).pow(2).sum(1).sum().item()
                m2_tot += (1 - cos).sum().item()
                rnorm_tot += nr.sum().item()
                n += r.shape[0]
            tlab = "inf" if t == float("inf") else int(t)
            print(f"{str(tlab):>5} {rnorm_tot/n:8.3f} {m1sq_tot/n:9.4f} {ndabs_tot/n:10.4f} "
                  f"{m2_tot/n:9.5f} {100*nd_tot/m1sq_tot:9.1f} {100*dir_tot/m1sq_tot:10.1f}")
