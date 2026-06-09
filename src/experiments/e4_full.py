"""E4: Full rep M1/M2 over depth (widens gap beyond Mixed). -> e4_full.csv."""
import csv

import torch

from src.depths import depth_to_str
from src.experiments.base import Experiment
from src.metrics import M1, M2

_QS = torch.tensor([0.05, 0.25, 0.75, 0.95])
_FIELDS = ["t", "M1_mean", "M1_q5", "M1_q25", "M1_q75", "M1_q95",
           "M2_mean", "M2_q5", "M2_q25", "M2_q75", "M2_q95"]


class E4Full(Experiment):
    name = "e4"

    def run(self):
        self._ensure_results_dir()
        depth_grid = self.configs.eval.depth_grid
        print(f"Seeds: {self.seeds} | Pairs: {len(self.pairs)} | Depths: {len(depth_grid)}")

        rows = []
        for t in depth_grid:
            reps = {s: self.full_rep(s, t) for s in self.seeds}
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

        with open(self.results_dir / "e4_full.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=_FIELDS)
            w.writeheader()
            w.writerows(rows)
        print("Saved e4_full.csv")
