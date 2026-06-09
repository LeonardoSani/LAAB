"""E0: reps agree across seeds at t=0. -> e0_baseline.csv, e0_m3_pooled.pt."""
import csv

import torch

from src.experiments.base import Experiment
from src.metrics import M1, M2, M3


class E0Baseline(Experiment):
    name = "e0"

    def run(self):
        self._ensure_results_dir()
        print(f"Seeds: {self.seeds} | Pairs: {len(self.pairs)} | Device: {self.device}")

        reps = {s: self.base_rep(s) for s in self.seeds}

        m1_vals, m2_vals, m3_parts = [], [], []
        for s, sp in self.pairs:
            m1_vals.append(M1(reps[s], reps[sp]))
            m2_vals.append(M2(reps[s], reps[sp]))
            m3_parts.append(M3(reps[s], reps[sp]).flatten())

        m1_all = torch.cat(m1_vals)
        m2_all = torch.cat(m2_vals)
        m3_pooled = torch.cat(m3_parts)

        m1_mean, m1_std = float(m1_all.mean()), float(m1_all.std())
        m2_mean, m2_std = float(m2_all.mean()), float(m2_all.std())
        print(f"M1: mean={m1_mean:.4f} std={m1_std:.4f}")
        print(f"M2: mean={m2_mean:.4f} std={m2_std:.4f}")

        with open(self.results_dir / "e0_baseline.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["metric", "mean", "std"])
            w.writerow(["M1", f"{m1_mean:.6f}", f"{m1_std:.6f}"])
            w.writerow(["M2", f"{m2_mean:.6f}", f"{m2_std:.6f}"])
        torch.save(m3_pooled, self.results_dir / "e0_m3_pooled.pt")
        print("Saved e0_baseline.csv and e0_m3_pooled.pt")

        assert m2_mean < 0.02, f"E0 acceptance FAIL: M2 mean={m2_mean:.4f} >= 0.02"
        print("Acceptance checks PASSED.")
