"""E3: zero-shot stitching error M4 over the decoder depths.
Needs trained relative decoders. -> e3_stitching{,_per_seed}.csv."""
import csv
import sys

from src.depths import depth_to_str
from src.experiments.base import Experiment
from src.metrics import M4


class E3Stitching(Experiment):
    name = "e3"

    def run(self):
        self._ensure_results_dir()
        decoder_depths = self.configs.eval.decoder_depths
        ref_seed = self.configs.eval.reference_seed

        missing = [depth_to_str(t) for t in decoder_depths
                   if not self.ckpts.has_relative_decoder(t)]
        if missing:
            print("Missing relative decoders for depths:", missing)
            print("Run: python scripts/train_relative_decoder.py --depth T")
            sys.exit(1)

        print(f"Seeds: {self.seeds} | Decoder depths: {decoder_depths} | ref_seed: {ref_seed}")
        X_d = self.test_images

        rows, per_seed_rows = [], []
        for t in decoder_depths:
            rel_dec = self.ckpts.load_relative_decoder(t)
            rep_per_seed = {s: self.mixed_rep(s, t) for s in self.seeds}
            mean_mse, std_mse, per_seed = M4(rel_dec, rep_per_seed, X_d, ref_seed, self.device)
            rows.append({"t": depth_to_str(t), "mse_mean": mean_mse, "mse_std": std_mse})
            for s, v in per_seed.items():
                per_seed_rows.append({"t": depth_to_str(t), "seed": s, "mse": v})
            print(f"  t={depth_to_str(t):>6}: MSE mean={mean_mse:.4f}±{std_mse:.4f}")

        with open(self.results_dir / "e3_stitching.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["t", "mse_mean", "mse_std"])
            w.writeheader()
            w.writerows(rows)
        with open(self.results_dir / "e3_stitching_per_seed.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["t", "seed", "mse"])
            w.writeheader()
            w.writerows(per_seed_rows)
        print("Saved e3_stitching.csv, e3_stitching_per_seed.csv")
