#!/usr/bin/env python
"""Emit all paper figures from cached results."""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib

# Non-interactive backend (avoid a dock icon on macOS when importing matplotlib).
os.environ.setdefault("MPLBACKEND", "Agg")
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Configs
from src.data.mnist import get_mnist
from src.store.anchors import AnchorStore
from src.store.checkpoints import CheckpointStore
from src.utils.device import get_device
from src.viz.anchor_consistency import plot_anchor_consistency
from src.viz.basin_consistency import plot_basin_consistency
from src.viz.class_tails import plot_class_tails
from src.viz.delta_curve import plot_delta_curve
from src.viz.delta_kde import plot_delta_kde_grid, plot_delta_kde_single
from src.viz.metric_curves import plot_full_vs_mixed, plot_mixed_curves
from src.viz.reconstruction_grid import plot_reference_grid, plot_stitching_grid
from src.viz.stitching_fig import plot_stitching

DEFAULT_FIGS = "1,2,3,4,5,6,7,8,9,10,11"


class FigureMaker:
    """One method per figure; reads cached result files, loads models via the
    stores only for the reconstruction grids. Each method skips (with a note)
    if its inputs are missing."""

    def __init__(self, results_dir="artifacts/results", figures_dir="figures"):
        self.results = Path(results_dir)
        self.fig_dir = Path(figures_dir)
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        self._stores = None
        self._test_ds = None

    # --- lazy shared resources (only needed by the recon grids) ---
    @property
    def stores(self):
        if self._stores is None:
            configs = Configs.load()
            device = get_device()
            self._stores = (configs, CheckpointStore(configs.ae, device),
                            AnchorStore(), device)
        return self._stores

    @property
    def test_ds(self):
        if self._test_ds is None:
            _, self._test_ds = get_mnist("artifacts/data")
        return self._test_ds

    def _have(self, *names) -> bool:
        return all((self.results / n).exists() for n in names)

    def make(self, nums):
        for n in sorted(nums):
            method = getattr(self, f"fig{n}", None)
            if method is None:
                print(f"Fig {n}: no such figure — skip")
                continue
            method()

    # --- E0 / E1 ---
    def fig1(self):
        if not self._have("e0_m3_pooled.pt"):
            return print("Fig 1: e0_m3_pooled.pt missing — skip")
        vals = torch.load(self.results / "e0_m3_pooled.pt", weights_only=True)
        plot_delta_kde_single(vals, save_path=self.fig_dir / "fig1_e0_delta_kde")
        print("Fig 1 saved.")

    def fig2(self):
        if not self._have("e1_mixed.csv", "e0_baseline.csv"):
            return print("Fig 2: e1_mixed.csv or e0_baseline.csv missing — skip")
        base = pd.read_csv(self.results / "e0_baseline.csv", index_col="metric")
        plot_mixed_curves(
            pd.read_csv(self.results / "e1_mixed.csv"),
            save_path=self.fig_dir / "fig2_mixed_M1M2",
            baseline={"M1_mean": float(base.loc["M1", "mean"]),
                      "M2_mean": float(base.loc["M2", "mean"])},
        )
        print("Fig 2 saved.")

    # --- E2 ---
    def fig3(self):
        if not self._have("e2_delta.csv"):
            return print("Fig 3: e2_delta.csv missing — skip")
        if self._have("e0_m3_pooled.pt"):
            ref = float(torch.load(self.results / "e0_m3_pooled.pt", weights_only=True).mean())
        else:
            ref = 0.006
        plot_delta_curve(pd.read_csv(self.results / "e2_delta.csv"),
                         save_path=self.fig_dir / "fig3_delta_curve", ref_t0_mean=ref)
        print("Fig 3 saved.")

    def fig4(self):
        names = ["e0_m3_pooled.pt"] + [f"e2_m3_pooled_t{t}.pt"
                                       for t in ("8", "64", "256", "512", "inf")]
        if not self._have(*names):
            return print("Fig 4: e0/e2 pooled M3 tensors missing — skip")
        delta_per_t = {0: torch.load(self.results / "e0_m3_pooled.pt", weights_only=True)}
        for t, ts in [(8, "8"), (64, "64"), (256, "256"), (512, "512"), (float("inf"), "inf")]:
            delta_per_t[t] = torch.load(self.results / f"e2_m3_pooled_t{ts}.pt", weights_only=True)
        plot_delta_kde_grid(delta_per_t, save_path=self.fig_dir / "fig4_delta_kde")
        print("Fig 4 saved.")

    def fig5(self):
        if not self._have("e2_anchor_badness.npy", "e2_consistency.csv", "e2_summary.csv"):
            return print("Fig 5: e2_anchor_badness.npy or e2_consistency/summary.csv missing — skip")
        cube = np.load(self.results / "e2_anchor_badness.npy")
        slice_inf = cube[:, :, cube.shape[2] - 1]
        per = pd.read_csv(self.results / "e2_consistency.csv")
        summ = pd.read_csv(self.results / "e2_summary.csv", index_col="metric")["value"]
        mu0 = cube[:, :, 0].mean(axis=1)   # t=0 alignment baseline
        plot_anchor_consistency(
            slice_inf=slice_inf,
            mu=per.sort_values("anchor")["mu"].to_numpy(),
            summary={"pearson_half_mean": float(summ["pearson_half_mean"]),
                     "spearman_brown_full": float(summ["spearman_brown_full"])},
            baseline={"min": float(mu0.min()), "median": float(np.median(mu0)),
                      "max": float(mu0.max())},
            save_path=self.fig_dir / "fig5_anchor_consistency",
        )
        print("Fig 5 saved.")

    def fig6(self):
        if not self._have("e2_class_tails.csv"):
            return print("Fig 6: e2_class_tails.csv missing — skip")
        plot_class_tails(pd.read_csv(self.results / "e2_class_tails.csv"),
                         save_path=self.fig_dir / "fig6_class_tails")
        print("Fig 6 saved.")

    # --- E3 / E4 curves ---
    def fig7(self):
        if not self._have("e3_stitching.csv", "e2_delta.csv"):
            return print("Fig 7: e3_stitching.csv or e2_delta.csv missing — skip")
        plot_stitching(pd.read_csv(self.results / "e3_stitching.csv"),
                       pd.read_csv(self.results / "e2_delta.csv"),
                       save_path=self.fig_dir / "fig7_stitching_mse")
        print("Fig 7 saved.")

    def fig8(self):
        if not self._have("e1_mixed.csv", "e4_full.csv"):
            return print("Fig 8: e1_mixed.csv or e4_full.csv missing — skip")
        plot_full_vs_mixed(pd.read_csv(self.results / "e1_mixed.csv"),
                           pd.read_csv(self.results / "e4_full.csv"),
                           save_path=self.fig_dir / "fig8_full_vs_mixed")
        print("Fig 8 saved.")

    # --- E3 reconstruction grids ---
    def _decoder_depths(self):
        return self.stores[0].eval.decoder_depths

    def _anchor_dict(self, seed):
        _, _, anchors, _ = self.stores
        depths = self._decoder_depths()
        if not all(anchors.has_anchor(seed, t) for t in depths):
            return None
        return {t: anchors.phi_anchors(seed, t) for t in depths}

    def _rel_decoders(self):
        _, ckpts, _, _ = self.stores
        depths = self._decoder_depths()
        if not all(ckpts.has_relative_decoder(t) for t in depths):
            return None
        return {t: ckpts.load_relative_decoder(t) for t in depths}

    def fig9(self):
        _, ckpts, _, device = self.stores
        anchors_s2 = self._anchor_dict(2)
        rel_decoders = self._rel_decoders()
        if anchors_s2 is None:
            return print("Fig 9: phi_anchors_s2_t* missing — skip")
        if rel_decoders is None:
            return print("Fig 9: relative_decoder_M_t* missing — skip")
        plot_stitching_grid(
            ae_s1=ckpts.load_ae(1), encoder_s2=ckpts.load_ae(2).encoder,
            anchors_s2=anchors_s2, rel_decoders=rel_decoders,
            test_ds=self.test_ds, device=device,
            save_path=self.fig_dir / "fig9_recon_stitching",
        )
        print("Fig 9 saved.")

    def fig10(self):
        _, ckpts, _, device = self.stores
        anchors_s1 = self._anchor_dict(1)
        rel_decoders = self._rel_decoders()
        if anchors_s1 is None:
            return print("Fig 10: phi_anchors_s1_t* missing — skip")
        if rel_decoders is None:
            return print("Fig 10: relative_decoder_M_t* missing — skip")
        plot_reference_grid(
            ae_s1=ckpts.load_ae(1), anchors_s1=anchors_s1,
            rel_decoders=rel_decoders, test_ds=self.test_ds, device=device,
            save_path=self.fig_dir / "fig10_recon_reference",
        )
        print("Fig 10 saved.")

    # --- E5 ---
    def fig11(self):
        if not self._have("e5_basin_jaccard.npy"):
            return print("Fig 11: e5_basin_jaccard.npy missing — skip")
        J, J_null, _size, _mu = np.load(self.results / "e5_basin_jaccard.npy")
        plot_basin_consistency(J, J_null,
                               save_path=self.fig_dir / "fig11_basin_consistency")
        print("Fig 11 saved.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--figs", default=DEFAULT_FIGS,
                        help="Comma-separated figure numbers to emit")
    parser.add_argument("--results-dir", default="artifacts/results")
    parser.add_argument("--figures-dir", default="figures")
    args = parser.parse_args()
    FigureMaker(args.results_dir, args.figures_dir).make(
        int(x) for x in args.figs.split(","))


if __name__ == "__main__":
    main()
