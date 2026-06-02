#!/usr/bin/env python
"""Compute and cache phi_s^t for anchors and/or the full test set over the
depth grid: phi_anchors_s{s}_t{T}.pt and phi_data_s{s}_t{T}.pt."""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.attractors.iteration import compute_attractors
from src.attractors.snapshot import iterate_with_snapshots
from src.config import AEConfig, AnchorConfig, EvalConfig
from src.data.mnist import get_mnist, sample_anchor_source
from src.depths import depth_to_str
from src.store.checkpoints import CheckpointStore
from src.utils.device import get_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["anchors", "data", "both"], default="both")
    parser.add_argument("--data-dir", default="artifacts/data")
    parser.add_argument("--ckpt-dir", default="artifacts/checkpoints")
    args = parser.parse_args()

    ae_cfg = AEConfig.load()
    acfg = AnchorConfig.load()
    ecfg = EvalConfig.load()

    device = get_device()
    anchor_dir = Path("artifacts/anchors")
    anchor_dir.mkdir(parents=True, exist_ok=True)
    ckpts = CheckpointStore(ae_cfg, device, args.ckpt_dir)
    seeds = ckpts.seeds
    depth_grid = ecfg.depth_grid
    print(f"Device: {device} | Seeds: {seeds}")

    train_ds, test_ds = get_mnist(args.data_dir)

    xa_path = anchor_dir / "X_a.pt"
    if xa_path.exists():
        X_a = torch.load(xa_path, weights_only=True)
        print(f"Loaded X_a: {X_a.shape}")
    else:
        X_a = sample_anchor_source(train_ds, N=acfg.N, seed=acfg.sampling_seed)
        torch.save(X_a, xa_path)
        print(f"Saved X_a: {X_a.shape}")

    X_d = torch.stack([test_ds[i][0] for i in range(len(test_ds))])
    print(f"X_d (full test set): {X_d.shape}")

    do_anchors = args.target in ("anchors", "both")
    do_data = args.target in ("data", "both")

    for s in seeds:
        model = ckpts.load_ae(s)
        print(f"\nSeed {s}")

        for X, prefix, enabled in [(X_a, "phi_anchors", do_anchors),
                                   (X_d, "phi_data", do_data)]:
            if not enabled:
                continue
            if all((anchor_dir / f"{prefix}_s{s}_t{depth_to_str(t)}.pt").exists()
                   for t in depth_grid):
                print(f"  {prefix}: all iterates exist, skipping.")
                continue
            with torch.no_grad():
                z0 = model.encoder(X.to(device))
            snaps = iterate_with_snapshots(
                model.encoder, model.decoder, z0, depth_grid,
                acfg.attractor_tol, acfg.attractor_max_iter)
            for t, snap in snaps.items():
                p = anchor_dir / f"{prefix}_s{s}_t{depth_to_str(t)}.pt"
                if not p.exists():
                    torch.save(snap, p)
            if prefix == "phi_anchors":
                _, stats = compute_attractors(model.encoder, model.decoder, z0,
                                              acfg.attractor_tol, acfg.attractor_max_iter)
                print(f"  Anchors: {stats}")
            else:
                print(f"  Data iterates saved.")

    print("\nDone.")


if __name__ == "__main__":
    main()
