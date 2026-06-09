"""Compute and cache phi_s^t for anchors and the test set over the depth grid."""
from pathlib import Path

import torch

from src.attractors.snapshot import iterate_with_snapshots
from src.config import AEConfig, AnchorConfig, AttractorConfig, EvalConfig
from src.data.mnist import get_mnist, sample_anchor_source
from src.depths import depth_to_str
from src.store.checkpoints import CheckpointStore
from src.utils.device import get_device


def run_compute_iterates(target: str = "both", data_dir="artifacts/data",
                         ckpt_dir="artifacts/checkpoints",
                         anchor_dir="artifacts/anchors"):
    ae_cfg = AEConfig.load()
    acfg = AnchorConfig.load()
    atcfg = AttractorConfig.load()
    ecfg = EvalConfig.load()

    device = get_device()
    anchor_dir = Path(anchor_dir)
    anchor_dir.mkdir(parents=True, exist_ok=True)
    ckpts = CheckpointStore(ae_cfg, device, ckpt_dir)
    seeds = ckpts.seeds
    depth_grid = ecfg.depth_grid
    print(f"Device: {device} | Seeds: {seeds}")

    train_ds, test_ds = get_mnist(data_dir)

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

    do_anchors = target in ("anchors", "both")
    do_data = target in ("data", "both")

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
            want_stats = prefix == "phi_anchors"
            result = iterate_with_snapshots(
                model.encoder, model.decoder, z0, depth_grid,
                atcfg.attractor_tol, atcfg.attractor_max_iter,
                return_stats=want_stats,
            )
            snaps, stats = result if want_stats else (result, None)
            for t, snap in snaps.items():
                p = anchor_dir / f"{prefix}_s{s}_t{depth_to_str(t)}.pt"
                if not p.exists():
                    torch.save(snap, p)
            print(f"  Anchors: {stats}" if want_stats else "  Data iterates saved.")

    print("\nDone.")
