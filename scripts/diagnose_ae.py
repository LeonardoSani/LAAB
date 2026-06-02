#!/usr/bin/env python
"""Convergence report + PCA scatter + uniqueness per seed. Diagnostic only."""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.attractors.iteration import compute_attractors
from src.attractors.uniqueness import count_unique_attractors
from src.config import AEConfig, AnchorConfig
from src.data.mnist import get_mnist, sample_anchor_source
from src.store.checkpoints import CheckpointStore
from src.utils.device import get_device
from src.viz.convergence import plot_convergence
from src.viz.decoded_attractors import plot_decoded_attractors
from src.viz.latent_scatter import plot_latent_scatter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default=None, help="Comma-separated seeds, default: all found")
    parser.add_argument("--data-dir", default="artifacts/data")
    parser.add_argument("--ckpt-dir", default="artifacts/checkpoints")
    args = parser.parse_args()

    ae_cfg = AEConfig.load()
    acfg = AnchorConfig.load()
    device = get_device()
    ckpts = CheckpointStore(ae_cfg, device, args.ckpt_dir)
    fig_dir = Path("figures/diagnostics")
    fig_dir.mkdir(parents=True, exist_ok=True)

    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else ckpts.seeds
    print(f"Device: {device} | Seeds: {seeds}")

    train_ds, test_ds = get_mnist(args.data_dir)
    xa_path = Path("artifacts/anchors/X_a.pt")
    if xa_path.exists():
        X_a = torch.load(xa_path, weights_only=True)
    else:
        X_a = sample_anchor_source(train_ds, N=acfg.N, seed=acfg.sampling_seed)
        xa_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(X_a, xa_path)
    print(f"X_a: {X_a.shape}")

    all_test_images = torch.stack([test_ds[i][0] for i in range(len(test_ds))])
    all_test_labels = torch.tensor([test_ds[i][1] for i in range(len(test_ds))])
    test_images = all_test_images[:1000]   # first 1000 for PCA scatter
    test_labels = all_test_labels[:1000]

    summary_rows = []
    embeddings_per_seed: dict[int, torch.Tensor] = {}

    for s in seeds:
        model = ckpts.load_ae(s)

        with torch.no_grad():
            z0 = model.encoder(X_a.to(device))
        attractors, stats = compute_attractors(
            model.encoder, model.decoder, z0,
            tol=acfg.attractor_tol, max_iter=acfg.attractor_max_iter,
            track_residuals=True,
        )
        print(f"Seed s={s}: {stats}")
        plot_convergence({f"s={s}": stats},
                         save_path=fig_dir / f"convergence_s{s}.png",
                         tol=acfg.attractor_tol)

        u_stats = count_unique_attractors(attractors, threshold=0.99)
        print(f"  Uniqueness: {u_stats}")

        plot_decoded_attractors(model.decoder, attractors.cpu(), X_a.cpu(),
                                save_path=fig_dir / f"decoded_attractors_s{s}",
                                n=64, device=device)

        with torch.no_grad():
            x_all = all_test_images.to(device)
            recon_mse = float(((model(x_all) - x_all) ** 2).flatten(1).mean(1).mean().cpu())
        print(f"  Recon MSE (10k test): {recon_mse:.6f}")

        with torch.no_grad():
            embeddings_per_seed[s] = model.encoder(test_images.to(device)).cpu()

        summary_rows.append({
            "seed": s, "converged": f"{stats.n_converged}/{stats.n_total}",
            "mean_iters": f"{stats.mean_iters:.1f}", "max_iters": stats.max_iters,
            "unique": getattr(u_stats, "n_unique", "?"), "recon_mse": f"{recon_mse:.6f}",
        })

    ref_seed = 1
    if ref_seed in embeddings_per_seed:
        plot_latent_scatter(embeddings_per_seed, test_labels,
                            save_dir=fig_dir, shared_pca_ref_seed=ref_seed)

    print("\n{:<6} {:<12} {:<12} {:<12} {:<20} {:<14}".format(
        "Seed", "Converged", "Mean iters", "Max iters", "Unique attractors", "Recon MSE"))
    for r in summary_rows:
        print("{:<6} {:<12} {:<12} {:<12} {:<20} {:<14}".format(
            r["seed"], r["converged"], r["mean_iters"], r["max_iters"],
            r["unique"], r["recon_mse"]))


if __name__ == "__main__":
    main()
