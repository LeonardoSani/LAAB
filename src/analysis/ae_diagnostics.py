"""AE diagnostics: convergence, uniqueness, recon MSE, latent PCA scatter."""
from pathlib import Path

import torch

from src.attractors.iteration import compute_attractors
from src.attractors.uniqueness import count_unique_attractors
from src.config import AEConfig, AnchorConfig, AttractorConfig
from src.data.mnist import get_mnist, sample_anchor_source
from src.store.checkpoints import CheckpointStore
from src.utils.device import get_device
from src.viz.convergence import plot_convergence
from src.viz.decoded_attractors import plot_decoded_attractors
from src.viz.latent_scatter import plot_latent_scatter


def run_ae_diagnostics(seeds=None, data_dir="artifacts/data",
                       ckpt_dir="artifacts/checkpoints",
                       fig_dir="figures/diagnostics"):
    ae_cfg = AEConfig.load()
    acfg = AnchorConfig.load()
    atcfg = AttractorConfig.load()
    device = get_device()
    ckpts = CheckpointStore(ae_cfg, device, ckpt_dir)
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    seed_list = [int(s) for s in seeds] if seeds else ckpts.seeds
    print(f"Device: {device} | Seeds: {seed_list}")

    train_ds, test_ds = get_mnist(data_dir)
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
    test_images = all_test_images[:1000]
    test_labels = all_test_labels[:1000]

    summary_rows = []
    embeddings_per_seed: dict[int, torch.Tensor] = {}

    for s in seed_list:
        model = ckpts.load_ae(s)

        with torch.no_grad():
            z0 = model.encoder(X_a.to(device))
        attractors, stats = compute_attractors(
            model.encoder, model.decoder, z0,
            tol=atcfg.attractor_tol, max_iter=atcfg.attractor_max_iter,
            track_residuals=True,
        )
        print(f"Seed s={s}: {stats}")
        plot_convergence({f"s={s}": stats},
                         save_path=fig_dir / f"convergence_s{s}.png",
                         tol=atcfg.attractor_tol)

        u_stats = count_unique_attractors(attractors, threshold=atcfg.tau)
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