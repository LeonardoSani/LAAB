#!/usr/bin/env python
"""Train the relative decoder D_r^{M,t} for a given depth t.

Reference seed s=1, encoder frozen; target = the reference Mixed rep
relative_cosine(E_1(x), phi_anchors_s1_t). Saves
artifacts/checkpoints/relative_decoder_M_t{t}.pt.
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import AEConfig
from src.data.mnist import get_dataloaders
from src.depths import depth_to_str, str_to_depth
from src.models.autoencoder import Autoencoder
from src.models.relative_decoder import RelativeDecoder
from src.relative.cosine_map import relative_cosine
from src.store.anchors import AnchorStore
from src.utils.device import get_device
from src.utils.seeds import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", required=True, help="Depth t: 0, 8, 64, 512, or inf")
    parser.add_argument("--config", default="configs/ae.yaml")
    parser.add_argument("--data-dir", default="artifacts/data")
    parser.add_argument("--ckpt-dir", default="artifacts/checkpoints")
    args = parser.parse_args()

    depth = str_to_depth(args.depth)
    depth_str = depth_to_str(depth)
    cfg = AEConfig.load(args.config)

    set_seed(cfg.relative_decoder_init_seed)
    device = get_device()
    print(f"Depth t={depth_str} | Device: {device}")

    ckpt_dir = Path(args.ckpt_dir)
    anchors = AnchorStore()

    # Reference encoder E_1 (frozen)
    encoder = Autoencoder.from_checkpoint(ckpt_dir / "ae_s1.pt", cfg, device).encoder
    for p in encoder.parameters():
        p.requires_grad_(False)

    if not anchors.has_anchor(1, depth):
        raise FileNotFoundError(
            f"phi_anchors_s1_t{depth_str}.pt not found. "
            "Run compute_iterates.py --target anchors first.")
    S = anchors.phi_anchors(1, depth).to(device)   # (N, k)
    print(f"Anchor set S: {S.shape}")

    set_seed(cfg.relative_decoder_init_seed)
    decoder = RelativeDecoder(
        N=cfg.latent_dim, channel_base=cfg.channel_base,
        out_channels=cfg.in_channels, img_size=cfg.img_size,
    ).to(device)

    train_loader, val_loader, _ = get_dataloaders(cfg, args.data_dir)
    criterion = nn.MSELoss()
    optimizer = Adam(decoder.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=cfg.epochs)

    best_val = float("inf")
    no_improve = 0
    save_path = ckpt_dir / f"relative_decoder_M_t{depth_str}.pt"

    for epoch in range(1, cfg.epochs + 1):
        decoder.train()
        train_loss, n_train = 0.0, 0
        for x, _ in tqdm(train_loader, desc=f"Ep {epoch:3d}", leave=False):
            x = x.to(device)
            with torch.no_grad():
                R = relative_cosine(encoder(x), S)   # (B, N) Mixed rep
            loss = criterion(decoder(R), x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            n_train += x.size(0)
        train_loss /= n_train

        decoder.eval()
        val_loss, n_val = 0.0, 0
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device)
                R = relative_cosine(encoder(x), S)
                val_loss += criterion(decoder(R), x).item() * x.size(0)
                n_val += x.size(0)
        val_loss /= n_val
        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d} | train {train_loss:.6f} | val {val_loss:.6f} | "
                  f"lr {optimizer.param_groups[0]['lr']:.2e}")

        if val_loss < best_val:
            best_val = val_loss
            no_improve = 0
            torch.save({"epoch": epoch, "model_state": decoder.state_dict(),
                        "val_loss": val_loss, "depth": depth_str}, save_path)
        else:
            no_improve += 1
            if no_improve >= cfg.early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"Done. Best val MSE: {best_val:.6f} | Saved: {save_path}")


if __name__ == "__main__":
    main()
