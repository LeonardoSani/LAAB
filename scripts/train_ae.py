#!/usr/bin/env python
"""Train one autoencoder for a given seed. Saves the best checkpoint to
artifacts/checkpoints/ae_s{seed}.pt."""
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
from src.models.autoencoder import Autoencoder
from src.utils.device import get_device
from src.utils.seeds import set_seed


def train(seed: int, cfg: AEConfig, data_dir="artifacts/data",
          ckpt_dir="artifacts/checkpoints") -> float:
    set_seed(seed)
    device = get_device()
    print(f"Seed {seed} | Device {device}")

    train_loader, val_loader, _ = get_dataloaders(cfg, data_dir)

    model = Autoencoder(
        latent_dim=cfg.latent_dim, channel_base=cfg.channel_base,
        in_channels=cfg.in_channels, img_size=cfg.img_size,
    ).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=cfg.epochs)
    criterion = nn.MSELoss()

    ckpt_path = Path(ckpt_dir)
    ckpt_path.mkdir(parents=True, exist_ok=True)
    save_path = ckpt_path / f"ae_s{seed}.pt"

    best_val = float("inf")
    no_improve = 0
    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)

    for epoch in range(1, cfg.epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        model.train()
        train_loss = 0.0
        for x, _ in tqdm(train_loader, desc=f"Ep {epoch:3d}", leave=False):
            x = x.to(device)
            loss = criterion(model(x), x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= n_train

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device)
                val_loss += criterion(model(x), x).item() * x.size(0)
        val_loss /= n_val
        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d} | train {train_loss:.6f} | val {val_loss:.6f} | lr {current_lr:.2e}")

        if val_loss < best_val:
            best_val = val_loss
            no_improve = 0
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict(),
                 "val_loss": val_loss, "seed": seed},
                save_path,
            )
        else:
            no_improve += 1
            if no_improve >= cfg.early_stopping_patience:
                print(f"Early stopping at epoch {epoch} (no improvement for "
                      f"{cfg.early_stopping_patience} epochs)")
                break

    print(f"Done. Best val MSE: {best_val:.6f} | Saved: {save_path}")
    return best_val


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, help="Training seed (s=1 is reference)")
    parser.add_argument("--config", default="configs/ae.yaml")
    parser.add_argument("--data-dir", default="artifacts/data")
    parser.add_argument("--ckpt-dir", default="artifacts/checkpoints")
    args = parser.parse_args()

    train(args.seed, AEConfig.load(args.config), args.data_dir, args.ckpt_dir)
