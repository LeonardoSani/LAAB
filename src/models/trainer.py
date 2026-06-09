"""Trainer for the AE and the relative decoder. Both share _fit (Adam + linear
decay, MSE, early stop, save-best), differing only in model and per-batch loss"""
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR
from tqdm import tqdm

from src.config import AEConfig
from src.data.mnist import get_dataloaders
from src.depths import DepthLike, depth_to_str
from src.models.autoencoder import Autoencoder
from src.models.relative_decoder import RelativeDecoder
from src.relative.cosine_map import relative_cosine
from src.store.anchors import AnchorStore
from src.utils.device import get_device
from src.utils.seeds import set_seed


class Trainer:
    def __init__(self, cfg: AEConfig, device=None, data_dir="artifacts/data",
                 ckpt_dir="artifacts/checkpoints"):
        self.cfg = cfg
        self.device = device or get_device()
        self.data_dir = data_dir
        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def train_autoencoder(self, seed: int) -> float:
        """Train one AE for `seed`. Saves ae_s{seed}.pt."""
        cfg = self.cfg
        set_seed(seed)
        print(f"Seed {seed} | Device {self.device}")
        train_loader, val_loader, _ = get_dataloaders(cfg, self.data_dir)
        model = Autoencoder(
            latent_dim=cfg.latent_dim, channel_base=cfg.channel_base,
            in_channels=cfg.in_channels, img_size=cfg.img_size,
        ).to(self.device)
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

        def batch_loss(criterion, x):
            return criterion(model(x), x)

        return self._fit(model, train_loader, val_loader, batch_loss,
                         self.ckpt_dir / f"ae_s{seed}.pt", {"seed": seed})

    def train_relative_decoder(self, depth: DepthLike) -> float:
        """Train the relative decoder at `depth` (ref s=1, encoder frozen).
        Saves relative_decoder_M_t{t}.pt."""
        cfg = self.cfg
        depth_str = depth_to_str(depth)
        set_seed(cfg.relative_decoder_init_seed)
        print(f"Depth t={depth_str} | Device: {self.device}")

        encoder = Autoencoder.from_checkpoint(
            self.ckpt_dir / "ae_s1.pt", cfg, self.device).encoder
        for p in encoder.parameters():
            p.requires_grad_(False)

        anchors = AnchorStore()
        if not anchors.has_anchor(1, depth):
            raise FileNotFoundError(
                f"phi_anchors_s1_t{depth_str}.pt not found. "
                "Run compute_iterates.py --target anchors first.")
        S = anchors.phi_anchors(1, depth).to(self.device)
        print(f"Anchor set S: {S.shape}")

        set_seed(cfg.relative_decoder_init_seed)
        decoder = RelativeDecoder(
            N=cfg.latent_dim, channel_base=cfg.channel_base,
            out_channels=cfg.in_channels, img_size=cfg.img_size,
        ).to(self.device)
        train_loader, val_loader, _ = get_dataloaders(cfg, self.data_dir)

        def batch_loss(criterion, x):
            with torch.no_grad():
                R = relative_cosine(encoder(x), S)   # Mixed rep, frozen
            return criterion(decoder(R), x)

        return self._fit(decoder, train_loader, val_loader, batch_loss,
                         self.ckpt_dir / f"relative_decoder_M_t{depth_str}.pt",
                         {"depth": depth_str})

    def _fit(self, model, train_loader, val_loader, batch_loss, save_path,
             save_extra: dict) -> float:
        """Shared loop: Adam + linear decay, MSE, early stop, save best."""
        cfg, device = self.cfg, self.device
        criterion = nn.MSELoss()
        optimizer = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.01,
                             total_iters=cfg.epochs)
        best_val = float("inf")
        no_improve = 0

        for epoch in range(1, cfg.epochs + 1):
            current_lr = optimizer.param_groups[0]["lr"]
            model.train()
            train_loss, n_train = 0.0, 0
            for x, _ in tqdm(train_loader, desc=f"Ep {epoch:3d}", leave=False):
                x = x.to(device)
                loss = batch_loss(criterion, x)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * x.size(0)
                n_train += x.size(0)
            train_loss /= n_train

            model.eval()
            val_loss, n_val = 0.0, 0
            with torch.no_grad():
                for x, _ in val_loader:
                    x = x.to(device)
                    val_loss += batch_loss(criterion, x).item() * x.size(0)
                    n_val += x.size(0)
            val_loss /= n_val
            scheduler.step()

            if epoch % 10 == 0 or epoch == 1:
                print(f"Epoch {epoch:4d} | train {train_loss:.6f} | "
                      f"val {val_loss:.6f} | lr {current_lr:.2e}")

            if val_loss < best_val:
                best_val = val_loss
                no_improve = 0
                torch.save({"epoch": epoch, "model_state": model.state_dict(),
                            "val_loss": val_loss, **save_extra}, save_path)
            else:
                no_improve += 1
                if no_improve >= cfg.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch} (no improvement for "
                          f"{cfg.early_stopping_patience} epochs)")
                    break

        print(f"Done. Best val MSE: {best_val:.6f} | Saved: {save_path}")
        return best_val
