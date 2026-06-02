"""Access to trained model checkpoints in artifacts/checkpoints/.

Single home for seed discovery and model loading — replaces the find_seeds /
load_ae helpers that were copy-pasted across every script.
"""
import re
from pathlib import Path

from src.depths import DepthLike, depth_to_str
from src.models.autoencoder import Autoencoder
from src.models.relative_decoder import RelativeDecoder


class CheckpointStore:
    def __init__(self, ae_cfg, device, ckpt_dir="artifacts/checkpoints"):
        self.ckpt_dir = Path(ckpt_dir)
        self.ae_cfg = ae_cfg
        self.device = device

    @property
    def seeds(self) -> list[int]:
        """Sorted training seeds discovered from ae_s{seed}.pt files."""
        seeds = []
        for p in sorted(self.ckpt_dir.glob("ae_s*.pt")):
            m = re.match(r"ae_s(\d+)\.pt", p.name)
            if m:
                seeds.append(int(m.group(1)))
        return sorted(seeds)

    def load_ae(self, seed: int) -> Autoencoder:
        return Autoencoder.from_checkpoint(
            self.ckpt_dir / f"ae_s{seed}.pt", self.ae_cfg, self.device)

    def relative_decoder_path(self, t: DepthLike) -> Path:
        return self.ckpt_dir / f"relative_decoder_M_t{depth_to_str(t)}.pt"

    def has_relative_decoder(self, t: DepthLike) -> bool:
        return self.relative_decoder_path(t).exists()

    def load_relative_decoder(self, t: DepthLike) -> RelativeDecoder:
        return RelativeDecoder.from_checkpoint(
            self.relative_decoder_path(t), self.ae_cfg, self.device)
