"""Access to the anchor source set and cached iterate tensors in
artifacts/anchors/.

phi_anchors_s{s}_t{T}.pt : (N, k)     anchor set X_a iterated to depth T under E_s
phi_data_s{s}_t{T}.pt    : (|D|, k)   full test set iterated to depth T under E_s

Replaces every ad-hoc torch.load(anchor_dir / "phi_...") in the scripts.
"""
from pathlib import Path

import torch

from src.depths import DepthLike, depth_to_str


class AnchorStore:
    def __init__(self, anchor_dir="artifacts/anchors"):
        self.anchor_dir = Path(anchor_dir)

    @property
    def X_a(self) -> torch.Tensor:
        return torch.load(self.anchor_dir / "X_a.pt", weights_only=True)

    def _anchor_path(self, seed: int, t: DepthLike) -> Path:
        return self.anchor_dir / f"phi_anchors_s{seed}_t{depth_to_str(t)}.pt"

    def _data_path(self, seed: int, t: DepthLike) -> Path:
        return self.anchor_dir / f"phi_data_s{seed}_t{depth_to_str(t)}.pt"

    def phi_anchors(self, seed: int, t: DepthLike) -> torch.Tensor:
        return torch.load(self._anchor_path(seed, t), weights_only=True)

    def phi_data(self, seed: int, t: DepthLike) -> torch.Tensor:
        return torch.load(self._data_path(seed, t), weights_only=True)

    def has_anchor(self, seed: int, t: DepthLike) -> bool:
        return self._anchor_path(seed, t).exists()

    def has_data(self, seed: int, t: DepthLike) -> bool:
        return self._data_path(seed, t).exists()
