"""Base / Mixed / Full relative reps: same cosine map, differing in which
operands are iterated.
  Base  = cos(E(x),    E(x_i))       anchors un-iterated (= Mixed at t=0)
  Mixed = cos(E(x),    phi^t(x_i))   anchors iterated
  Full  = cos(phi^t(x), phi^t(x_i))  data iterated too
"""
import torch

from src.depths import DepthLike
from src.relative.cosine_map import relative_cosine
from src.store.anchors import AnchorStore


class RelativeRepresentations:
    def __init__(self, encode_test, anchors: AnchorStore):
        """encode_test: () -> {seed: (|D|, k) embeddings}. anchors: iterate store."""
        self._encode_test = encode_test
        self._anchors = anchors

    def base(self, seed: int) -> torch.Tensor:
        """(|D|, N)."""
        return relative_cosine(self._encode_test()[seed], self._anchors.phi_anchors(seed, 0))

    def mixed(self, seed: int, t: DepthLike) -> torch.Tensor:
        """(|D|, N)."""
        return relative_cosine(self._encode_test()[seed], self._anchors.phi_anchors(seed, t))

    def full(self, seed: int, t: DepthLike) -> torch.Tensor:
        """(|D|, N)."""
        return relative_cosine(self._anchors.phi_data(seed, t), self._anchors.phi_anchors(seed, t))
