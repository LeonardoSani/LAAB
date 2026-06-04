"""The Base / Mixed / Full relative representations (Method, eq. B/M/F).

Each is the same cosine map relative_cosine(data, anchors); they differ only in
which operands are iterated under the latent self-map phi_s^t:

  Base  B_s(x)   = cos( E_s(x),        E_s(x_i) )        anchors un-iterated (t=0)
  Mixed M_s^t(x) = cos( E_s(x),        phi_s^t(x_i) )    anchors iterated
  Full  F_s^t(x) = cos( phi_s^t(x),    phi_s^t(x_i) )    data iterated too

The cosine op is one primitive (cosine_map.relative_cosine); the only real
variation is operand sourcing, which needs the raw embeddings and the iterated-
anchor store — so the trio lives here as one store-backed builder, decoupled
from the experiment runner that uses it. Base is exactly Mixed at t=0.
"""
import torch

from src.depths import DepthLike
from src.relative.cosine_map import relative_cosine
from src.store.anchors import AnchorStore


class RelativeRepresentations:
    def __init__(self, encode_test, anchors: AnchorStore):
        """encode_test: zero-arg callable returning {seed: (|D|, k) raw embeddings},
        cached by the caller. anchors: store of iterated phi_anchors / phi_data."""
        self._encode_test = encode_test
        self._anchors = anchors

    def base(self, seed: int) -> torch.Tensor:
        """Base rep B_s: raw embeddings vs un-iterated anchors. (|D|, N)."""
        return relative_cosine(self._encode_test()[seed], self._anchors.phi_anchors(seed, 0))

    def mixed(self, seed: int, t: DepthLike) -> torch.Tensor:
        """Mixed rep M_s^t: raw embeddings vs depth-t iterated anchors. (|D|, N)."""
        return relative_cosine(self._encode_test()[seed], self._anchors.phi_anchors(seed, t))

    def full(self, seed: int, t: DepthLike) -> torch.Tensor:
        """Full rep F_s^t: depth-t iterated data vs depth-t iterated anchors. (|D|, N)."""
        return relative_cosine(self._anchors.phi_data(seed, t), self._anchors.phi_anchors(seed, t))
