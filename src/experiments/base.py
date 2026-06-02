"""Shared backbone for the cross-encoder alignment experiments E0-E4.

Holds the configs, the checkpoint/anchor stores, the seed list and seed pairs,
lazily encodes the test set once, and exposes the two relative representations
the experiments build on:
  - Mixed: raw embeddings Z_s(X_test) against iterated anchors phi_anchors.
  - Full : iterated data phi_data against iterated anchors phi_anchors.
Subclasses implement run() to write their result files.
"""
from itertools import combinations
from pathlib import Path

import torch

from src.config import Configs
from src.data.mnist import get_mnist
from src.relative.cosine_map import relative_cosine
from src.store.anchors import AnchorStore
from src.store.checkpoints import CheckpointStore
from src.utils.device import get_device


class Experiment:
    name = "experiment"

    def __init__(self, configs: Configs, ckpts: CheckpointStore,
                 anchors: AnchorStore, device, results_dir="artifacts/results",
                 data_dir="artifacts/data"):
        self.configs = configs
        self.ckpts = ckpts
        self.anchors = anchors
        self.device = device
        self.results_dir = Path(results_dir)
        self.data_dir = data_dir
        self.seeds = ckpts.seeds
        self.pairs = list(combinations(self.seeds, 2))
        self._Z = None
        self._test_images = None

    @classmethod
    def create(cls, config_dir="configs", results_dir="artifacts/results",
               data_dir="artifacts/data") -> "Experiment":
        """Wire up configs + stores + device from the standard layout."""
        configs = Configs.load(config_dir)
        device = get_device()
        return cls(
            configs=configs,
            ckpts=CheckpointStore(configs.ae, device),
            anchors=AnchorStore(),
            device=device,
            results_dir=results_dir,
            data_dir=data_dir,
        )

    # --- shared data ---

    @property
    def test_images(self) -> torch.Tensor:
        """Full MNIST test set (|D|, 1, 28, 28), fixed across depths."""
        if self._test_images is None:
            _, test_ds = get_mnist(self.data_dir)
            self._test_images = torch.stack(
                [test_ds[i][0] for i in range(len(test_ds))])
        return self._test_images

    def encode_test(self) -> dict[int, torch.Tensor]:
        """Raw embeddings Z_s = E_s(X_test) per seed (CPU), encoded once."""
        if self._Z is None:
            X = self.test_images.to(self.device)
            self._Z = {}
            with torch.no_grad():
                for s in self.seeds:
                    self._Z[s] = self.ckpts.load_ae(s).encoder(X).cpu()
        return self._Z

    # --- relative representations ---

    def mixed_rep(self, seed: int, t) -> torch.Tensor:
        """Mixed rep r_s^t: raw embeddings vs depth-t iterated anchors. (|D|, N)."""
        return relative_cosine(self.encode_test()[seed], self.anchors.phi_anchors(seed, t))

    def full_rep(self, seed: int, t) -> torch.Tensor:
        """Full rep: depth-t iterated data vs depth-t iterated anchors. (|D|, N)."""
        return relative_cosine(self.anchors.phi_data(seed, t), self.anchors.phi_anchors(seed, t))

    # --- to implement ---

    def run(self):
        raise NotImplementedError

    def _ensure_results_dir(self):
        self.results_dir.mkdir(parents=True, exist_ok=True)
