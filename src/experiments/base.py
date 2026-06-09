"""Shared backbone for E0-E5: configs, stores, seed pairs, cached embeddings,
and the Base/Mixed/Full reps. Subclasses implement run()."""
from itertools import combinations
from pathlib import Path

import torch

from src.config import Configs
from src.data.mnist import get_mnist
from src.relative.representations import RelativeRepresentations
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
        self.reps = RelativeRepresentations(self.encode_test, self.anchors)

    @classmethod
    def create(cls, config_dir="configs", results_dir="artifacts/results",
               data_dir="artifacts/data") -> "Experiment":
        """Wire configs + stores + device."""
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

    @property
    def test_images(self) -> torch.Tensor:
        """Full MNIST test set (|D|, 1, 28, 28)."""
        if self._test_images is None:
            _, test_ds = get_mnist(self.data_dir)
            self._test_images = torch.stack(
                [test_ds[i][0] for i in range(len(test_ds))])
        return self._test_images

    def encode_test(self) -> dict[int, torch.Tensor]:
        """Embeddings E_s(X_test) per seed (CPU), cached."""
        if self._Z is None:
            X = self.test_images.to(self.device)
            self._Z = {}
            with torch.no_grad():
                for s in self.seeds:
                    self._Z[s] = self.ckpts.load_ae(s).encoder(X).cpu()
        return self._Z

    def base_rep(self, seed: int) -> torch.Tensor:
        return self.reps.base(seed)

    def mixed_rep(self, seed: int, t) -> torch.Tensor:
        return self.reps.mixed(seed, t)

    def full_rep(self, seed: int, t) -> torch.Tensor:
        return self.reps.full(seed, t)

    def run(self):
        raise NotImplementedError

    def _ensure_results_dir(self):
        self.results_dir.mkdir(parents=True, exist_ok=True)
