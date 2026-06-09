"""Typed configs from configs/*.yaml; unknown keys ignored."""
from dataclasses import dataclass, fields
from pathlib import Path

import yaml

from src.depths import DepthLike, parse_depth_grid


def _read(path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _known(cls, d: dict) -> dict:
    names = {f.name for f in fields(cls)}
    return {k: v for k, v in d.items() if k in names}


@dataclass
class AEConfig:
    """AE architecture + training (configs/ae.yaml)."""
    latent_dim: int
    channel_base: int
    in_channels: int
    img_size: int
    optimizer: str = "adam"
    lr: float = 5.0e-4
    weight_decay: float = 1.0e-4
    epochs: int = 500
    batch_size: int = 128
    val_fraction: float = 0.1
    val_seed: int = 0
    early_stopping_patience: int = 50
    relative_decoder_init_seed: int = 42

    @classmethod
    def load(cls, path="configs/ae.yaml") -> "AEConfig":
        return cls(**_known(cls, _read(path)))


@dataclass
class AnchorConfig:
    """Anchor sampling (configs/anchors.yaml)."""
    N: int
    sampling_seed: int = 42

    @classmethod
    def load(cls, path="configs/anchors.yaml") -> "AnchorConfig":
        return cls(**_known(cls, _read(path)))


@dataclass
class AttractorConfig:
    """Iteration + cosine threshold (configs/attractors.yaml)."""
    attractor_tol: float = 1.0e-6     # ||Δz||² convergence threshold
    attractor_max_iter: int = 3000
    tau: float = 0.99                 # cos threshold for basin / attractor equality

    @classmethod
    def load(cls, path="configs/attractors.yaml") -> "AttractorConfig":
        return cls(**_known(cls, _read(path)))


@dataclass
class EvalConfig:
    """Eval grid (configs/eval.yaml)."""
    reference_seed: int
    num_seeds: int
    depth_grid: list[DepthLike]
    decoder_depths: list[DepthLike]

    @classmethod
    def load(cls, path="configs/eval.yaml") -> "EvalConfig":
        d = _read(path)
        return cls(
            reference_seed=d["reference_seed"],
            num_seeds=d["num_seeds"],
            depth_grid=parse_depth_grid(d["depth_grid"]),
            decoder_depths=parse_depth_grid(d["decoder_depths"]),
        )


@dataclass
class Configs:
    """All four configs."""
    ae: AEConfig
    anchors: AnchorConfig
    attractors: AttractorConfig
    eval: EvalConfig

    @classmethod
    def load(cls, config_dir="configs") -> "Configs":
        d = Path(config_dir)
        return cls(
            ae=AEConfig.load(d / "ae.yaml"),
            anchors=AnchorConfig.load(d / "anchors.yaml"),
            attractors=AttractorConfig.load(d / "attractors.yaml"),
            eval=EvalConfig.load(d / "eval.yaml"),
        )
