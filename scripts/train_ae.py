#!/usr/bin/env python
"""Train one AE for a seed. Saves artifacts/checkpoints/ae_s{seed}.pt."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import AEConfig
from src.models import Trainer


def train(seed: int, cfg: AEConfig, data_dir="artifacts/data",
          ckpt_dir="artifacts/checkpoints") -> float:
    return Trainer(cfg, data_dir=data_dir, ckpt_dir=ckpt_dir).train_autoencoder(seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, help="Training seed (s=1 is reference)")
    parser.add_argument("--config", default="configs/ae.yaml")
    parser.add_argument("--data-dir", default="artifacts/data")
    parser.add_argument("--ckpt-dir", default="artifacts/checkpoints")
    args = parser.parse_args()

    train(args.seed, AEConfig.load(args.config), args.data_dir, args.ckpt_dir)
