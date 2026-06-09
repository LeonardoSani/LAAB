#!/usr/bin/env python
"""Train the relative decoder at depth t (ref s=1, encoder frozen).
Saves artifacts/checkpoints/relative_decoder_M_t{t}.pt."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import AEConfig
from src.depths import depth_to_str, str_to_depth
from src.models import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", required=True, help="Depth t: 0, 8, 64, 512, or inf")
    parser.add_argument("--config", default="configs/ae.yaml")
    parser.add_argument("--data-dir", default="artifacts/data")
    parser.add_argument("--ckpt-dir", default="artifacts/checkpoints")
    args = parser.parse_args()

    depth = str_to_depth(args.depth)
    depth_str = depth_to_str(depth)
    cfg = AEConfig.load(args.config)
    Trainer(cfg, data_dir=args.data_dir, ckpt_dir=args.ckpt_dir).train_relative_decoder(depth)


if __name__ == "__main__":
    main()
