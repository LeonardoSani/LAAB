#!/usr/bin/env python
"""Compute and cache phi_s^t for anchors and/or the full test set."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.attractors.iterates import run_compute_iterates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["anchors", "data", "both"], default="both")
    parser.add_argument("--data-dir", default="artifacts/data")
    parser.add_argument("--ckpt-dir", default="artifacts/checkpoints")
    args = parser.parse_args()

    run_compute_iterates(args.target, data_dir=args.data_dir, ckpt_dir=args.ckpt_dir)


if __name__ == "__main__":
    main()
