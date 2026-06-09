#!/usr/bin/env python
"""Convergence report + PCA scatter + uniqueness per seed. Diagnostic only."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis import run_ae_diagnostics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default=None, help="Comma-separated seeds, default: all found")
    parser.add_argument("--data-dir", default="artifacts/data")
    parser.add_argument("--ckpt-dir", default="artifacts/checkpoints")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else None
    run_ae_diagnostics(seeds=seeds, data_dir=args.data_dir, ckpt_dir=args.ckpt_dir)


if __name__ == "__main__":
    main()
