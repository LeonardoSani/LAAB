#!/usr/bin/env python
"""Emit the paper figures from cached results (runner over FigureMaker)."""
import argparse
import os
import sys
from pathlib import Path

import matplotlib

# non-interactive backend, before any pyplot import
os.environ.setdefault("MPLBACKEND", "Agg")
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.viz.figure_maker import DEFAULT_FIGS, FigureMaker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--figs", default=DEFAULT_FIGS,
                        help="Comma-separated figure numbers to emit")
    parser.add_argument("--results-dir", default="artifacts/results")
    parser.add_argument("--figures-dir", default="figures")
    args = parser.parse_args()
    FigureMaker(args.results_dir, args.figures_dir).make(
        int(x) for x in args.figs.split(","))


if __name__ == "__main__":
    main()
