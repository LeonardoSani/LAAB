#!/usr/bin/env python
"""E1: Mixed-alignment breakdown over the depth grid."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse

from src.experiments import E1Mixed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--decomposition", "-d", action="store_true",
                   help="Run the M1^2 L2 decomposition after the main E1 run")
    args = p.parse_args()

    E1Mixed.create().run(run_decomposition=bool(args.decomposition))


if __name__ == "__main__":
    main()
