#!/usr/bin/env python
"""E0 baseline: relative representations agree across encoders at t=0."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.experiments import E0Baseline

if __name__ == "__main__":
    E0Baseline.create().run()
