#!/usr/bin/env python
"""E2: per-anchor mismatch, badness consistency, and digit-class tails."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.experiments import E2Anchors

if __name__ == "__main__":
    E2Anchors.create().run()
