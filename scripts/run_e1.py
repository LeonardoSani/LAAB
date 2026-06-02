#!/usr/bin/env python
"""E1: Mixed-alignment breakdown over the depth grid."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.experiments import E1Mixed

if __name__ == "__main__":
    E1Mixed.create().run()
