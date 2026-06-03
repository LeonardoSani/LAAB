#!/usr/bin/env python
"""E5: basin-of-attraction consistency across seeds (Jaccard vs null, link to mu_i)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.experiments import E5Basins

if __name__ == "__main__":
    E5Basins.create().run()
