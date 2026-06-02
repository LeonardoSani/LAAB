#!/usr/bin/env python
"""E3: stitching degradation (M4) over the relative-decoder depths."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.experiments import E3Stitching

if __name__ == "__main__":
    E3Stitching.create().run()
