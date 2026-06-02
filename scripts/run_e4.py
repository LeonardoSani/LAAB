#!/usr/bin/env python
"""E4: Full vector-field check (M1/M2 on the Full rep over the depth grid)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.experiments import E4Full

if __name__ == "__main__":
    E4Full.create().run()
