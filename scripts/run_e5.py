#!/usr/bin/env python
"""E5: basin-of-attraction consistency across seeds (plus all-data robustness check)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.experiments import E5Basins

if __name__ == "__main__":
    e5 = E5Basins.create()
    e5.run()
    print("\n--- E5 robustness check: all-of-D refs ---")
    e5.run_all_data_check()
