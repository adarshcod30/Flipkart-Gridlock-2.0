#!/usr/bin/env python3
"""CLI entry point: python scripts/train.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gridlock.train import main

if __name__ == "__main__":
    main()
