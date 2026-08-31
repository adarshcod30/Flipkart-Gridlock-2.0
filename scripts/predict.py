#!/usr/bin/env python3
"""CLI entry point: python scripts/predict.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gridlock.predict import main

if __name__ == "__main__":
    main()
