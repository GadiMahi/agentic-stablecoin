"""Project-wide configuration and paths."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
DB_PATH = ROOT / "sthirasense.db"

MODEL_PREFIX = "improved_hybrid_depeg"

# Rolling windows the trained models actually require (minutes).
# The 7d/30d windows present in the original research pipeline were dropped
# during training (train.py DROP_PATTERNS), so the live engine never needs
# more than 24h of history to produce a complete feature vector.
WINDOWS = (180, 360, 720, 1440)
WARMUP_BARS = max(WINDOWS)

# Peg reference. Stablecoins target exactly 1.00 USD.
PEG = 1.0

__all__ = ["ROOT", "MODEL_DIR", "DB_PATH", "MODEL_PREFIX", "WINDOWS", "WARMUP_BARS", "PEG"]
