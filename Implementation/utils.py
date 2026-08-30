"""Compatibility utility functions retained from BASS V1."""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bass.evaluation import psnr  # noqa: F401


def Dominance(a_f, b_f):
    a_better = any(a < b for a, b in zip(a_f, b_f))
    b_better = any(b < a for a, b in zip(a_f, b_f))
    if a_better and not b_better:
        return 1
    if b_better and not a_better:
        return -1
    return 0
