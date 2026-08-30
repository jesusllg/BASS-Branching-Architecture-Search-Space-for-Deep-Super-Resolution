"""Compatibility export for the repaired NSGA-III implementation."""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bass.nsga3 import NSGA3, ReferencePoint  # noqa: F401
