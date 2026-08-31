"""Historical NSGA3 alias for the unvalidated reference-direction EA."""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bass.shared.nsga3 import (  # noqa: F401
    NSGA3,
    ReferenceDirectionEA,
    ReferencePoint,
)
