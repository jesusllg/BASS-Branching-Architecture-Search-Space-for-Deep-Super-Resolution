"""Backward-compatible launcher for the repaired BASS demonstration."""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bass.cli import main
from bass.config import V1_GENOME_BITS, V2_GENOME_BITS
from bass.problem import BASSProblem


class OptimizationProblem(BASSProblem):
    """Backward-compatible name for the repaired bitstring problem adapter."""

    def __init__(self, n_var=V1_GENOME_BITS, n_obj=3, **kwargs):
        if n_var not in {V1_GENOME_BITS, V2_GENOME_BITS}:
            raise ValueError("n_var must be 84 (V1) or 93 (V2)")
        if n_obj not in {2, 3}:
            raise ValueError("n_obj must be 2 or 3")
        super().__init__(
            genome_version=1 if n_var == V1_GENOME_BITS else 2,
            include_flops=n_obj == 3,
            **kwargs,
        )


if __name__ == "__main__":
    raise SystemExit(main())
