"""Backward-compatible launcher for the repaired BASS demonstration."""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bass.cli import main as bass_main
from bass.v1.config import GENOME_BITS
from bass.v1.problem import BASSProblem


class OptimizationProblem(BASSProblem):
    """Backward-compatible name for the repaired bitstring problem adapter."""

    def __init__(self, n_var=GENOME_BITS, n_obj=3, **kwargs):
        if n_var != GENOME_BITS:
            raise ValueError("The historical Implementation namespace is BASS V1 only")
        if n_obj not in {2, 3}:
            raise ValueError("n_obj must be 2 or 3")
        super().__init__(
            include_flops=n_obj == 3,
            **kwargs,
        )


def main(argv=None):
    """Run the V1 CLI while rejecting attempts to select V2 here."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--genome-version" in arguments:
        index = arguments.index("--genome-version")
        try:
            requested = int(arguments[index + 1])
        except (IndexError, ValueError) as error:
            raise SystemExit("--genome-version requires the value 1") from error
        if requested != 1:
            raise SystemExit(
                "Implementation/ is BASS V1 only; use bass-search --genome-version 2"
            )
        del arguments[index : index + 2]
    return bass_main(["--genome-version", "1", *arguments])


if __name__ == "__main__":
    raise SystemExit(main())
