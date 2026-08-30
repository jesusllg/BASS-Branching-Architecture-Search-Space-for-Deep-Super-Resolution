"""Compatibility exports for the original Keras model builder."""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bass.encoding import decode
from bass.model_builder import PixelShuffle, build_model, get_model  # noqa: F401
from bass.registry import make_unit_layers


def get_branches(genotype):
    """Build the three legacy layer lists without constructing a full model."""

    architecture = decode(genotype)
    branches = []
    for branch_index, branch in enumerate(architecture.branches, start=1):
        branch_layers = []
        for unit_index, block in enumerate(branch, start=1):
            branch_layers.extend(
                make_unit_layers(
                    block,
                    architecture.channels,
                    f"branch{branch_index}_unit{unit_index}_{block.op}",
                )
            )
        branches.append(branch_layers)
    return branches, architecture.channels
