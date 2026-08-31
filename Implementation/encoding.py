"""Compatibility exports for the original ``Implementation/encoding.py``."""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bass.v1 import config as _config
from bass.v1.encoding import *
from bass.v1.genotype import ArchitectureSpec

CHANNELS = list(_config.CHANNELS)
KERNEL_SIZES = list(_config.KERNEL_SIZES)
PRIMITIVES = list(_config.CNN_PRIMITIVES)
REPEAT = list(_config.REPEATS)
Genotype = ArchitectureSpec


def convert_cell(cell_bit_string):
    """Return the original nested three-field representation of branch bits."""

    units = [
        cell_bit_string[index : index + 9]
        for index in range(0, len(cell_bit_string), 9)
    ]
    return [
        [unit[index : index + 3] for index in range(0, len(unit), 3)] for unit in units
    ]


def convert(bit_string):
    """Split the 81 post-channel V1 bits into three nested branches."""

    if len(bit_string) != 81:
        raise ValueError("The V1 branch payload must contain 81 bits")
    return [convert_cell(bit_string[index : index + 27]) for index in range(0, 81, 27)]
