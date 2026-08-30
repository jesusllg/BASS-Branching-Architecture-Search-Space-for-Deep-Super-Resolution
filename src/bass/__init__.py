"""BASS: Branching Architecture Search Space for super-resolution."""

from .config import (
    ATTENTION_PRIMITIVES,
    CHANNELS,
    CNN_PRIMITIVES,
    KERNEL_SIZES,
    REPEATS,
    WINDOW_SIZES,
)
from .encoding import (
    decode,
    decode_v1_bits,
    decode_v1_gene,
    decode_v2_bits,
    encode_v2_bits,
    sample_v2,
    upgrade_v1,
)
from .genotype import ArchitectureSpec, BlockGene
from .repair import repair_architecture

__all__ = [
    "ATTENTION_PRIMITIVES",
    "CHANNELS",
    "CNN_PRIMITIVES",
    "KERNEL_SIZES",
    "REPEATS",
    "WINDOW_SIZES",
    "ArchitectureSpec",
    "BlockGene",
    "decode",
    "decode_v1_bits",
    "decode_v1_gene",
    "decode_v2_bits",
    "encode_v2_bits",
    "repair_architecture",
    "sample_v2",
    "upgrade_v1",
]

__version__ = "0.2.0"
