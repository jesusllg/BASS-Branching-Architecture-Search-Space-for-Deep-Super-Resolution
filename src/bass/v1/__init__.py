"""BASS V1: frozen CNN-only 84-bit search space."""

from .config import CHANNELS, CNN_PRIMITIVES, KERNEL_SIZES, REPEATS
from .encoding import decode, decode_bits, decode_gene
from .genotype import ArchitectureSpec, BlockGene
from .model_builder import build_model
from .problem import BASSProblem

__all__ = [
    "CHANNELS",
    "CNN_PRIMITIVES",
    "KERNEL_SIZES",
    "REPEATS",
    "ArchitectureSpec",
    "BASSProblem",
    "BlockGene",
    "build_model",
    "decode",
    "decode_bits",
    "decode_gene",
]
