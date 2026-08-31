"""BASS V2: canonical semantic NAS with optional CNN/attention units."""

from .config import (
    ATTENTION_PRIMITIVES,
    CHANNELS,
    CNN_PRIMITIVES,
    KERNEL_SIZES,
    LEGACY_GENOME_BITS,
    REPEATS,
    SEMANTIC_GENOME_LENGTH,
    UNIT_STATE_COUNT,
    WINDOW_SIZES,
)
from .encoding import (
    block_to_state,
    canonicalize_genome,
    decode,
    decode_legacy_bits,
    decode_v2_bits,
    encode,
    encode_legacy_bits,
    encode_v2_bits,
    migrate_legacy93,
    migrate_v1,
    sample,
    sample_genome,
    sample_v2,
    state_to_block,
    upgrade_v1,
)
from .genotype import (
    ArchitectureSpec,
    BlockGene,
    canonicalize_architecture,
)
from .legacy93 import LegacyArchitectureSpec, LegacyBlockGene
from .model_builder import build_model
from .problem import BASSProblem
from .repair import repair_architecture

__all__ = [
    "ATTENTION_PRIMITIVES",
    "CHANNELS",
    "CNN_PRIMITIVES",
    "KERNEL_SIZES",
    "LEGACY_GENOME_BITS",
    "REPEATS",
    "SEMANTIC_GENOME_LENGTH",
    "UNIT_STATE_COUNT",
    "WINDOW_SIZES",
    "ArchitectureSpec",
    "BASSProblem",
    "BlockGene",
    "LegacyArchitectureSpec",
    "LegacyBlockGene",
    "block_to_state",
    "build_model",
    "canonicalize_architecture",
    "canonicalize_genome",
    "decode",
    "decode_legacy_bits",
    "decode_v2_bits",
    "encode",
    "encode_legacy_bits",
    "encode_v2_bits",
    "migrate_legacy93",
    "migrate_v1",
    "repair_architecture",
    "sample",
    "sample_genome",
    "sample_v2",
    "state_to_block",
    "upgrade_v1",
]
