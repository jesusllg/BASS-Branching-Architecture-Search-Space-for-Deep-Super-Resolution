"""Compatibility dispatcher for explicit :mod:`bass.v1` and :mod:`bass.v2`."""

from __future__ import annotations

from collections.abc import Sequence

from .v1.encoding import (
    bits_to_values,
    gray_to_int,
    int_to_gray_bits,
    values_to_bits,
)
from .v1.encoding import (
    decode_bits as decode_v1_bits,
)
from .v1.encoding import (
    decode_gene as decode_v1_gene,
)
from .v1.genotype import ArchitectureSpec as V1ArchitectureSpec
from .v2.encoding import (
    canonicalize_genome,
    upgrade_v1,
)
from .v2.encoding import (
    decode as decode_v2,
)
from .v2.encoding import (
    decode_bits as decode_v2_bits,
)
from .v2.encoding import (
    encode as encode_v2,
)
from .v2.encoding import (
    encode_bits as encode_v2_bits,
)
from .v2.encoding import (
    sample as sample_v2,
)
from .v2.genotype import ArchitectureSpec as V2ArchitectureSpec

__all__ = [
    "bits_to_values",
    "bstr_to_rstr",
    "canonicalize_genome",
    "decode",
    "decode_v1_bits",
    "decode_v1_gene",
    "decode_v2",
    "decode_v2_bits",
    "encode_v2",
    "encode_v2_bits",
    "gray_to_int",
    "int_to_gray_bits",
    "sample_v2",
    "upgrade_v1",
    "values_to_bits",
]

bstr_to_rstr = bits_to_values


def decode(
    genome: V1ArchitectureSpec | V2ArchitectureSpec | Sequence[int],
) -> V1ArchitectureSpec | V2ArchitectureSpec:
    """Dispatch legacy callers by explicit architecture type or genome length."""

    if isinstance(genome, V1ArchitectureSpec):
        return genome
    if isinstance(genome, V2ArchitectureSpec):
        return genome
    values = list(genome)
    if len(values) in {28, 84}:
        return decode_v1_gene(values) if len(values) == 28 else decode_v1_bits(values)
    if len(values) == 10:
        return decode_v2(values)
    if len(values) == 93:
        return decode_v2_bits(values)
    raise ValueError(
        "Expected V1 84-bit/28-value, V2 10-integer, or legacy V2 93-bit input"
    )
