"""Gray-code codec for the frozen 84-bit BASS V1 genome."""

from __future__ import annotations

from collections.abc import Sequence

from .config import (
    CHANNELS,
    CNN_PRIMITIVES,
    DECODED_VALUES,
    GENOME_BITS,
    KERNEL_SIZES,
    REPEATS,
)
from .genotype import ArchitectureSpec, BlockGene, architecture_from_blocks


def gray_to_int(gray_code: str | Sequence[int]) -> int:
    bits = [int(bit) for bit in gray_code]
    if not bits or any(bit not in {0, 1} for bit in bits):
        raise ValueError("Gray code must contain at least one binary digit")
    binary = [bits[0]]
    for bit in bits[1:]:
        binary.append(bit ^ binary[-1])
    return int("".join(str(bit) for bit in binary), 2)


def int_to_gray_bits(value: int, width: int) -> list[int]:
    if value < 0 or value >= 2**width:
        raise ValueError(f"{value} cannot be represented with {width} Gray bits")
    gray = value ^ (value >> 1)
    return [int(bit) for bit in f"{gray:0{width}b}"]


def bits_to_values(bits: Sequence[int], width: int = 3) -> list[int]:
    normalized = [int(bit) for bit in bits]
    if len(normalized) % width:
        raise ValueError(f"Bit length must be divisible by {width}")
    return [
        gray_to_int(normalized[index : index + width])
        for index in range(0, len(normalized), width)
    ]


def values_to_bits(values: Sequence[int], width: int = 3) -> list[int]:
    output: list[int] = []
    for value in values:
        output.extend(int_to_gray_bits(int(value), width))
    return output


bstr_to_rstr = bits_to_values


def decode_gene(gene: Sequence[int]) -> ArchitectureSpec:
    values = [int(value) for value in gene]
    if len(values) != DECODED_VALUES:
        raise ValueError(
            f"A decoded V1 gene requires {DECODED_VALUES} values, got {len(values)}"
        )

    channels = CHANNELS[values[0] % len(CHANNELS)]
    blocks = []
    for index in range(1, len(values), 3):
        op_index, kernel_index, repeat_index = values[index : index + 3]
        op = CNN_PRIMITIVES[op_index % len(CNN_PRIMITIVES)]
        if op == "identity":
            blocks.append(BlockGene(op="identity", arg=1, repeat=1))
        else:
            blocks.append(
                BlockGene(
                    op=op,
                    arg=KERNEL_SIZES[kernel_index % len(KERNEL_SIZES)],
                    repeat=REPEATS[repeat_index % len(REPEATS)],
                )
            )
    return architecture_from_blocks(channels, blocks)


def decode_bits(bits: Sequence[int]) -> ArchitectureSpec:
    normalized = list(bits)
    if len(normalized) != GENOME_BITS:
        raise ValueError(f"A V1 chromosome requires {GENOME_BITS} bits")
    if any(value not in {0, 1} for value in normalized):
        raise ValueError("A V1 chromosome may only contain binary values")
    return decode_gene(bits_to_values(normalized, width=3))


def decode(genome: ArchitectureSpec | Sequence[int]) -> ArchitectureSpec:
    if isinstance(genome, ArchitectureSpec):
        return genome
    values = list(genome)
    if len(values) == GENOME_BITS:
        return decode_bits(values)
    if len(values) == DECODED_VALUES:
        return decode_gene(values)
    raise ValueError("BASS V1 expects 84 bits or 28 decoded values")
