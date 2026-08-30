"""Backward-compatible Gray-code codecs for BASS V1 and hybrid BASS V2."""

from __future__ import annotations

import random
from collections.abc import Sequence

from .config import (
    ATTENTION_PRIMITIVES,
    BRANCH_COUNT,
    CHANNELS,
    CNN_PRIMITIVES,
    KERNEL_SIZES,
    REPEATS,
    UNITS_PER_BRANCH,
    V1_DECODED_VALUES,
    V1_GENOME_BITS,
    V2_GENOME_BITS,
    WINDOW_SIZES,
)
from .genotype import ArchitectureSpec, BlockGene, architecture_from_blocks
from .repair import repair_architecture


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


# Historical public name retained for users of the original implementation.
bstr_to_rstr = bits_to_values


def decode_v1_gene(gene: Sequence[int]) -> ArchitectureSpec:
    values = [int(value) for value in gene]
    if len(values) != V1_DECODED_VALUES:
        raise ValueError(
            f"A decoded V1 gene requires {V1_DECODED_VALUES} values, got {len(values)}"
        )

    channels = CHANNELS[values[0] % len(CHANNELS)]
    unit_values = values[1:]
    blocks = []
    for index in range(0, len(unit_values), 3):
        op_index, kernel_index, repeat_index = unit_values[index : index + 3]
        blocks.append(
            BlockGene(
                family="cnn",
                op=CNN_PRIMITIVES[op_index % len(CNN_PRIMITIVES)],
                arg=KERNEL_SIZES[kernel_index % len(KERNEL_SIZES)],
                repeat=REPEATS[repeat_index % len(REPEATS)],
            )
        )
    return repair_architecture(
        architecture_from_blocks(channels, blocks, schema_version=1)
    )


def decode_v1_bits(bits: Sequence[int]) -> ArchitectureSpec:
    if len(bits) != V1_GENOME_BITS:
        raise ValueError(f"A V1 chromosome requires {V1_GENOME_BITS} bits")
    return decode_v1_gene(bits_to_values(bits, width=3))


def decode_v2_bits(bits: Sequence[int]) -> ArchitectureSpec:
    normalized = [int(bit) for bit in bits]
    if len(normalized) != V2_GENOME_BITS:
        raise ValueError(f"A V2 chromosome requires {V2_GENOME_BITS} bits")
    if any(bit not in {0, 1} for bit in normalized):
        raise ValueError("A V2 chromosome may only contain binary values")

    channel_value = gray_to_int(normalized[:3])
    channels = CHANNELS[channel_value % len(CHANNELS)]
    cursor = 3
    blocks = []
    for _ in range(BRANCH_COUNT * UNITS_PER_BRANCH):
        family_bit = normalized[cursor]
        op_index = gray_to_int(normalized[cursor + 1 : cursor + 4])
        arg_index = gray_to_int(normalized[cursor + 4 : cursor + 7])
        repeat_index = gray_to_int(normalized[cursor + 7 : cursor + 10])
        cursor += 10

        if family_bit == 0:
            blocks.append(
                BlockGene(
                    "cnn",
                    CNN_PRIMITIVES[op_index % len(CNN_PRIMITIVES)],
                    KERNEL_SIZES[arg_index % len(KERNEL_SIZES)],
                    REPEATS[repeat_index % len(REPEATS)],
                )
            )
        else:
            op = ATTENTION_PRIMITIVES[op_index % len(ATTENTION_PRIMITIVES)]
            arg = 0 if op == "channel_attention" else WINDOW_SIZES[arg_index % 2]
            blocks.append(
                BlockGene(
                    "attention",
                    op,
                    arg,
                    REPEATS[repeat_index % len(REPEATS)],
                )
            )

    return repair_architecture(architecture_from_blocks(channels, blocks))


def encode_v2_bits(spec: ArchitectureSpec) -> list[int]:
    spec = repair_architecture(spec)
    channel_index = CHANNELS.index(spec.channels)
    bits = int_to_gray_bits(channel_index, 3)

    for block in spec.flat_blocks:
        family_bit = 1 if block.family == "attention" else 0
        if family_bit:
            op_index = ATTENTION_PRIMITIVES.index(block.op)
            arg_index = (
                0 if block.op == "channel_attention" else WINDOW_SIZES.index(block.arg)
            )
        else:
            op_index = CNN_PRIMITIVES.index(block.op)
            arg_index = KERNEL_SIZES.index(block.arg)
        repeat_index = REPEATS.index(block.repeat)

        bits.append(family_bit)
        bits.extend(int_to_gray_bits(op_index, 3))
        bits.extend(int_to_gray_bits(arg_index, 3))
        bits.extend(int_to_gray_bits(repeat_index, 3))

    if len(bits) != V2_GENOME_BITS:
        raise AssertionError("Internal V2 codec length mismatch")
    return bits


def decode(genome: ArchitectureSpec | Sequence[int]) -> ArchitectureSpec:
    """Decode a V1/V2 chromosome or the historical 28-value V1 gene."""

    if isinstance(genome, ArchitectureSpec):
        return repair_architecture(genome)
    values = [int(value) for value in genome]
    if len(values) == V1_GENOME_BITS:
        return decode_v1_bits(values)
    if len(values) == V2_GENOME_BITS:
        return decode_v2_bits(values)
    if len(values) == V1_DECODED_VALUES:
        return decode_v1_gene(values)
    raise ValueError(
        "Expected an ArchitectureSpec, 84 V1 bits, 93 V2 bits, or 28 decoded V1 values"
    )


def upgrade_v1(genome: Sequence[int]) -> ArchitectureSpec:
    """Upgrade an old chromosome/gene without changing its phenotype."""

    values = list(genome)
    if len(values) == V1_GENOME_BITS:
        legacy = decode_v1_bits(values)
    else:
        legacy = decode_v1_gene(values)
    return ArchitectureSpec(
        channels=legacy.channels,
        branches=legacy.branches,
        schema_version=2,
    )


def sample_v2(
    *, seed: int | None = None, attention_probability: float = 0.5
) -> ArchitectureSpec:
    if not 0.0 <= attention_probability <= 1.0:
        raise ValueError("attention_probability must lie in [0, 1]")
    rng = random.Random(seed)
    blocks = []
    for _ in range(BRANCH_COUNT * UNITS_PER_BRANCH):
        if rng.random() < attention_probability:
            op = rng.choice(ATTENTION_PRIMITIVES)
            arg = 0 if op == "channel_attention" else rng.choice(WINDOW_SIZES)
            blocks.append(BlockGene("attention", op, arg, rng.choice(REPEATS)))
        else:
            op = rng.choice(CNN_PRIMITIVES)
            arg = 1 if op == "identity" else rng.choice(KERNEL_SIZES)
            repeat = 1 if op == "identity" else rng.choice(REPEATS)
            blocks.append(BlockGene("cnn", op, arg, repeat))
    return repair_architecture(architecture_from_blocks(rng.choice(CHANNELS), blocks))
