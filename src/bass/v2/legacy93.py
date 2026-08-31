"""Import/export-only codec for the superseded 93-bit BASS V2 prototype.

This module deliberately does not participate in scientific search. It exists
so previously stored chromosomes remain inspectable and round-trippable.
"""

from __future__ import annotations

from dataclasses import dataclass

LEGACY_CHANNELS = (16, 32, 48, 64)
LEGACY_CNN_PRIMITIVES = (
    "conv",
    "dil_conv_d2",
    "dil_conv_d3",
    "dil_conv_d4",
    "depthwise_separable_conv",
    "inverted_bottleneck_e2",
    "conv_transpose",
    "identity",
)
LEGACY_ATTENTION_PRIMITIVES = (
    "channel_attention",
    "window_attention",
    "shifted_window_attention",
    "hybrid_conv_attention",
)
LEGACY_KERNELS = (1, 3, 5, 7)
LEGACY_REPEATS = (1, 2, 3, 4)
LEGACY_WINDOWS = (4, 8)
LEGACY_GENOME_BITS = 93


@dataclass(frozen=True, slots=True)
class LegacyBlockGene:
    family: str
    op: str
    arg: int
    repeat: int


@dataclass(frozen=True, slots=True)
class LegacyArchitectureSpec:
    channels: int
    branches: tuple[tuple[LegacyBlockGene, ...], ...]
    schema_version: int = 2
    representation: str = "legacy93"

    @property
    def flat_blocks(self) -> tuple[LegacyBlockGene, ...]:
        return tuple(block for branch in self.branches for block in branch)


def _gray_to_int(bits) -> int:
    normalized = [int(bit) for bit in bits]
    binary = [normalized[0]]
    for bit in normalized[1:]:
        binary.append(bit ^ binary[-1])
    return int("".join(str(bit) for bit in binary), 2)


def _int_to_gray_bits(value: int, width: int = 3) -> list[int]:
    gray = value ^ (value >> 1)
    return [int(bit) for bit in f"{gray:0{width}b}"]


def decode_legacy_bits(bits) -> LegacyArchitectureSpec:
    normalized = [int(bit) for bit in bits]
    if len(normalized) != LEGACY_GENOME_BITS:
        raise ValueError("A legacy V2 chromosome requires 93 bits")
    if any(bit not in {0, 1} for bit in normalized):
        raise ValueError("A legacy V2 chromosome may only contain binary values")

    channels = LEGACY_CHANNELS[_gray_to_int(normalized[:3]) % 4]
    blocks = []
    cursor = 3
    for _ in range(9):
        family_bit = normalized[cursor]
        op_index = _gray_to_int(normalized[cursor + 1 : cursor + 4])
        arg_index = _gray_to_int(normalized[cursor + 4 : cursor + 7])
        repeat_index = _gray_to_int(normalized[cursor + 7 : cursor + 10])
        cursor += 10
        repeat = LEGACY_REPEATS[repeat_index % 4]
        if family_bit == 0:
            op = LEGACY_CNN_PRIMITIVES[op_index % 8]
            if op == "identity":
                blocks.append(LegacyBlockGene("cnn", op, 1, 1))
            else:
                blocks.append(
                    LegacyBlockGene("cnn", op, LEGACY_KERNELS[arg_index % 4], repeat)
                )
        else:
            op = LEGACY_ATTENTION_PRIMITIVES[op_index % 4]
            arg = 0 if op == "channel_attention" else LEGACY_WINDOWS[arg_index % 2]
            blocks.append(LegacyBlockGene("attention", op, arg, repeat))
    return LegacyArchitectureSpec(
        channels=channels,
        branches=tuple(tuple(blocks[i : i + 3]) for i in range(0, 9, 3)),
    )


def encode_legacy_bits(spec: LegacyArchitectureSpec) -> list[int]:
    if not isinstance(spec, LegacyArchitectureSpec):
        raise TypeError("encode_legacy_bits requires LegacyArchitectureSpec")
    bits = _int_to_gray_bits(LEGACY_CHANNELS.index(spec.channels))
    for block in spec.flat_blocks:
        attention = block.family == "attention"
        bits.append(int(attention))
        if attention:
            op_index = LEGACY_ATTENTION_PRIMITIVES.index(block.op)
            arg_index = 0 if block.arg == 0 else LEGACY_WINDOWS.index(block.arg)
        else:
            op_index = LEGACY_CNN_PRIMITIVES.index(block.op)
            arg_index = LEGACY_KERNELS.index(block.arg)
        bits.extend(_int_to_gray_bits(op_index))
        bits.extend(_int_to_gray_bits(arg_index))
        bits.extend(_int_to_gray_bits(LEGACY_REPEATS.index(block.repeat)))
    return bits
