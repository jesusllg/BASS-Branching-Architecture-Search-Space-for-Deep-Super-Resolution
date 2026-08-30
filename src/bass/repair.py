"""Deterministic canonicalization and repair for hybrid BASS genotypes."""

from __future__ import annotations

from .config import (
    ATTENTION_PRIMITIVES,
    CHANNELS,
    CNN_PRIMITIVES,
    KERNEL_SIZES,
    REPEATS,
    WINDOW_SIZES,
)
from .genotype import ArchitectureSpec, BlockGene


def _nearest(value: int, choices: tuple[int, ...]) -> int:
    return min(choices, key=lambda candidate: (abs(candidate - value), candidate))


def repair_block(block: BlockGene) -> BlockGene:
    family = block.family if block.family in {"cnn", "attention"} else "cnn"
    repeat = _nearest(int(block.repeat), REPEATS)

    if family == "cnn":
        op = block.op if block.op in CNN_PRIMITIVES else "identity"
        if op == "identity":
            return BlockGene("cnn", "identity", 1, 1)
        kernel = _nearest(int(block.arg), KERNEL_SIZES)
        return BlockGene("cnn", op, kernel, repeat)

    op = block.op if block.op in ATTENTION_PRIMITIVES else "channel_attention"
    if op == "channel_attention":
        return BlockGene("attention", op, 0, repeat)
    window = _nearest(int(block.arg), WINDOW_SIZES)
    return BlockGene("attention", op, window, repeat)


def repair_architecture(spec: ArchitectureSpec) -> ArchitectureSpec:
    """Return a canonical, buildable architecture.

    Repair is deliberately idempotent. Resource limits belong to the search
    runner; this function only enforces the BASS structural contract.
    """

    channels = _nearest(int(spec.channels), CHANNELS)
    branches = tuple(
        tuple(repair_block(block) for block in branch) for branch in spec.branches
    )
    return ArchitectureSpec(
        channels=channels,
        branches=branches,
        schema_version=spec.schema_version,
    )
