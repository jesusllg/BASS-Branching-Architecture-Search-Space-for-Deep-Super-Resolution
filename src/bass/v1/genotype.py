"""Canonical CNN-only architecture representation for BASS V1."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Any

from .config import (
    BRANCH_COUNT,
    CHANNELS,
    CNN_PRIMITIVES,
    KERNEL_SIZES,
    REPEATS,
    UNITS_PER_BRANCH,
)


@dataclass(frozen=True, slots=True)
class BlockGene:
    """One V1 convolutional unit."""

    op: str
    arg: int
    repeat: int

    def __post_init__(self) -> None:
        if self.op not in CNN_PRIMITIVES:
            raise ValueError(f"Unknown V1 primitive: {self.op}")
        if self.arg not in KERNEL_SIZES:
            raise ValueError(f"Unsupported V1 kernel size: {self.arg}")
        if self.repeat not in REPEATS:
            raise ValueError(f"Unsupported V1 repeat count: {self.repeat}")
        if self.op == "identity" and (self.arg != 1 or self.repeat != 1):
            raise ValueError("V1 identity is canonicalized to arg=1 and repeat=1")

    @property
    def family(self) -> str:
        return "cnn"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> BlockGene:
        return cls(
            op=str(payload["op"]),
            arg=int(payload["arg"]),
            repeat=int(payload["repeat"]),
        )


@dataclass(frozen=True, slots=True)
class ArchitectureSpec:
    """A frozen three-branch BASS V1 phenotype."""

    channels: int
    branches: tuple[tuple[BlockGene, ...], ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("A V1 architecture must use schema_version=1")
        if self.channels not in CHANNELS:
            raise ValueError(f"Unsupported V1 channel count: {self.channels}")
        normalized = tuple(tuple(branch) for branch in self.branches)
        object.__setattr__(self, "branches", normalized)
        if len(normalized) != BRANCH_COUNT:
            raise ValueError(f"BASS V1 requires {BRANCH_COUNT} branches")
        if any(len(branch) != UNITS_PER_BRANCH for branch in normalized):
            raise ValueError(
                f"Each V1 branch must contain {UNITS_PER_BRANCH} searchable units"
            )

    @property
    def flat_blocks(self) -> tuple[BlockGene, ...]:
        return tuple(block for branch in self.branches for block in branch)

    @property
    def attention_fraction(self) -> float:
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "channels": self.channels,
            "branches": [
                [block.to_dict() for block in branch] for branch in self.branches
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ArchitectureSpec:
        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            channels=int(payload["channels"]),
            branches=tuple(
                tuple(BlockGene.from_dict(block) for block in branch)
                for branch in payload["branches"]
            ),
        )

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )

    def canonical_hash(self) -> str:
        return sha256(self.canonical_json().encode("utf-8")).hexdigest()


def architecture_from_blocks(
    channels: int, blocks: Iterable[BlockGene]
) -> ArchitectureSpec:
    flat = tuple(blocks)
    expected = BRANCH_COUNT * UNITS_PER_BRANCH
    if len(flat) != expected:
        raise ValueError(f"Expected {expected} V1 blocks, got {len(flat)}")
    branches = tuple(
        flat[index : index + UNITS_PER_BRANCH]
        for index in range(0, expected, UNITS_PER_BRANCH)
    )
    return ArchitectureSpec(channels=channels, branches=branches)
