"""Canonical, versioned architecture representation for BASS."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Any, Literal

from .config import BRANCH_COUNT, UNITS_PER_BRANCH

Family = Literal["cnn", "attention"]


@dataclass(frozen=True, slots=True)
class BlockGene:
    """One spatially shape-preserving searchable unit.

    ``arg`` is operator-specific: a kernel size for CNN operators, a window
    size for spatial-attention operators, and zero for channel attention.
    """

    family: Family
    op: str
    arg: int
    repeat: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> BlockGene:
        return cls(
            family=str(payload["family"]),
            op=str(payload["op"]),
            arg=int(payload["arg"]),
            repeat=int(payload["repeat"]),
        )


@dataclass(frozen=True, slots=True)
class ArchitectureSpec:
    """Canonical BASS phenotype.

    The macro-topology is intentionally fixed to three independently searched
    branches with three units each. All units preserve H, W and C so branch
    addition remains valid.
    """

    channels: int
    branches: tuple[tuple[BlockGene, ...], ...]
    schema_version: int = 2

    def __post_init__(self) -> None:
        if self.schema_version not in {1, 2}:
            raise ValueError("schema_version must be 1 or 2")
        normalized = tuple(tuple(branch) for branch in self.branches)
        object.__setattr__(self, "branches", normalized)
        if len(normalized) != BRANCH_COUNT:
            raise ValueError(f"BASS requires {BRANCH_COUNT} branches")
        if any(len(branch) != UNITS_PER_BRANCH for branch in normalized):
            raise ValueError(
                f"Each branch must contain {UNITS_PER_BRANCH} searchable units"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "channels": self.channels,
            "branches": [
                [block.to_dict() for block in branch] for branch in self.branches
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ArchitectureSpec:
        return cls(
            schema_version=int(payload.get("schema_version", 2)),
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

    @property
    def flat_blocks(self) -> tuple[BlockGene, ...]:
        return tuple(block for branch in self.branches for block in branch)

    @property
    def attention_fraction(self) -> float:
        blocks = self.flat_blocks
        return sum(block.family == "attention" for block in blocks) / len(blocks)


def architecture_from_blocks(
    channels: int,
    blocks: Iterable[BlockGene],
    *,
    schema_version: int = 2,
) -> ArchitectureSpec:
    """Build an architecture from nine blocks in branch-major order."""

    flat = tuple(blocks)
    expected = BRANCH_COUNT * UNITS_PER_BRANCH
    if len(flat) != expected:
        raise ValueError(f"Expected {expected} blocks, got {len(flat)}")
    branches = tuple(
        flat[index : index + UNITS_PER_BRANCH]
        for index in range(0, expected, UNITS_PER_BRANCH)
    )
    return ArchitectureSpec(
        channels=channels,
        branches=branches,
        schema_version=schema_version,
    )
