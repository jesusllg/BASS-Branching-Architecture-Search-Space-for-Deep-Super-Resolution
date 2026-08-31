"""Strict and structurally canonical architecture objects for BASS V2."""

from __future__ import annotations

import json
import numbers
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Any, Literal

from .config import (
    BRANCH_COUNT,
    CHANNELS,
    PRIMITIVE_CONFIGS,
    REPEATS,
    UNITS_PER_BRANCH,
)

Family = Literal["skip", "cnn", "attention"]
_VALID_CONFIGS = frozenset(PRIMITIVE_CONFIGS)


@dataclass(frozen=True, slots=True)
class BlockGene:
    """One complete semantic state; ``arg`` is never conditionally inactive."""

    family: Family
    op: str
    arg: int
    repeat: int

    def __post_init__(self) -> None:
        if not isinstance(self.family, str) or not isinstance(self.op, str):
            raise TypeError("V2 block family and operation must be strings")
        if (
            isinstance(self.arg, bool)
            or not isinstance(self.arg, numbers.Integral)
            or isinstance(self.repeat, bool)
            or not isinstance(self.repeat, numbers.Integral)
        ):
            raise TypeError("V2 block arg and repeat must be integers")
        object.__setattr__(self, "arg", int(self.arg))
        object.__setattr__(self, "repeat", int(self.repeat))
        if self.family == "skip":
            if (self.op, self.arg, self.repeat) != ("skip", 0, 0):
                raise ValueError("skip must be encoded exactly as skip/0/0")
            return
        if (self.family, self.op, self.arg) not in _VALID_CONFIGS:
            raise ValueError(
                "Unsupported V2 primitive configuration: "
                f"{self.family}/{self.op}/{self.arg}"
            )
        if self.repeat not in REPEATS:
            raise ValueError(f"V2 repeat must be one of {REPEATS}")

    @classmethod
    def skip(cls) -> BlockGene:
        return cls("skip", "skip", 0, 0)

    @property
    def is_skip(self) -> bool:
        return self.family == "skip"

    @property
    def operation_key(self) -> tuple[str, str, int]:
        return self.family, self.op, self.arg

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> BlockGene:
        expected = {"family", "op", "arg", "repeat"}
        if set(payload) != expected:
            raise ValueError(f"V2 block fields must be exactly {sorted(expected)}")
        return cls(
            family=payload["family"],
            op=payload["op"],
            arg=payload["arg"],
            repeat=payload["repeat"],
        )


def _compress_branch(blocks: Sequence[BlockGene]) -> tuple[BlockGene, ...]:
    """Remove skips and canonicalize equivalent adjacent repeat groupings."""

    active = [block for block in blocks if not block.is_skip]
    compressed: list[BlockGene] = []
    index = 0
    while index < len(active):
        template = active[index]
        total_repeat = template.repeat
        index += 1
        while (
            index < len(active)
            and active[index].operation_key == template.operation_key
        ):
            total_repeat += active[index].repeat
            index += 1
        while total_repeat:
            repeat = min(max(REPEATS), total_repeat)
            compressed.append(
                BlockGene(template.family, template.op, template.arg, repeat)
            )
            total_repeat -= repeat

    if len(compressed) > UNITS_PER_BRANCH:
        raise ValueError(
            "Canonical branch exceeds three units; reduce its effective depth"
        )
    return tuple(compressed + [BlockGene.skip()] * (UNITS_PER_BRANCH - len(compressed)))


def canonicalize_branches(
    branches: Iterable[Iterable[BlockGene]],
) -> tuple[tuple[BlockGene, ...], ...]:
    """Pack skips, normalize repeats, then quotient branch permutations."""

    normalized = tuple(_compress_branch(tuple(branch)) for branch in branches)
    if len(normalized) != BRANCH_COUNT:
        raise ValueError(f"BASS V2 requires {BRANCH_COUNT} branches")
    return tuple(
        sorted(
            normalized,
            key=lambda branch: tuple(
                (block.family, block.op, block.arg, block.repeat) for block in branch
            ),
        )
    )


@dataclass(frozen=True, slots=True)
class ArchitectureSpec:
    """One canonical V2 graph description.

    Construction is strict: persisted or research architectures are rejected
    when non-canonical instead of being silently changed.
    """

    channels: int
    branches: tuple[tuple[BlockGene, ...], ...]
    schema_version: int = 2
    representation: str = "semantic-v1"

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, numbers.Integral)
            or isinstance(self.channels, bool)
            or not isinstance(self.channels, numbers.Integral)
        ):
            raise TypeError("V2 schema version and channels must be integers")
        object.__setattr__(self, "schema_version", int(self.schema_version))
        object.__setattr__(self, "channels", int(self.channels))
        if self.schema_version != 2:
            raise ValueError("A V2 architecture must use schema_version=2")
        if self.representation != "semantic-v1":
            raise ValueError("Unsupported V2 representation")
        if self.channels not in CHANNELS:
            raise ValueError(f"Unsupported V2 channel count: {self.channels}")
        normalized = tuple(tuple(branch) for branch in self.branches)
        if len(normalized) != BRANCH_COUNT:
            raise ValueError(f"BASS V2 requires {BRANCH_COUNT} branches")
        if any(len(branch) != UNITS_PER_BRANCH for branch in normalized):
            raise ValueError(
                f"Each V2 branch must contain {UNITS_PER_BRANCH} unit slots"
            )
        canonical = canonicalize_branches(normalized)
        if normalized != canonical:
            raise ValueError(
                "Architecture is not canonical; use canonicalize_architecture()"
            )
        object.__setattr__(self, "branches", normalized)

    @property
    def flat_blocks(self) -> tuple[BlockGene, ...]:
        return tuple(block for branch in self.branches for block in branch)

    @property
    def active_blocks(self) -> tuple[BlockGene, ...]:
        return tuple(block for block in self.flat_blocks if not block.is_skip)

    @property
    def attention_fraction(self) -> float:
        """Fraction of active unit slots tagged as attention, not FLOP share."""

        blocks = self.active_blocks
        if not blocks:
            return 0.0
        return sum(block.family == "attention" for block in blocks) / len(blocks)

    @property
    def operator_counts(self) -> dict[str, int]:
        return dict(Counter(block.op for block in self.active_blocks))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "representation": self.representation,
            "channels": self.channels,
            "branches": [
                [block.to_dict() for block in branch] for branch in self.branches
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ArchitectureSpec:
        expected = {"schema_version", "representation", "channels", "branches"}
        if set(payload) != expected:
            raise ValueError(
                f"V2 architecture fields must be exactly {sorted(expected)}"
            )
        return cls(
            schema_version=payload["schema_version"],
            representation=payload["representation"],
            channels=payload["channels"],
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


def canonicalize_architecture(
    channels: int,
    branches: Iterable[Iterable[BlockGene]],
) -> ArchitectureSpec:
    if isinstance(channels, bool) or not isinstance(channels, numbers.Integral):
        raise TypeError("V2 channels must be an integer")
    return ArchitectureSpec(
        channels=int(channels),
        branches=canonicalize_branches(branches),
    )


def architecture_from_blocks(
    channels: int, blocks: Iterable[BlockGene]
) -> ArchitectureSpec:
    flat = tuple(blocks)
    expected = BRANCH_COUNT * UNITS_PER_BRANCH
    if len(flat) != expected:
        raise ValueError(f"Expected {expected} V2 unit slots, got {len(flat)}")
    branches = tuple(
        flat[index : index + UNITS_PER_BRANCH]
        for index in range(0, expected, UNITS_PER_BRANCH)
    )
    return canonicalize_architecture(channels, branches)
