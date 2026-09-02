"""Strict interaction-aware architecture objects for BASS V3."""

from __future__ import annotations

import json
import numbers
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
from itertools import pairwise
from typing import Any, Literal

from bass.v2.genotype import BlockGene

from .config import (
    BRANCH_COUNT,
    CHANNELS,
    EXCHANGE_CONFIGS,
    EXCHANGE_SITES,
    REPEATS,
    UNITS_PER_BRANCH,
)

ExchangeOperation = Literal["none", "cimex"]
_VALID_EXCHANGES = frozenset(EXCHANGE_CONFIGS)


def _branch_key(branch: Sequence[BlockGene]) -> tuple[tuple[str, str, int, int], ...]:
    return tuple((block.family, block.op, block.arg, block.repeat) for block in branch)


def _compress_segment(blocks: Sequence[BlockGene]) -> tuple[BlockGene, ...]:
    """Canonicalize one sequential segment without crossing a CIMEX barrier."""

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

    if len(compressed) > len(blocks):
        raise ValueError(
            "Canonical V3 segment exceeds its available stage slots; "
            "reduce its effective depth"
        )
    return tuple(compressed + [BlockGene.skip()] * (len(blocks) - len(compressed)))


def canonicalize_branch(
    blocks: Iterable[BlockGene],
    exchanges: Sequence[ExchangeGene],
) -> tuple[BlockGene, ...]:
    """Canonicalize one branch while treating enabled exchanges as barriers."""

    branch = tuple(blocks)
    if len(branch) != UNITS_PER_BRANCH:
        raise ValueError(f"An IBASS branch must contain {UNITS_PER_BRANCH} stage slots")
    if any(not isinstance(block, BlockGene) for block in branch):
        raise TypeError("An IBASS branch may contain only BlockGene objects")
    if len(exchanges) != EXCHANGE_SITES:
        raise ValueError(f"IBASS requires {EXCHANGE_SITES} exchange decisions")
    if any(not isinstance(exchange, ExchangeGene) for exchange in exchanges):
        raise TypeError("V3 exchanges must contain only ExchangeGene objects")

    boundaries = [0]
    boundaries.extend(
        site + 1 for site, exchange in enumerate(exchanges) if exchange.is_enabled
    )
    boundaries.append(UNITS_PER_BRANCH)

    normalized: list[BlockGene] = []
    for start, stop in pairwise(boundaries):
        normalized.extend(_compress_segment(branch[start:stop]))
    return tuple(normalized)


def _canonicalize_branches(
    branches: Iterable[Iterable[BlockGene]],
    exchanges: Sequence[ExchangeGene],
) -> tuple[tuple[BlockGene, ...], ...]:
    normalized = tuple(canonicalize_branch(branch, exchanges) for branch in branches)
    if len(normalized) != BRANCH_COUNT:
        raise ValueError(f"IBASS requires {BRANCH_COUNT} branches")
    return tuple(sorted(normalized, key=_branch_key))


def _canonicalize_exchanges(
    branches: tuple[tuple[BlockGene, ...], ...],
    exchanges: tuple[ExchangeGene, ...],
) -> tuple[ExchangeGene, ...]:
    """Disable exchange sites that are guaranteed to cancel before fusion."""

    if len(exchanges) != EXCHANGE_SITES:
        raise ValueError(f"IBASS requires {EXCHANGE_SITES} exchange decisions")
    normalized = list(exchanges)
    if all(branch[2].is_skip for branch in branches):
        normalized[1] = ExchangeGene.none()
    if all(branch[1].is_skip and branch[2].is_skip for branch in branches):
        normalized[0] = ExchangeGene.none()
    return tuple(normalized)


def _joint_canonicalize(
    branches: Iterable[Iterable[BlockGene]],
    exchanges: Iterable[ExchangeGene],
) -> tuple[tuple[tuple[BlockGene, ...], ...], tuple[ExchangeGene, ...]]:
    """Canonicalize stages and exchanges to a semantics-preserving fixed point."""

    current_branches = tuple(tuple(branch) for branch in branches)
    current_exchanges = tuple(exchanges)
    if len(current_exchanges) != EXCHANGE_SITES:
        raise ValueError(f"IBASS requires {EXCHANGE_SITES} exchange decisions")
    if any(not isinstance(exchange, ExchangeGene) for exchange in current_exchanges):
        raise TypeError("V3 exchanges must contain only ExchangeGene objects")

    # Exchanges can only change from enabled to none. Two sites therefore need
    # at most three passes, but the explicit bound guards future extensions.
    for _ in range(EXCHANGE_SITES + 1):
        normalized_branches = _canonicalize_branches(
            current_branches, current_exchanges
        )
        normalized_exchanges = _canonicalize_exchanges(
            normalized_branches, current_exchanges
        )
        if (
            normalized_branches == current_branches
            and normalized_exchanges == current_exchanges
        ):
            return normalized_branches, normalized_exchanges
        current_branches = normalized_branches
        current_exchanges = normalized_exchanges
    raise AssertionError("V3 joint canonicalization failed to reach a fixed point")


@dataclass(frozen=True, slots=True)
class ExchangeGene:
    """One complete cross-branch exchange decision."""

    op: ExchangeOperation
    prototypes: int

    def __post_init__(self) -> None:
        if not isinstance(self.op, str):
            raise TypeError("V3 exchange operation must be a string")
        if isinstance(self.prototypes, bool) or not isinstance(
            self.prototypes, numbers.Integral
        ):
            raise TypeError("V3 exchange prototype count must be an integer")
        object.__setattr__(self, "prototypes", int(self.prototypes))
        if (self.op, self.prototypes) not in _VALID_EXCHANGES:
            raise ValueError(
                f"Unsupported V3 exchange configuration: {self.op}/{self.prototypes}"
            )

    @classmethod
    def none(cls) -> ExchangeGene:
        return cls("none", 0)

    @classmethod
    def cimex(cls, prototypes: int) -> ExchangeGene:
        return cls("cimex", prototypes)

    @property
    def is_enabled(self) -> bool:
        return self.op == "cimex"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ExchangeGene:
        expected = {"op", "prototypes"}
        if set(payload) != expected:
            raise ValueError(f"V3 exchange fields must be exactly {sorted(expected)}")
        return cls(op=payload["op"], prototypes=payload["prototypes"])


@dataclass(frozen=True, slots=True)
class ArchitectureSpec:
    """One canonical IBASS graph with two searchable exchange sites."""

    channels: int
    branches: tuple[tuple[BlockGene, ...], ...]
    exchanges: tuple[ExchangeGene, ...]
    schema_version: int = 3
    representation: str = "interaction-semantic-v2"

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, numbers.Integral)
            or isinstance(self.channels, bool)
            or not isinstance(self.channels, numbers.Integral)
        ):
            raise TypeError("V3 schema version and channels must be integers")
        object.__setattr__(self, "schema_version", int(self.schema_version))
        object.__setattr__(self, "channels", int(self.channels))
        if self.schema_version != 3:
            raise ValueError("A V3 architecture must use schema_version=3")
        if self.representation != "interaction-semantic-v2":
            if self.representation == "interaction-semantic-v1":
                raise ValueError(
                    "interaction-semantic-v1 used a stage-unsafe quotient and "
                    "cannot be loaded as a corrected V3 architecture"
                )
            raise ValueError("Unsupported V3 representation")
        if self.channels not in CHANNELS:
            raise ValueError(f"Unsupported V3 channel count: {self.channels}")

        branches = tuple(tuple(branch) for branch in self.branches)
        if len(branches) != BRANCH_COUNT:
            raise ValueError(f"IBASS requires {BRANCH_COUNT} branches")
        if any(len(branch) != UNITS_PER_BRANCH for branch in branches):
            raise ValueError(
                f"Each IBASS branch must contain {UNITS_PER_BRANCH} unit slots"
            )
        exchanges = tuple(self.exchanges)
        if len(exchanges) != EXCHANGE_SITES:
            raise ValueError(f"IBASS requires {EXCHANGE_SITES} exchange decisions")
        if any(not isinstance(exchange, ExchangeGene) for exchange in exchanges):
            raise TypeError("V3 exchanges must contain only ExchangeGene objects")
        canonical_branches, canonical_exchanges = _joint_canonicalize(
            branches, exchanges
        )
        if branches != canonical_branches or exchanges != canonical_exchanges:
            raise ValueError(
                "V3 architecture is not jointly canonical; enabled exchanges "
                "are hard stage barriers and algebraically inactive exchanges "
                "must be removed with canonicalize_architecture()"
            )
        object.__setattr__(self, "branches", branches)
        object.__setattr__(self, "exchanges", exchanges)

    @property
    def flat_blocks(self) -> tuple[BlockGene, ...]:
        return tuple(block for branch in self.branches for block in branch)

    @property
    def active_blocks(self) -> tuple[BlockGene, ...]:
        return tuple(block for block in self.flat_blocks if not block.is_skip)

    @property
    def attention_fraction(self) -> float:
        blocks = self.active_blocks
        if not blocks:
            return 0.0
        return sum(block.family == "attention" for block in blocks) / len(blocks)

    @property
    def exchange_count(self) -> int:
        return sum(exchange.is_enabled for exchange in self.exchanges)

    @property
    def uses_cimex(self) -> bool:
        return self.exchange_count > 0

    @property
    def operator_counts(self) -> dict[str, int]:
        counts = Counter(block.op for block in self.active_blocks)
        counts.update(
            f"cimex_k{exchange.prototypes}"
            for exchange in self.exchanges
            if exchange.is_enabled
        )
        return dict(counts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "representation": self.representation,
            "channels": self.channels,
            "branches": [
                [block.to_dict() for block in branch] for branch in self.branches
            ],
            "exchanges": [exchange.to_dict() for exchange in self.exchanges],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ArchitectureSpec:
        expected = {
            "schema_version",
            "representation",
            "channels",
            "branches",
            "exchanges",
        }
        if set(payload) != expected:
            raise ValueError(
                f"V3 architecture fields must be exactly {sorted(expected)}"
            )
        return cls(
            schema_version=payload["schema_version"],
            representation=payload["representation"],
            channels=payload["channels"],
            branches=tuple(
                tuple(BlockGene.from_dict(block) for block in branch)
                for branch in payload["branches"]
            ),
            exchanges=tuple(
                ExchangeGene.from_dict(exchange) for exchange in payload["exchanges"]
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
    exchanges: Iterable[ExchangeGene],
) -> ArchitectureSpec:
    if isinstance(channels, bool) or not isinstance(channels, numbers.Integral):
        raise TypeError("V3 channels must be an integer")
    normalized_branches, normalized_exchanges = _joint_canonicalize(branches, exchanges)
    return ArchitectureSpec(
        channels=int(channels),
        branches=normalized_branches,
        exchanges=normalized_exchanges,
    )


def architecture_from_blocks(
    channels: int,
    blocks: Iterable[BlockGene],
    exchanges: Iterable[ExchangeGene],
) -> ArchitectureSpec:
    flat = tuple(blocks)
    expected = BRANCH_COUNT * UNITS_PER_BRANCH
    if len(flat) != expected:
        raise ValueError(f"Expected {expected} V3 unit slots, got {len(flat)}")
    branches = tuple(
        flat[index : index + UNITS_PER_BRANCH]
        for index in range(0, expected, UNITS_PER_BRANCH)
    )
    return canonicalize_architecture(channels, branches, exchanges)
