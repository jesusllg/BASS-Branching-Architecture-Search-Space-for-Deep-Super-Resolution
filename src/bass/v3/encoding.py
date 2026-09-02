"""Canonical 12-integer codec for interaction-aware BASS V3."""

from __future__ import annotations

import numbers
import random
from collections.abc import Sequence
from functools import lru_cache
from itertools import product
from math import comb

from bass.v2.encoding import (
    block_to_state,
    state_to_block,
)
from bass.v2.encoding import (
    decode as decode_v2,
)
from bass.v2.genotype import ArchitectureSpec as V2ArchitectureSpec
from bass.v2.genotype import canonicalize_architecture as canonicalize_v2

from .config import (
    ATTENTION_PRIMITIVE_CONFIGS,
    BRANCH_COUNT,
    CHANNELS,
    CNN_PRIMITIVE_CONFIGS,
    DEFAULT_EXCHANGE_PROBABILITY,
    EXCHANGE_CONFIGS,
    EXCHANGE_SITES,
    REPEATS,
    SEMANTIC_GENOME_LENGTH,
    UNIT_STATE_COUNT,
    UNITS_PER_BRANCH,
)
from .genotype import (
    ArchitectureSpec,
    BlockGene,
    ExchangeGene,
    architecture_from_blocks,
    canonicalize_architecture,
    canonicalize_branch,
)


def state_to_exchange(state: int) -> ExchangeGene:
    if isinstance(state, bool) or not isinstance(state, numbers.Integral):
        raise TypeError(
            f"Exchange state must be an integer in [0, {len(EXCHANGE_CONFIGS) - 1}]"
        )
    value = int(state)
    if not 0 <= value < len(EXCHANGE_CONFIGS):
        raise ValueError(
            f"Exchange state must be an integer in [0, {len(EXCHANGE_CONFIGS) - 1}]"
        )
    op, prototypes = EXCHANGE_CONFIGS[value]
    return ExchangeGene(op, prototypes)


def exchange_to_state(exchange: ExchangeGene) -> int:
    if not isinstance(exchange, ExchangeGene):
        raise TypeError("exchange_to_state requires a V3 ExchangeGene")
    return EXCHANGE_CONFIGS.index((exchange.op, exchange.prototypes))


def encode(spec: ArchitectureSpec) -> list[int]:
    if not isinstance(spec, ArchitectureSpec):
        raise TypeError("encode requires a canonical bass.v3 ArchitectureSpec")
    return (
        [CHANNELS.index(spec.channels)]
        + [block_to_state(block) for block in spec.flat_blocks]
        + [exchange_to_state(exchange) for exchange in spec.exchanges]
    )


def canonicalize_genome(genome: Sequence[int]) -> list[int]:
    values = list(genome)
    if len(values) != SEMANTIC_GENOME_LENGTH:
        raise ValueError(
            f"A semantic V3 genome requires {SEMANTIC_GENOME_LENGTH} integers"
        )
    if isinstance(values[0], bool) or not isinstance(values[0], numbers.Integral):
        raise TypeError("V3 channel state must be an integer in [0, 3]")
    channel_id = int(values[0])
    if not 0 <= channel_id < len(CHANNELS):
        raise ValueError("V3 channel state must be an integer in [0, 3]")
    unit_end = 1 + BRANCH_COUNT * UNITS_PER_BRANCH
    blocks = [state_to_block(value) for value in values[1:unit_end]]
    exchanges = [state_to_exchange(value) for value in values[unit_end:]]
    return encode(architecture_from_blocks(CHANNELS[channel_id], blocks, exchanges))


def decode(genome: ArchitectureSpec | Sequence[int]) -> ArchitectureSpec:
    if isinstance(genome, ArchitectureSpec):
        return genome
    raw_values = list(genome)
    canonical = canonicalize_genome(raw_values)
    values = [int(value) for value in raw_values]
    if values != canonical:
        raise ValueError(
            "Semantic V3 genome is not canonical; call canonicalize_genome() first"
        )
    unit_end = 1 + BRANCH_COUNT * UNITS_PER_BRANCH
    blocks = [state_to_block(value) for value in values[1:unit_end]]
    exchanges = [state_to_exchange(value) for value in values[unit_end:]]
    return architecture_from_blocks(CHANNELS[values[0]], blocks, exchanges)


def _exchange_barriers(states: Sequence[int]) -> tuple[bool, ...]:
    return tuple(int(state) != 0 for state in states)


@lru_cache(maxsize=4)
def canonical_branch_genomes(
    barriers: tuple[bool, ...],
) -> tuple[tuple[int, ...], ...]:
    """Enumerate branch states for one exact enabled-exchange barrier mask."""

    if len(barriers) != EXCHANGE_SITES:
        raise ValueError(f"IBASS requires {EXCHANGE_SITES} barrier decisions")
    if any(type(enabled) is not bool for enabled in barriers):
        raise TypeError("V3 barrier decisions must be booleans")
    exchanges = tuple(
        ExchangeGene.cimex(8) if enabled else ExchangeGene.none()
        for enabled in barriers
    )
    branches = {
        tuple(
            block_to_state(block)
            for block in canonicalize_branch(
                (state_to_block(state) for state in raw_states), exchanges
            )
        )
        for raw_states in product(range(UNIT_STATE_COUNT), repeat=UNITS_PER_BRANCH)
    }
    return tuple(sorted(branches))


def _sample_branch_multiset(
    rng: random.Random,
    catalog: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, ...], ...]:
    bars = sorted(rng.sample(range(len(catalog) + BRANCH_COUNT - 1), BRANCH_COUNT))
    indices = tuple(position - rank for rank, position in enumerate(bars))
    return tuple(catalog[index] for index in indices)


def _has_required_downstream_transform(
    branch_states: Sequence[Sequence[int]],
    barriers: Sequence[bool],
) -> bool:
    enabled_sites = [site for site, enabled in enumerate(barriers) if enabled]
    if not enabled_sites:
        return True
    first_required_stage = max(enabled_sites) + 1
    return any(
        any(int(state) != 0 for state in branch[first_required_stage:])
        for branch in branch_states
    )


def _valid_branch_multiset_count(barriers: tuple[bool, ...]) -> int:
    catalog = canonical_branch_genomes(barriers)
    total = comb(len(catalog) + BRANCH_COUNT - 1, BRANCH_COUNT)
    if not any(barriers):
        return total
    last_site = max(site for site, enabled in enumerate(barriers) if enabled)
    inactive = sum(
        all(state == 0 for state in branch[last_site + 1 :]) for branch in catalog
    )
    return total - comb(inactive + BRANCH_COUNT - 1, BRANCH_COUNT)


@lru_cache(maxsize=1)
def _exchange_configuration_count_items() -> tuple[tuple[tuple[int, ...], int], ...]:
    return tuple(
        (
            tuple(states),
            len(CHANNELS) * _valid_branch_multiset_count(_exchange_barriers(states)),
        )
        for states in product(range(len(EXCHANGE_CONFIGS)), repeat=EXCHANGE_SITES)
    )


def canonical_exchange_configuration_counts() -> dict[tuple[int, ...], int]:
    """Return exact complete-architecture counts for all nine exchange states."""

    return dict(_exchange_configuration_count_items())


def canonical_architecture_count() -> int:
    """Return the exact size of the corrected stage-aware V3 search space."""

    return sum(canonical_exchange_configuration_counts().values())


def _sample_exchange_states(
    rng: random.Random, exchange_probability: float | None
) -> tuple[int, ...]:
    if exchange_probability is not None:
        return tuple(
            0
            if rng.random() >= exchange_probability
            else rng.randrange(1, len(EXCHANGE_CONFIGS))
            for _ in range(EXCHANGE_SITES)
        )

    counts = canonical_exchange_configuration_counts()
    ticket = rng.randrange(sum(counts.values()))
    for states, count in counts.items():
        if ticket < count:
            return states
        ticket -= count
    raise AssertionError("V3 exchange-state sampling exhausted its exact count table")


def sample_canonical_genome(
    *,
    seed: int | None = None,
    exchange_probability: float | None = None,
) -> list[int]:
    """Sample V3 directly in canonical space.

    With ``exchange_probability=None`` (the scientific-search default), every
    complete canonical V3 architecture has exactly equal probability.  A
    numeric probability deliberately requests a hierarchical exchange prior.
    Branch multisets remain uniform within the selected stage-barrier mask.
    """

    if exchange_probability is not None and not 0.0 <= exchange_probability <= 1.0:
        raise ValueError("exchange_probability must lie in [0, 1] or be None")
    rng = random.Random(seed)
    states = _sample_exchange_states(rng, exchange_probability)
    barriers = _exchange_barriers(states)
    catalog = canonical_branch_genomes(barriers)
    while True:
        branch_states = _sample_branch_multiset(rng, catalog)
        if not _has_required_downstream_transform(branch_states, barriers):
            continue
        blocks = [state_to_block(state) for branch in branch_states for state in branch]
        exchanges = tuple(state_to_exchange(state) for state in states)
        return encode(architecture_from_blocks(rng.choice(CHANNELS), blocks, exchanges))


def sample_genome(
    *,
    seed: int | None = None,
    attention_probability: float = 0.5,
    skip_probability: float = 1.0 / UNIT_STATE_COUNT,
    exchange_probability: float = DEFAULT_EXCHANGE_PROBABILITY,
) -> list[int]:
    """Sample controlled family strata, then canonicalize.

    This is intended for validation experiments.  Scientific NAS
    initialization uses :func:`sample_canonical_genome`.
    """
    if not 0.0 <= attention_probability <= 1.0:
        raise ValueError("attention_probability must lie in [0, 1]")
    if not 0.0 <= skip_probability < 1.0:
        raise ValueError("skip_probability must lie in [0, 1)")
    if not 0.0 <= exchange_probability <= 1.0:
        raise ValueError("exchange_probability must lie in [0, 1]")

    rng = random.Random(seed)
    blocks: list[BlockGene] = []
    for _ in range(BRANCH_COUNT * UNITS_PER_BRANCH):
        if rng.random() < skip_probability:
            blocks.append(BlockGene.skip())
            continue
        if rng.random() < attention_probability:
            op, arg = rng.choice(ATTENTION_PRIMITIVE_CONFIGS)
            blocks.append(BlockGene("attention", op, arg, rng.choice(REPEATS)))
        else:
            op, arg = rng.choice(CNN_PRIMITIVE_CONFIGS)
            blocks.append(BlockGene("cnn", op, arg, rng.choice(REPEATS)))

    exchanges = []
    for _ in range(EXCHANGE_SITES):
        if rng.random() < exchange_probability:
            exchanges.append(ExchangeGene.cimex(rng.choice((8, 16))))
        else:
            exchanges.append(ExchangeGene.none())
    return encode(architecture_from_blocks(rng.choice(CHANNELS), blocks, exchanges))


def sample(
    *,
    seed: int | None = None,
    attention_probability: float = 0.5,
    skip_probability: float = 1.0 / UNIT_STATE_COUNT,
    exchange_probability: float = DEFAULT_EXCHANGE_PROBABILITY,
) -> ArchitectureSpec:
    return decode(
        sample_genome(
            seed=seed,
            attention_probability=attention_probability,
            skip_probability=skip_probability,
            exchange_probability=exchange_probability,
        )
    )


def migrate_v2(
    genome: V2ArchitectureSpec | Sequence[int],
) -> ArchitectureSpec:
    """Embed a canonical V2 architecture exactly with both exchanges disabled."""

    previous = decode_v2(genome)
    return canonicalize_architecture(
        previous.channels,
        previous.branches,
        (ExchangeGene.none(),) * EXCHANGE_SITES,
    )


def to_v2(spec: ArchitectureSpec) -> V2ArchitectureSpec:
    """Project only the exact V2 subspace; enabled exchange cannot be discarded."""

    architecture = decode(spec)
    if architecture.uses_cimex:
        raise ValueError("A V3 architecture with enabled CIMEX cannot project to V2")
    return canonicalize_v2(architecture.channels, architecture.branches)


upgrade_v2 = migrate_v2
