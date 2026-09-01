"""Canonical 12-integer codec for interaction-aware BASS V3."""

from __future__ import annotations

import numbers
import random
from collections.abc import Sequence

from bass.v2.encoding import (
    block_to_state,
    state_to_block,
)
from bass.v2.encoding import (
    decode as decode_v2,
)
from bass.v2.encoding import (
    sample_canonical_genome as sample_canonical_v2_genome,
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


def _valid_exchange_states(
    branches: tuple[tuple[BlockGene, ...], ...],
) -> tuple[tuple[int, ...], ...]:
    after_stage_1 = (
        (0,)
        if all(branch[1].is_skip and branch[2].is_skip for branch in branches)
        else tuple(range(len(EXCHANGE_CONFIGS)))
    )
    after_stage_2 = (
        (0,)
        if all(branch[2].is_skip for branch in branches)
        else tuple(range(len(EXCHANGE_CONFIGS)))
    )
    return after_stage_1, after_stage_2


def sample_canonical_genome(
    *,
    seed: int | None = None,
    exchange_probability: float | None = None,
) -> list[int]:
    """Sample V3 directly in canonical space.

    With ``exchange_probability=None`` (the scientific-search default), every
    complete canonical V3 architecture has exactly equal probability.  A
    numeric probability deliberately requests a family-conditioned prior while
    retaining uniform canonical V2 branch sampling.
    """

    if exchange_probability is not None and not 0.0 <= exchange_probability <= 1.0:
        raise ValueError("exchange_probability must lie in [0, 1] or be None")
    rng = random.Random(seed)
    while True:
        previous = decode_v2(sample_canonical_v2_genome(seed=rng.randrange(0, 2**63)))
        options = _valid_exchange_states(previous.branches)
        valid_count = len(options[0]) * len(options[1])

        if exchange_probability is None:
            # Rejection weighting compensates for base architectures that have
            # only one or three algebraically active exchange combinations.
            # The maximum is 3 * 3 = 9.
            if rng.randrange(len(EXCHANGE_CONFIGS) ** EXCHANGE_SITES) >= valid_count:
                continue
            states = [rng.choice(site_options) for site_options in options]
        else:
            states = [
                (
                    0
                    if len(site_options) == 1 or rng.random() >= exchange_probability
                    else rng.choice(site_options[1:])
                )
                for site_options in options
            ]
        exchanges = tuple(state_to_exchange(state) for state in states)
        return encode(
            canonicalize_architecture(previous.channels, previous.branches, exchanges)
        )


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
