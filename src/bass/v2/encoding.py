"""Canonical semantic codec used by the BASS V2 scientific search."""

from __future__ import annotations

import numbers
import random
from collections.abc import Sequence
from functools import lru_cache
from itertools import product

from bass.v1.encoding import decode as decode_v1
from bass.v1.genotype import ArchitectureSpec as V1ArchitectureSpec

from .config import (
    ATTENTION_PRIMITIVE_CONFIGS,
    BRANCH_COUNT,
    CHANNELS,
    CNN_PRIMITIVE_CONFIGS,
    PRIMITIVE_CONFIGS,
    REPEATS,
    SEMANTIC_GENOME_LENGTH,
    UNIT_STATE_COUNT,
    UNITS_PER_BRANCH,
)
from .genotype import (
    ArchitectureSpec,
    BlockGene,
    architecture_from_blocks,
    canonicalize_branch,
)
from .legacy93 import (
    LegacyArchitectureSpec,
    decode_legacy_bits,
    encode_legacy_bits,
)


def state_to_block(state: int) -> BlockGene:
    if isinstance(state, bool) or not isinstance(state, numbers.Integral):
        raise TypeError(f"Unit state must be an integer in [0, {UNIT_STATE_COUNT - 1}]")
    value = int(state)
    if not 0 <= value < UNIT_STATE_COUNT:
        raise ValueError(
            f"Unit state must be an integer in [0, {UNIT_STATE_COUNT - 1}]"
        )
    if value == 0:
        return BlockGene.skip()
    primitive_index, repeat_index = divmod(value - 1, len(REPEATS))
    family, op, arg = PRIMITIVE_CONFIGS[primitive_index]
    return BlockGene(family, op, arg, REPEATS[repeat_index])


def block_to_state(block: BlockGene) -> int:
    if block.is_skip:
        return 0
    try:
        primitive_index = PRIMITIVE_CONFIGS.index(block.operation_key)
        repeat_index = REPEATS.index(block.repeat)
    except ValueError as error:
        raise ValueError(
            f"Block is outside the scientific V2 catalog: {block}"
        ) from error
    return 1 + primitive_index * len(REPEATS) + repeat_index


def encode(spec: ArchitectureSpec) -> list[int]:
    if not isinstance(spec, ArchitectureSpec):
        raise TypeError("encode requires a canonical bass.v2 ArchitectureSpec")
    return [CHANNELS.index(spec.channels)] + [
        block_to_state(block) for block in spec.flat_blocks
    ]


def canonicalize_genome(genome: Sequence[int]) -> list[int]:
    values = list(genome)
    if len(values) != SEMANTIC_GENOME_LENGTH:
        raise ValueError(
            f"A semantic V2 genome requires {SEMANTIC_GENOME_LENGTH} integers"
        )
    if isinstance(values[0], bool) or not isinstance(values[0], numbers.Integral):
        raise TypeError("V2 channel state must be an integer in [0, 3]")
    channel_id = int(values[0])
    if not 0 <= channel_id < len(CHANNELS):
        raise ValueError("V2 channel state must be an integer in [0, 3]")
    blocks = [state_to_block(value) for value in values[1:]]
    return encode(architecture_from_blocks(CHANNELS[channel_id], blocks))


def decode(genome: ArchitectureSpec | Sequence[int]) -> ArchitectureSpec:
    """Decode only the canonical 10-integer scientific representation."""

    if isinstance(genome, ArchitectureSpec):
        return genome
    raw_values = list(genome)
    canonical = canonicalize_genome(raw_values)
    values = [int(value) for value in raw_values]
    if values != canonical:
        raise ValueError(
            "Semantic V2 genome is not canonical; call canonicalize_genome() first"
        )
    blocks = [state_to_block(value) for value in values[1:]]
    return architecture_from_blocks(CHANNELS[values[0]], blocks)


@lru_cache(maxsize=1)
def canonical_branch_genomes() -> tuple[tuple[int, ...], ...]:
    """Enumerate the 68,923 distinct canonical three-unit branch states.

    The raw three-slot grid has many skip-placement and repeat-grouping
    preimages.  Building this catalog once lets scientific initialization
    sample canonical branches directly instead of inheriting that bias.
    """

    branches = {
        tuple(
            block_to_state(block)
            for block in canonicalize_branch(
                state_to_block(state) for state in raw_states
            )
        )
        for raw_states in product(range(UNIT_STATE_COUNT), repeat=UNITS_PER_BRANCH)
    }
    return tuple(sorted(branches))


def _sample_branch_multiset(rng: random.Random) -> tuple[tuple[int, ...], ...]:
    """Sample one unordered branch multiset exactly uniformly."""

    catalog = canonical_branch_genomes()
    # Stars-and-bars: every size-three multiset maps bijectively to one
    # size-three subset of range(B + 2).  Sampling the subset uniformly avoids
    # overweighting three-distinct-branch architectures by a factor of six.
    bars = sorted(rng.sample(range(len(catalog) + BRANCH_COUNT - 1), BRANCH_COUNT))
    indices = tuple(position - rank for rank, position in enumerate(bars))
    return tuple(catalog[index] for index in indices)


def sample_canonical_genome(*, seed: int | None = None) -> list[int]:
    """Sample uniformly from complete canonical V2 architectures.

    Channels and unordered branch multisets are uniform.  Use
    :func:`sample_genome` only when a deliberately family-conditioned prior is
    required for an audit or controlled construction.
    """

    rng = random.Random(seed)
    branch_states = _sample_branch_multiset(rng)
    blocks = [state_to_block(state) for branch in branch_states for state in branch]
    return encode(architecture_from_blocks(rng.choice(CHANNELS), blocks))


def sample_genome(
    *,
    seed: int | None = None,
    attention_probability: float = 0.5,
    skip_probability: float = 1.0 / UNIT_STATE_COUNT,
) -> list[int]:
    """Sample a family-conditioned raw grid, then canonicalize it.

    This helper is useful for CNN/attention strata, but it is intentionally not
    the default NAS initializer because its many-to-one canonicalization gives
    architectures unequal prior probability.
    """
    if not 0.0 <= attention_probability <= 1.0:
        raise ValueError("attention_probability must lie in [0, 1]")
    if not 0.0 <= skip_probability < 1.0:
        raise ValueError("skip_probability must lie in [0, 1)")
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
    spec = architecture_from_blocks(rng.choice(CHANNELS), blocks)
    return encode(spec)


def sample(
    *,
    seed: int | None = None,
    attention_probability: float = 0.5,
    skip_probability: float = 1.0 / UNIT_STATE_COUNT,
) -> ArchitectureSpec:
    return decode(
        sample_genome(
            seed=seed,
            attention_probability=attention_probability,
            skip_probability=skip_probability,
        )
    )


def _nearest_kernel(value: int) -> int:
    return 3 if int(value) <= 3 else 5


def _map_old_block(family: str, op: str, arg: int, repeat: int) -> BlockGene:
    repeat = min(max(int(repeat), min(REPEATS)), max(REPEATS))
    if family == "attention":
        window = 4 if int(arg) <= 4 else 8
        mapping = {
            "channel_attention": ("channel_attention_residual", 0),
            "window_attention": ("window_transformer", window),
            "shifted_window_attention": ("regular_shifted_pair", window),
            "hybrid_conv_attention": ("hybrid_conv_window", window),
        }
        new_op, new_arg = mapping[op]
        return BlockGene("attention", new_op, new_arg, repeat)

    if op == "identity":
        return BlockGene.skip()
    kernel = _nearest_kernel(arg)
    if op == "conv":
        return BlockGene("cnn", "res_conv", kernel, repeat)
    if op.startswith("dil_conv"):
        return BlockGene("cnn", "res_dilated_d2", 3, repeat)
    if op == "depthwise_separable_conv":
        return BlockGene("cnn", "res_depthwise_separable", kernel, repeat)
    if op == "inverted_bottleneck_e2":
        return BlockGene("cnn", "inverted_residual_e2", kernel, repeat)
    if op == "conv_transpose":
        return BlockGene("cnn", "res_conv", kernel, repeat)
    raise ValueError(f"Unsupported legacy operation: {op}")


def migrate_v1(genome: V1ArchitectureSpec | Sequence[int]) -> ArchitectureSpec:
    """Map V1 into the new catalog; this is explicit and not phenotype-exact."""

    legacy = decode_v1(genome)
    blocks = [
        _map_old_block("cnn", block.op, block.arg, block.repeat)
        for block in legacy.flat_blocks
    ]
    return architecture_from_blocks(legacy.channels, blocks)


def migrate_legacy93(
    genome: LegacyArchitectureSpec | Sequence[int],
) -> ArchitectureSpec:
    legacy = (
        genome
        if isinstance(genome, LegacyArchitectureSpec)
        else decode_legacy_bits(genome)
    )
    blocks = [
        _map_old_block(block.family, block.op, block.arg, block.repeat)
        for block in legacy.flat_blocks
    ]
    return architecture_from_blocks(legacy.channels, blocks)


# Compatibility names are deliberately explicit about the retired format.
decode_bits = decode_legacy_bits
decode_v2_bits = decode_legacy_bits
encode_bits = encode_legacy_bits
encode_v2_bits = encode_legacy_bits
sample_v2 = sample
upgrade_v1 = migrate_v1
