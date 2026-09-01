"""Semantically local variation operators shared by BASS V2 and V3."""

from __future__ import annotations

from typing import Any

from .config import PRIMITIVE_CONFIGS, REPEATS
from .genotype import BlockGene

# Explicit probabilities make the optimizer contract auditable.  Unavailable
# moves (for example, a kernel change on channel attention) are removed and the
# remaining weights are renormalized.
SEMANTIC_MUTATION_WEIGHTS = {
    "repeat": 0.30,
    "argument": 0.20,
    "operation": 0.20,
    "family_flip": 0.15,
    "delete": 0.15,
}


def _pick(values: list[Any], rng: Any) -> Any:
    return values[int(rng.integers(0, len(values)))]


def mutate_block(block: BlockGene, rng: Any) -> tuple[BlockGene, str]:
    """Apply one complete, logged semantic move to a unit state."""

    if not isinstance(block, BlockGene):
        raise TypeError("mutate_block requires a bass.v2 BlockGene")

    if block.is_skip:
        family, op, arg = _pick(list(PRIMITIVE_CONFIGS), rng)
        return BlockGene(family, op, arg, _pick(list(REPEATS), rng)), "insert"

    candidates: dict[str, list[BlockGene]] = {
        "repeat": [
            BlockGene(block.family, block.op, block.arg, repeat)
            for repeat in REPEATS
            if repeat != block.repeat
        ],
        "argument": [
            BlockGene(family, op, arg, block.repeat)
            for family, op, arg in PRIMITIVE_CONFIGS
            if family == block.family and op == block.op and arg != block.arg
        ],
        "operation": [
            BlockGene(family, op, arg, block.repeat)
            for family, op, arg in PRIMITIVE_CONFIGS
            if family == block.family and op != block.op
        ],
        "family_flip": [
            BlockGene(family, op, arg, block.repeat)
            for family, op, arg in PRIMITIVE_CONFIGS
            if family != block.family
        ],
        "delete": [BlockGene.skip()],
    }
    moves = [name for name, options in candidates.items() if options]
    weights = [SEMANTIC_MUTATION_WEIGHTS[name] for name in moves]
    probabilities = [weight / sum(weights) for weight in weights]
    move = moves[int(rng.choice(len(moves), p=probabilities))]
    return _pick(candidates[move], rng), move
