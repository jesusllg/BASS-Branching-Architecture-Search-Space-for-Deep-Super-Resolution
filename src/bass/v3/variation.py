"""Semantic variation operators for interaction-aware BASS V3."""

from __future__ import annotations

from typing import Any

from bass.v2.variation import SEMANTIC_MUTATION_WEIGHTS, mutate_block

from .genotype import ExchangeGene

EXCHANGE_MUTATION_WEIGHTS = {
    "prototype": 0.60,
    "exchange_delete": 0.40,
}


def mutate_exchange(exchange: ExchangeGene, rng: Any) -> tuple[ExchangeGene, str]:
    """Insert, delete, or locally resize one complete CIMEX state."""

    if not isinstance(exchange, ExchangeGene):
        raise TypeError("mutate_exchange requires a bass.v3 ExchangeGene")
    if not exchange.is_enabled:
        prototypes = (8, 16)[int(rng.integers(0, 2))]
        return ExchangeGene.cimex(prototypes), "exchange_insert"

    move = (
        "prototype"
        if rng.random()
        < EXCHANGE_MUTATION_WEIGHTS["prototype"]
        / sum(EXCHANGE_MUTATION_WEIGHTS.values())
        else "exchange_delete"
    )
    if move == "prototype":
        prototypes = 16 if exchange.prototypes == 8 else 8
        return ExchangeGene.cimex(prototypes), move
    return ExchangeGene.none(), move


__all__ = [
    "EXCHANGE_MUTATION_WEIGHTS",
    "SEMANTIC_MUTATION_WEIGHTS",
    "mutate_block",
    "mutate_exchange",
]
