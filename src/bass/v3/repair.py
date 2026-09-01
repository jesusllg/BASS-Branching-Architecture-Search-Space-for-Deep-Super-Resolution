"""Strict canonicalization helpers for BASS V3."""

from __future__ import annotations

from .genotype import ArchitectureSpec, ExchangeGene, canonicalize_architecture


def repair_exchange(exchange: ExchangeGene) -> ExchangeGene:
    if not isinstance(exchange, ExchangeGene):
        raise TypeError("repair_exchange requires a bass.v3 ExchangeGene")
    return exchange


def repair_architecture(spec: ArchitectureSpec) -> ArchitectureSpec:
    if not isinstance(spec, ArchitectureSpec):
        raise TypeError("repair_architecture accepts only a bass.v3 ArchitectureSpec")
    return canonicalize_architecture(spec.channels, spec.branches, spec.exchanges)
