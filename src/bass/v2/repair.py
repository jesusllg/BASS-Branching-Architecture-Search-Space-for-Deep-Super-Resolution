"""Explicit canonicalization helpers; invalid research artifacts fail strictly."""

from __future__ import annotations

from .genotype import ArchitectureSpec, BlockGene, canonicalize_architecture


def repair_block(block: BlockGene) -> BlockGene:
    """Validate a block without silently substituting another operation."""

    if not isinstance(block, BlockGene):
        raise TypeError("repair_block requires a bass.v2 BlockGene")
    return block


def repair_architecture(spec: ArchitectureSpec) -> ArchitectureSpec:
    """Canonicalize ordering only; never coerce invalid values to nearest ones."""

    if not isinstance(spec, ArchitectureSpec):
        raise TypeError("repair_architecture accepts only a bass.v2 ArchitectureSpec")
    return canonicalize_architecture(spec.channels, spec.branches)
