"""Versioned BASS search spaces.

Use :mod:`bass.v1` for the frozen CNN baseline and :mod:`bass.v2` for the
optional hybrid attention search space. Historical top-level exports are loaded
lazily for compatibility.
"""

from __future__ import annotations

from importlib import import_module

__version__ = "0.3.0"
__all__ = ["v1", "v2"]

_V2_EXPORTS = {
    "ATTENTION_PRIMITIVES",
    "CHANNELS",
    "CNN_PRIMITIVES",
    "KERNEL_SIZES",
    "REPEATS",
    "WINDOW_SIZES",
    "ArchitectureSpec",
    "BlockGene",
    "canonicalize_genome",
    "encode",
    "encode_v2_bits",
    "migrate_legacy93",
    "migrate_v1",
    "repair_architecture",
    "sample_v2",
    "sample_genome",
    "upgrade_v1",
}


def __getattr__(name: str):
    if name in {"v1", "v2"}:
        return import_module(f"{__name__}.{name}")
    if name == "decode":
        return import_module(f"{__name__}.encoding").decode
    if name in {"decode_v1_bits", "decode_v1_gene", "decode_v2_bits"}:
        return getattr(import_module(f"{__name__}.encoding"), name)
    if name in _V2_EXPORTS:
        return getattr(import_module(f"{__name__}.v2"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
