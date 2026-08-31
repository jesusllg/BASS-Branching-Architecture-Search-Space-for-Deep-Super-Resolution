"""Compatibility model-builder dispatcher for BASS V1 and V2."""

from __future__ import annotations

from .v1.genotype import ArchitectureSpec as V1ArchitectureSpec
from .v2.genotype import ArchitectureSpec as V2ArchitectureSpec


def _version_of(architecture) -> int:
    if isinstance(architecture, V1ArchitectureSpec):
        return 1
    if isinstance(architecture, V2ArchitectureSpec):
        return 2
    try:
        length = len(architecture)
    except TypeError:
        length = None
    if length is not None:
        if length in {28, 84}:
            return 1
        if length in {10, 93}:
            return 2
    raise ValueError("Cannot determine whether the architecture is BASS V1 or V2")


def build_model(architecture, **kwargs):
    if _version_of(architecture) == 1:
        from .v1.model_builder import build_model as build_v1

        return build_v1(architecture, **kwargs)
    from .v2.model_builder import build_model as build_v2

    if not isinstance(architecture, V2ArchitectureSpec) and len(architecture) == 93:
        from .v2.encoding import migrate_legacy93

        architecture = migrate_legacy93(architecture)

    return build_v2(architecture, **kwargs)


get_model = build_model


def __getattr__(name: str):
    if name == "PixelShuffle":
        from .v2.model_builder import PixelShuffle

        return PixelShuffle
    raise AttributeError(name)
