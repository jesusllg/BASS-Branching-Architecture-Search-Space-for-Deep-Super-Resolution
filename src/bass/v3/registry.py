"""V3 registry: frozen V2 unit catalog plus CIMEX exchange."""

from __future__ import annotations

from bass.v2.registry import make_unit_layers

from .blocks import CIMEXLayer
from .genotype import ExchangeGene


def make_exchange_layer(
    exchange: ExchangeGene,
    channels: int,
    name: str,
) -> CIMEXLayer | None:
    if not isinstance(exchange, ExchangeGene):
        raise TypeError("make_exchange_layer requires a V3 ExchangeGene")
    if not exchange.is_enabled:
        return None
    return CIMEXLayer(
        channels=channels,
        prototypes=exchange.prototypes,
        name=name,
    )


__all__ = ["make_exchange_layer", "make_unit_layers"]
