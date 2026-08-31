"""Compatibility exports for attention blocks now owned by BASS V2."""

from bass.v2.blocks import (
    ChannelAttentionBlock,
    HybridConvAttentionBlock,
    WindowAttentionBlock,
)

__all__ = [
    "ChannelAttentionBlock",
    "HybridConvAttentionBlock",
    "WindowAttentionBlock",
]
