"""Strict operation registry for the cost-rebalanced BASS V2 catalog."""

from __future__ import annotations

import tensorflow as tf

from .blocks import (
    ChannelAttentionBlock,
    HybridConvAttentionBlock,
    InvertedResidualBlock,
    RegularShiftedWindowPair,
    ResidualConvBlock,
    ResidualDepthwiseSeparableBlock,
    WindowAttentionBlock,
)
from .config import HEADS_BY_CHANNELS
from .genotype import BlockGene

layers = tf.keras.layers


def _make_layer(block: BlockGene, channels: int, name: str) -> layers.Layer:
    if block.op == "res_conv":
        return ResidualConvBlock(channels, block.arg, name=name)
    if block.op == "res_dilated_d2":
        return ResidualConvBlock(channels, block.arg, dilation_rate=2, name=name)
    if block.op == "res_depthwise_separable":
        return ResidualDepthwiseSeparableBlock(channels, block.arg, name=name)
    if block.op == "inverted_residual_e2":
        return InvertedResidualBlock(channels, block.arg, expansion=2, name=name)
    if block.op == "channel_attention_residual":
        return ChannelAttentionBlock(channels, reduction=4, name=name)

    heads = HEADS_BY_CHANNELS[channels]
    if block.op == "window_transformer":
        return WindowAttentionBlock(
            channels=channels,
            window_size=block.arg,
            num_heads=heads,
            shifted=False,
            mlp_ratio=2,
            name=name,
        )
    if block.op == "regular_shifted_pair":
        return RegularShiftedWindowPair(
            channels=channels,
            window_size=block.arg,
            num_heads=heads,
            mlp_ratio=2,
            name=name,
        )
    if block.op == "hybrid_conv_window":
        return HybridConvAttentionBlock(
            channels=channels,
            window_size=block.arg,
            num_heads=heads,
            shifted=False,
            name=name,
        )
    raise ValueError(f"Unknown V2 primitive: {block.op}")


def make_unit_layers(block: BlockGene, channels: int, name: str) -> list[layers.Layer]:
    """Instantiate complete residual units; skip has no trainable layer."""

    if block.is_skip:
        return [layers.Identity(name=f"{name}_skip")]
    return [
        _make_layer(block, channels, f"{name}_r{repeat_index + 1}")
        for repeat_index in range(block.repeat)
    ]
