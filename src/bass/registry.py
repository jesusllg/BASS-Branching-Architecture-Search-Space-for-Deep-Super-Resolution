"""Operator registry mapping canonical genes to Keras layers."""

from __future__ import annotations

import tensorflow as tf

from .blocks import (
    ChannelAttentionBlock,
    HybridConvAttentionBlock,
    WindowAttentionBlock,
)
from .config import HEADS_BY_CHANNELS
from .genotype import BlockGene

layers = tf.keras.layers


def _cnn_layer(block: BlockGene, channels: int, name: str) -> layers.Layer:
    conv_args = {"padding": "same", "activation": "relu"}
    kernel = block.arg
    if block.op == "conv":
        return layers.Conv2D(channels, kernel, name=name, **conv_args)
    if block.op.startswith("dil_conv_d"):
        dilation = int(block.op.rsplit("d", 1)[-1])
        return layers.Conv2D(
            channels,
            kernel,
            dilation_rate=dilation,
            name=name,
            **conv_args,
        )
    if block.op == "depthwise_separable_conv":
        return tf.keras.Sequential(
            [
                layers.DepthwiseConv2D(kernel, name=f"{name}_depthwise", **conv_args),
                layers.Conv2D(channels, 1, name=f"{name}_pointwise", **conv_args),
            ],
            name=name,
        )
    if block.op == "inverted_bottleneck_e2":
        return tf.keras.Sequential(
            [
                layers.Conv2D(channels * 2, 1, name=f"{name}_expand", **conv_args),
                layers.DepthwiseConv2D(kernel, name=f"{name}_depthwise", **conv_args),
                layers.Conv2D(channels, 1, name=f"{name}_project", **conv_args),
            ],
            name=name,
        )
    if block.op == "conv_transpose":
        return layers.Conv2DTranspose(channels, kernel, name=name, **conv_args)
    if block.op == "identity":
        return layers.Identity(name=name)
    raise ValueError(f"Unknown CNN primitive: {block.op}")


def make_unit_layers(block: BlockGene, channels: int, name: str) -> list[layers.Layer]:
    """Instantiate each repeat of one canonical unit."""

    output = []
    heads = HEADS_BY_CHANNELS[channels]
    for repeat_index in range(block.repeat):
        repeat_name = f"{name}_r{repeat_index + 1}"
        if block.family == "cnn":
            output.append(_cnn_layer(block, channels, repeat_name))
        elif block.op == "channel_attention":
            output.append(
                ChannelAttentionBlock(channels, reduction=4, name=repeat_name)
            )
        elif block.op in {"window_attention", "shifted_window_attention"}:
            output.append(
                WindowAttentionBlock(
                    channels=channels,
                    window_size=block.arg,
                    num_heads=heads,
                    shifted=block.op == "shifted_window_attention",
                    mlp_ratio=2,
                    name=repeat_name,
                )
            )
        elif block.op == "hybrid_conv_attention":
            output.append(
                HybridConvAttentionBlock(
                    channels=channels,
                    window_size=block.arg,
                    num_heads=heads,
                    shifted=bool(repeat_index % 2),
                    name=repeat_name,
                )
            )
        else:
            raise ValueError(f"Unknown attention primitive: {block.op}")
    return output
