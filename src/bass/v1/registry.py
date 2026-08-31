"""CNN-only operation registry for BASS V1."""

from __future__ import annotations

import tensorflow as tf

from .genotype import BlockGene

layers = tf.keras.layers


def _make_layer(block: BlockGene, channels: int, name: str) -> layers.Layer:
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
    raise ValueError(f"Unknown V1 primitive: {block.op}")


def make_unit_layers(block: BlockGene, channels: int, name: str) -> list[layers.Layer]:
    return [
        _make_layer(block, channels, f"{name}_r{repeat_index + 1}")
        for repeat_index in range(block.repeat)
    ]
