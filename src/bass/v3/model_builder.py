"""TensorFlow/Keras builder for interaction-aware BASS V3 (IBASS)."""

from __future__ import annotations

from collections.abc import Sequence

import tensorflow as tf

from bass.v2.model_builder import BicubicUpsample, PixelShuffle
from bass.v2.model_builder import build_model as build_v2_model

from .config import (
    BRANCH_COUNT,
    DEFAULT_HEAD_MODE,
    DEFAULT_INPUT_CHANNELS,
    DEFAULT_UPSCALE,
    UNITS_PER_BRANCH,
)
from .encoding import decode, to_v2
from .genotype import ArchitectureSpec
from .registry import make_exchange_layer, make_unit_layers

keras = tf.keras
layers = keras.layers


def feature_tap_metadata(
    architecture: ArchitectureSpec | Sequence[int],
) -> tuple[dict[str, int | str | bool | None], ...]:
    """Describe unit taps, effective depth, and following exchange state."""

    spec = decode(architecture)
    records = []
    for branch_index, branch in enumerate(spec.branches, start=1):
        cumulative_repeat_depth = 0
        for unit_index, block in enumerate(branch, start=1):
            cumulative_repeat_depth += block.repeat
            attention_blocks = 0
            if block.family == "attention":
                attention_blocks = block.repeat * (
                    2 if block.op == "regular_shifted_pair" else 1
                )
            exchange_after = None
            if unit_index <= len(spec.exchanges):
                exchange = spec.exchanges[unit_index - 1]
                exchange_after = (
                    "none"
                    if not exchange.is_enabled
                    else f"cimex_k{exchange.prototypes}"
                )
            records.append(
                {
                    "name": f"branch{branch_index}.unit{unit_index}",
                    "branch": branch_index,
                    "unit": unit_index,
                    "family": block.family,
                    "operation": block.op,
                    "argument": block.arg,
                    "repeat": block.repeat,
                    "cumulative_repeat_depth": cumulative_repeat_depth,
                    "internal_attention_blocks": attention_blocks,
                    "exchange_after": exchange_after,
                    "tap_is_pre_exchange": exchange_after is not None,
                }
            )
    return tuple(records)


def build_model(
    architecture: ArchitectureSpec | Sequence[int],
    *,
    upscale_factor: int = DEFAULT_UPSCALE,
    input_channels: int = DEFAULT_INPUT_CHANNELS,
    input_shape: tuple[int | None, int | None, int] | None = None,
    return_feature_model: bool = False,
    head_mode: str = DEFAULT_HEAD_MODE,
):
    """Build IBASS, delegating the exact no-exchange subspace to BASS V2."""

    spec = decode(architecture)
    if not spec.uses_cimex:
        return build_v2_model(
            to_v2(spec),
            upscale_factor=upscale_factor,
            input_channels=input_channels,
            input_shape=input_shape,
            return_feature_model=return_feature_model,
            head_mode=head_mode,
        )
    if upscale_factor not in {2, 3, 4}:
        raise ValueError("upscale_factor must be one of 2, 3, or 4")
    if input_channels <= 0:
        raise ValueError("input_channels must be positive")
    if input_shape is None:
        input_shape = (None, None, input_channels)
    if len(input_shape) != 3:
        raise ValueError("input_shape must contain height, width, and channels")
    if input_shape[-1] != input_channels:
        raise ValueError("input_shape channels must match input_channels")
    if head_mode not in {"residual", "direct"}:
        raise ValueError("head_mode must be 'residual' or 'direct'")

    inputs = layers.Input(shape=input_shape, dtype="float32", name="lr")
    stem = layers.Conv2D(
        spec.channels,
        3,
        padding="same",
        activation="gelu",
        name="stem",
    )(inputs)

    branch_states = [stem for _ in spec.branches]
    feature_tensors = [[None] * UNITS_PER_BRANCH for _ in spec.branches]
    for unit_index in range(UNITS_PER_BRANCH):
        next_states = []
        for branch_index, branch in enumerate(spec.branches, start=1):
            block = branch[unit_index]
            x = branch_states[branch_index - 1]
            unit_name = f"branch{branch_index}_unit{unit_index + 1}_{block.op}"
            for layer in make_unit_layers(block, spec.channels, unit_name):
                x = layer(x)
            next_states.append(x)
            feature_tensors[branch_index - 1][unit_index] = x
        branch_states = next_states

        if unit_index < len(spec.exchanges):
            exchange = spec.exchanges[unit_index]
            exchange_layer = make_exchange_layer(
                exchange,
                spec.channels,
                name=(
                    f"exchange{unit_index + 1}_cimex_k{exchange.prototypes}"
                    if exchange.is_enabled
                    else f"exchange{unit_index + 1}_none"
                ),
            )
            if exchange_layer is not None:
                branch_states = list(exchange_layer(branch_states))

    merged = layers.Add(name="branch_add")(branch_states)
    reconstruction_channels = input_channels * (upscale_factor**2)
    x = layers.Conv2D(
        reconstruction_channels,
        3,
        padding="same",
        activation=None if head_mode == "residual" else "relu",
        name="reconstruction",
    )(merged)
    x = PixelShuffle(upscale_factor, name=f"pixel_shuffle_x{upscale_factor}")(x)

    if head_mode == "residual":
        residual = layers.Conv2D(
            input_channels,
            3,
            padding="same",
            activation=None,
            dtype="float32",
            name="residual_rgb",
        )(x)
        baseline = BicubicUpsample(upscale_factor, name=f"bicubic_x{upscale_factor}")(
            inputs
        )
        outputs = layers.Add(name="sr")([baseline, residual])
    else:
        outputs = layers.Conv2D(
            input_channels,
            3,
            padding="same",
            activation="sigmoid",
            dtype="float32",
            name="sr",
        )(x)

    model = keras.Model(inputs, outputs, name=f"ibass_v3_x{upscale_factor}")
    if not return_feature_model:
        return model
    flat_features = [tensor for branch in feature_tensors for tensor in branch]
    feature_names = tuple(
        f"branch{branch_index}.unit{unit_index}"
        for branch_index in range(1, BRANCH_COUNT + 1)
        for unit_index in range(1, UNITS_PER_BRANCH + 1)
    )
    feature_model = keras.Model(inputs, flat_features, name="ibass_v3_feature_taps")
    return model, feature_model, feature_names


build_ibass_model = build_model
get_model = build_model
