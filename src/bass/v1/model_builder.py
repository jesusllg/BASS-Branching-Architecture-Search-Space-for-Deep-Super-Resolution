"""TensorFlow/Keras builder for the independent CNN-only BASS V1."""

from __future__ import annotations

from collections.abc import Sequence

import tensorflow as tf

from .config import DEFAULT_INPUT_CHANNELS, DEFAULT_UPSCALE
from .encoding import decode
from .genotype import ArchitectureSpec
from .registry import make_unit_layers

keras = tf.keras
layers = keras.layers


@keras.utils.register_keras_serializable(package="bass.v1")
class PixelShuffle(layers.Layer):
    def __init__(self, scale: int, **kwargs):
        super().__init__(**kwargs)
        self.scale = int(scale)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return tf.nn.depth_to_space(inputs, self.scale)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"scale": self.scale})
        return config


def build_model(
    architecture: ArchitectureSpec | Sequence[int],
    *,
    upscale_factor: int = DEFAULT_UPSCALE,
    input_channels: int = DEFAULT_INPUT_CHANNELS,
    input_shape: tuple[int | None, int | None, int] | None = None,
    return_feature_model: bool = False,
):
    """Build the original three-branch CNN-only BASS network."""

    spec = decode(architecture)
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

    inputs = layers.Input(shape=input_shape, dtype="float32", name="lr")
    stem = layers.Conv2D(
        spec.channels, 3, padding="same", activation="relu", name="stem"
    )(inputs)

    branch_outputs = []
    feature_tensors = []
    feature_names = []
    for branch_index, branch in enumerate(spec.branches, start=1):
        x = stem
        for unit_index, block in enumerate(branch, start=1):
            unit_name = f"branch{branch_index}_unit{unit_index}_{block.op}"
            for layer in make_unit_layers(block, spec.channels, unit_name):
                x = layer(x)
            feature_tensors.append(x)
            feature_names.append(f"branch{branch_index}.unit{unit_index}")
        branch_outputs.append(x)

    merged = layers.Add(name="branch_add")(branch_outputs)
    reconstruction_channels = input_channels * (upscale_factor**2)
    x = layers.Conv2D(
        reconstruction_channels,
        3,
        padding="same",
        activation="relu",
        name="reconstruction",
    )(merged)
    x = PixelShuffle(upscale_factor, name=f"pixel_shuffle_x{upscale_factor}")(x)
    outputs = layers.Conv2D(
        input_channels,
        3,
        padding="same",
        activation="sigmoid",
        dtype="float32",
        name="sr",
    )(x)

    model = keras.Model(inputs, outputs, name=f"bass_v1_x{upscale_factor}")
    if not return_feature_model:
        return model
    feature_model = keras.Model(inputs, feature_tensors, name="bass_v1_feature_taps")
    return model, feature_model, tuple(feature_names)


get_model = build_model
