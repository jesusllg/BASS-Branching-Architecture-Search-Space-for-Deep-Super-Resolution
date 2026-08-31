"""Residual CNN units used for fairer CNN/attention competition in V2."""

from __future__ import annotations

import tensorflow as tf

from ..config import RESIDUAL_SCALE

keras = tf.keras
layers = keras.layers


class _ResidualLayer(layers.Layer):
    def __init__(self, channels: int, residual_scale: float = RESIDUAL_SCALE, **kwargs):
        super().__init__(**kwargs)
        self.channels = int(channels)
        self.residual_scale = float(residual_scale)
        if self.channels <= 0:
            raise ValueError("channels must be positive")
        if self.residual_scale <= 0:
            raise ValueError("residual_scale must be positive")

    def _base_config(self) -> dict:
        return {
            "channels": self.channels,
            "residual_scale": self.residual_scale,
        }


@keras.utils.register_keras_serializable(package="bass.v2")
class ResidualConvBlock(_ResidualLayer):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation_rate: int = 1,
        residual_scale: float = RESIDUAL_SCALE,
        **kwargs,
    ):
        super().__init__(channels, residual_scale, **kwargs)
        self.kernel_size = int(kernel_size)
        self.dilation_rate = int(dilation_rate)
        self.conv = layers.Conv2D(
            self.channels,
            self.kernel_size,
            dilation_rate=self.dilation_rate,
            padding="same",
            activation="gelu",
            name="transform",
        )

    def build(self, input_shape) -> None:
        self.conv.build(input_shape)
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return inputs + self.residual_scale * self.conv(inputs)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(self._base_config())
        config.update(
            {
                "kernel_size": self.kernel_size,
                "dilation_rate": self.dilation_rate,
            }
        )
        return config


@keras.utils.register_keras_serializable(package="bass.v2")
class ResidualDepthwiseSeparableBlock(_ResidualLayer):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        residual_scale: float = RESIDUAL_SCALE,
        **kwargs,
    ):
        super().__init__(channels, residual_scale, **kwargs)
        self.kernel_size = int(kernel_size)
        self.depthwise = layers.DepthwiseConv2D(
            self.kernel_size,
            padding="same",
            activation="gelu",
            name="depthwise",
        )
        self.project = layers.Conv2D(self.channels, 1, padding="same", name="project")

    def build(self, input_shape) -> None:
        self.depthwise.build(input_shape)
        self.project.build(input_shape)
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        delta = self.project(self.depthwise(inputs))
        return inputs + self.residual_scale * delta

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(self._base_config())
        config.update({"kernel_size": self.kernel_size})
        return config


@keras.utils.register_keras_serializable(package="bass.v2")
class InvertedResidualBlock(_ResidualLayer):
    """Expansion-2 inverted residual with a linear projection."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        expansion: int = 2,
        residual_scale: float = RESIDUAL_SCALE,
        **kwargs,
    ):
        super().__init__(channels, residual_scale, **kwargs)
        self.kernel_size = int(kernel_size)
        self.expansion = int(expansion)
        if self.expansion < 1:
            raise ValueError("expansion must be positive")
        expanded = self.channels * self.expansion
        self.expand = layers.Conv2D(
            expanded, 1, padding="same", activation="gelu", name="expand"
        )
        self.depthwise = layers.DepthwiseConv2D(
            self.kernel_size,
            padding="same",
            activation="gelu",
            name="depthwise",
        )
        self.project = layers.Conv2D(
            self.channels, 1, padding="same", activation=None, name="linear_project"
        )

    def build(self, input_shape) -> None:
        self.expand.build(input_shape)
        expanded_shape = tuple(input_shape[:-1]) + (self.channels * self.expansion,)
        self.depthwise.build(expanded_shape)
        self.project.build(expanded_shape)
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        delta = self.project(self.depthwise(self.expand(inputs)))
        return inputs + self.residual_scale * delta

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(self._base_config())
        config.update({"kernel_size": self.kernel_size, "expansion": self.expansion})
        return config
