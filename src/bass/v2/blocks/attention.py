"""Spatially shape-preserving attention blocks for super-resolution."""

from __future__ import annotations

import tensorflow as tf

from ..config import RESIDUAL_SCALE

keras = tf.keras
layers = keras.layers


def _window_partition(inputs: tf.Tensor, window_size: int) -> tf.Tensor:
    """Convert BHWC features into a batch of flattened local windows."""

    shape = tf.shape(inputs)
    batch, height, width, channels = shape[0], shape[1], shape[2], shape[3]
    x = tf.reshape(
        inputs,
        [
            batch,
            height // window_size,
            window_size,
            width // window_size,
            window_size,
            channels,
        ],
    )
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    return tf.reshape(x, [-1, window_size * window_size, channels])


def _window_reverse(
    windows: tf.Tensor, height: tf.Tensor, width: tf.Tensor, window_size: int
) -> tf.Tensor:
    """Restore flattened windows to a BHWC feature map."""

    shape = tf.shape(windows)
    channels = shape[-1]
    windows_per_image = (height // window_size) * (width // window_size)
    batch = shape[0] // windows_per_image
    x = tf.reshape(
        windows,
        [
            batch,
            height // window_size,
            width // window_size,
            window_size,
            window_size,
            channels,
        ],
    )
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    return tf.reshape(x, [batch, height, width, channels])


@keras.utils.register_keras_serializable(package="bass.v2")
class ChannelAttentionBlock(layers.Layer):
    """Signed residual transform with squeeze-and-excitation gating."""

    def __init__(
        self,
        channels: int,
        reduction: int = 4,
        residual_scale: float = RESIDUAL_SCALE,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.channels = int(channels)
        self.reduction = int(reduction)
        self.residual_scale = float(residual_scale)
        if self.channels <= 0:
            raise ValueError("channels must be positive")
        if self.reduction <= 0:
            raise ValueError("reduction must be positive")
        if self.residual_scale <= 0:
            raise ValueError("residual_scale must be positive")
        hidden = max(1, self.channels // self.reduction)
        self.transform = layers.Conv2D(
            self.channels, 1, activation="gelu", name="transform"
        )
        self.reduce = layers.Conv2D(hidden, 1, activation="gelu", name="gate_reduce")
        self.expand = layers.Conv2D(
            self.channels, 1, activation="sigmoid", name="expand"
        )

    def build(self, input_shape) -> None:
        self.transform.build(input_shape)
        context_shape = (input_shape[0], 1, 1, input_shape[-1])
        self.reduce.build(context_shape)
        reduced_shape = (input_shape[0], 1, 1, self.reduce.filters)
        self.expand.build(reduced_shape)
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        delta = self.transform(inputs)
        context = tf.reduce_mean(delta, axis=[1, 2], keepdims=True)
        gate = self.expand(self.reduce(context))
        return inputs + self.residual_scale * delta * gate

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "channels": self.channels,
                "reduction": self.reduction,
                "residual_scale": self.residual_scale,
            }
        )
        return config


@keras.utils.register_keras_serializable(package="bass.v2")
class WindowAttentionBlock(layers.Layer):
    """Pre-norm local or shifted-window self-attention with an FFN.

    Inputs are padded internally, so inference is not restricted to dimensions
    divisible by the selected window. Shifted windows use a boolean region mask
    to prevent cyclic wrap-around from creating false image neighbours.
    """

    def __init__(
        self,
        channels: int,
        window_size: int,
        num_heads: int,
        shifted: bool = False,
        mlp_ratio: int = 2,
        residual_scale: float = RESIDUAL_SCALE,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.channels = int(channels)
        self.window_size = int(window_size)
        self.num_heads = int(num_heads)
        self.shifted = bool(shifted)
        self.mlp_ratio = int(mlp_ratio)
        self.residual_scale = float(residual_scale)

        if self.channels <= 0:
            raise ValueError("channels must be positive")
        if self.window_size <= 0:
            raise ValueError("window_size must be positive")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.channels % self.num_heads:
            raise ValueError("channels must be divisible by num_heads")
        if self.mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive")
        if self.residual_scale <= 0:
            raise ValueError("residual_scale must be positive")

        self.shift_size = self.window_size // 2 if self.shifted else 0
        self.position = layers.DepthwiseConv2D(
            3, padding="same", name="conv_positional_encoding"
        )
        self.norm1 = layers.LayerNormalization(epsilon=1e-5, name="norm1")
        self.attention = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.channels // self.num_heads,
            output_shape=self.channels,
            dropout=0.0,
            name="mha",
        )
        self.norm2 = layers.LayerNormalization(epsilon=1e-5, name="norm2")
        self.ffn_expand = layers.Dense(
            self.channels * self.mlp_ratio, activation="gelu", name="ffn_expand"
        )
        self.ffn_project = layers.Dense(self.channels, name="ffn_project")

    def build(self, input_shape) -> None:
        token_shape = (None, self.window_size * self.window_size, self.channels)
        self.position.build(input_shape)
        self.norm1.build(input_shape)
        self.attention.build(token_shape, token_shape)
        self.norm2.build(input_shape)
        self.ffn_expand.build(input_shape)
        expanded_shape = tuple(input_shape[:-1]) + (self.channels * self.mlp_ratio,)
        self.ffn_project.build(expanded_shape)
        super().build(input_shape)

    def _region_mask(self, height: tf.Tensor, width: tf.Tensor) -> tf.Tensor:
        ws, shift = self.window_size, self.shift_size
        h = tf.range(height)
        w = tf.range(width)
        h_region = tf.where(
            h < height - ws,
            0,
            tf.where(h < height - shift, 1, 2),
        )
        w_region = tf.where(
            w < width - ws,
            0,
            tf.where(w < width - shift, 1, 2),
        )
        region = h_region[:, None] * 3 + w_region[None, :]
        region = tf.cast(region[None, :, :, None], tf.float32)
        windows = _window_partition(region, ws)
        ids = tf.squeeze(windows, axis=-1)
        return tf.equal(ids[:, :, None], ids[:, None, :])

    def call(self, inputs: tf.Tensor, training=None) -> tf.Tensor:
        tf.debugging.assert_equal(
            tf.shape(inputs)[-1],
            self.channels,
            message="WindowAttentionBlock channel mismatch",
        )
        positioned = inputs + self.residual_scale * self.position(inputs)
        shortcut = positioned
        x = self.norm1(positioned)
        shape = tf.shape(x)
        batch, height, width = shape[0], shape[1], shape[2]
        ws = self.window_size

        pad_h = tf.math.floormod(-height, ws)
        pad_w = tf.math.floormod(-width, ws)
        x = tf.pad(x, [[0, 0], [0, pad_h], [0, pad_w], [0, 0]])
        valid = tf.ones([1, height, width, 1], dtype=tf.bool)
        valid = tf.pad(
            valid,
            [[0, 0], [0, pad_h], [0, pad_w], [0, 0]],
            constant_values=False,
        )
        padded_shape = tf.shape(x)
        padded_h, padded_w = padded_shape[1], padded_shape[2]

        if self.shift_size:
            shift = [-self.shift_size, -self.shift_size]
            x = tf.roll(x, shift=shift, axis=[1, 2])
            valid = tf.roll(valid, shift=shift, axis=[1, 2])

        windows = _window_partition(x, ws)
        valid_windows = tf.squeeze(_window_partition(valid, ws), axis=-1)
        key_mask = valid_windows[:, None, :]
        attention_mask = tf.broadcast_to(
            key_mask,
            [tf.shape(key_mask)[0], ws * ws, ws * ws],
        )

        if self.shift_size:
            attention_mask = tf.logical_and(
                attention_mask, self._region_mask(padded_h, padded_w)
            )

        attention_mask = tf.tile(attention_mask, [batch, 1, 1])
        attended = self.attention(
            windows,
            windows,
            attention_mask=attention_mask,
            training=training,
        )
        x = _window_reverse(attended, padded_h, padded_w, ws)

        if self.shift_size:
            x = tf.roll(
                x,
                shift=[self.shift_size, self.shift_size],
                axis=[1, 2],
            )

        x = x[:, :height, :width, :]
        x = shortcut + self.residual_scale * x
        ffn = self.ffn_project(self.ffn_expand(self.norm2(x)))
        return x + self.residual_scale * ffn

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "channels": self.channels,
                "window_size": self.window_size,
                "num_heads": self.num_heads,
                "shifted": self.shifted,
                "mlp_ratio": self.mlp_ratio,
                "residual_scale": self.residual_scale,
            }
        )
        return config


@keras.utils.register_keras_serializable(package="bass.v2")
class RegularShiftedWindowPair(layers.Layer):
    """One regular and one shifted block, guaranteeing cross-window exchange."""

    def __init__(
        self,
        channels: int,
        window_size: int,
        num_heads: int,
        mlp_ratio: int = 2,
        residual_scale: float = RESIDUAL_SCALE,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.channels = int(channels)
        self.window_size = int(window_size)
        self.num_heads = int(num_heads)
        self.mlp_ratio = int(mlp_ratio)
        self.residual_scale = float(residual_scale)
        self.regular = WindowAttentionBlock(
            channels=self.channels,
            window_size=self.window_size,
            num_heads=self.num_heads,
            shifted=False,
            mlp_ratio=self.mlp_ratio,
            residual_scale=self.residual_scale,
            name="regular",
        )
        self.shifted = WindowAttentionBlock(
            channels=self.channels,
            window_size=self.window_size,
            num_heads=self.num_heads,
            shifted=True,
            mlp_ratio=self.mlp_ratio,
            residual_scale=self.residual_scale,
            name="shifted",
        )

    def build(self, input_shape) -> None:
        self.regular.build(input_shape)
        self.shifted.build(input_shape)
        super().build(input_shape)

    def call(self, inputs: tf.Tensor, training=None) -> tf.Tensor:
        x = self.regular(inputs, training=training)
        return self.shifted(x, training=training)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "channels": self.channels,
                "window_size": self.window_size,
                "num_heads": self.num_heads,
                "mlp_ratio": self.mlp_ratio,
                "residual_scale": self.residual_scale,
            }
        )
        return config


@keras.utils.register_keras_serializable(package="bass.v2")
class HybridConvAttentionBlock(layers.Layer):
    """Residual 50/50 fusion of local convolution and window attention."""

    def __init__(
        self,
        channels: int,
        window_size: int,
        num_heads: int,
        shifted: bool = False,
        residual_scale: float = RESIDUAL_SCALE,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.channels = int(channels)
        self.window_size = int(window_size)
        self.num_heads = int(num_heads)
        self.shifted = bool(shifted)
        self.residual_scale = float(residual_scale)
        if self.channels <= 0:
            raise ValueError("channels must be positive")
        if self.window_size <= 0:
            raise ValueError("window_size must be positive")
        if self.num_heads <= 0 or self.channels % self.num_heads:
            raise ValueError("num_heads must be positive and divide channels")
        if self.residual_scale <= 0:
            raise ValueError("residual_scale must be positive")
        self.local_depthwise = layers.DepthwiseConv2D(
            3, padding="same", activation="gelu", name="local_depthwise"
        )
        self.local_project = layers.Conv2D(
            self.channels, 1, activation=None, name="local_project"
        )
        self.context = WindowAttentionBlock(
            channels=self.channels,
            window_size=self.window_size,
            num_heads=self.num_heads,
            shifted=self.shifted,
            residual_scale=1.0,
            name="context",
        )

    def build(self, input_shape) -> None:
        self.local_depthwise.build(input_shape)
        self.local_project.build(input_shape)
        self.context.build(input_shape)
        super().build(input_shape)

    def call(self, inputs: tf.Tensor, training=None) -> tf.Tensor:
        local_delta = self.local_project(self.local_depthwise(inputs))
        context_output = self.context(inputs, training=training)
        context_delta = context_output - inputs
        fused_delta = 0.5 * local_delta + 0.5 * context_delta
        return inputs + self.residual_scale * fused_delta

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "channels": self.channels,
                "window_size": self.window_size,
                "num_heads": self.num_heads,
                "shifted": self.shifted,
                "residual_scale": self.residual_scale,
            }
        )
        return config
