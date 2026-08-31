"""Model evaluation utilities shared by the legacy demo and new runners."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import tensorflow as tf

from .config import DEFAULT_SEED
from .genotype import ArchitectureSpec
from .model_builder import build_model


def psnr(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Mean batch PSNR for inputs normalized to [0, 1]."""

    return tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=1.0))


def ssim(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))


def gradient_flow_score(
    model: tf.keras.Model,
    *,
    input_shape: tuple[int, int, int] = (64, 64, 3),
) -> float:
    """Deterministic SynFlow-style parameter-gradient product.

    This retains the intent of the original public implementation while being
    safe for variables with disconnected gradients. It is a baseline proxy,
    not the future AZ-SR implementation.
    """

    sample = tf.ones((1, *input_shape), dtype=tf.float32)
    with tf.GradientTape() as tape:
        output = model(sample, training=False)
        objective = tf.reduce_sum(output)
    gradients = tape.gradient(objective, model.trainable_variables)
    terms = [
        tf.reduce_sum(tf.abs(variable * gradient))
        for variable, gradient in zip(model.trainable_variables, gradients)
        if gradient is not None
    ]
    if not terms:
        return float("nan")
    score = tf.add_n(terms)
    value = float(score.numpy())
    return value if math.isfinite(value) else float("nan")


# Historical function name retained.
synflow_metric_nas = gradient_flow_score


def count_flops(
    model: tf.keras.Model,
    *,
    input_shape: tuple[int, int, int] = (64, 64, 3),
) -> int:
    """Count TensorFlow floating-point operations for one forward pass."""

    from tensorflow.python.framework.convert_to_constants import (
        convert_variables_to_constants_v2,
    )

    @tf.function
    def forward(inputs):
        return model(inputs, training=False)

    concrete = forward.get_concrete_function(
        tf.TensorSpec((1, *input_shape), tf.float32)
    )
    frozen = convert_variables_to_constants_v2(concrete)
    options = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    options["output"] = "none"
    profile = tf.compat.v1.profiler.profile(
        graph=frozen.graph,
        run_meta=tf.compat.v1.RunMetadata(),
        cmd="op",
        options=options,
    )
    return int(profile.total_float_ops if profile is not None else 0)


@dataclass(frozen=True)
class EvaluationResult:
    score: float
    params: int
    flops: int
    details: dict


def evaluate_architecture(
    architecture: ArchitectureSpec,
    *,
    metric: str = "synflow",
    upscale_factor: int = 2,
    input_shape: tuple[int, int, int] = (64, 64, 3),
    train_dataset: Iterable | None = None,
    validation_dataset: Iterable | None = None,
    epochs: int = 5,
    include_flops: bool = True,
    evaluation_seed: int = DEFAULT_SEED,
) -> EvaluationResult:
    """Build once and evaluate quality, parameters and optional FLOPs."""

    tf.keras.utils.set_random_seed(evaluation_seed)
    model = build_model(
        architecture,
        upscale_factor=upscale_factor,
        input_channels=input_shape[-1],
        input_shape=input_shape,
    )
    try:
        params = int(model.count_params())
        flops = count_flops(model, input_shape=input_shape) if include_flops else 0

        metric_key = metric.lower()
        if metric_key in {"synflow", "gradient_flow"}:
            score = gradient_flow_score(model, input_shape=input_shape)
            details = {"metric": "gradient_flow"}
        elif metric_key == "psnr":
            if train_dataset is None or validation_dataset is None:
                raise ValueError(
                    "train_dataset and validation_dataset are required for PSNR"
                )
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                loss=tf.keras.losses.MeanSquaredError(),
                metrics=[psnr, ssim],
            )
            history = model.fit(
                train_dataset,
                validation_data=validation_dataset,
                epochs=epochs,
                verbose=0,
            )
            score = float(history.history["val_psnr"][-1])
            details = {
                "metric": "psnr",
                "val_ssim": float(history.history["val_ssim"][-1]),
            }
        else:
            raise ValueError("metric must be 'synflow', 'gradient_flow', or 'psnr'")

        return EvaluationResult(
            score=float(score), params=params, flops=flops, details=details
        )
    finally:
        del model
        tf.keras.backend.clear_session()


def evaluate_model(architecture: ArchitectureSpec, **kwargs) -> float:
    """Historical scalar API; returns a minimization-ready negative score."""

    return -evaluate_architecture(architecture, **kwargs).score


# Historical helper names used by the old NSGA-III module.
calculate_model_flops = count_flops


def count_params(model: tf.keras.Model) -> int:
    return int(model.count_params())
