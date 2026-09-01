"""Evaluation and proxy diagnostics for canonical hybrid BASS V2 models."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import tensorflow as tf

from .config import DEFAULT_HEAD_MODE, DEFAULT_SEED
from .genotype import ArchitectureSpec
from .model_builder import build_model


def psnr(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    prediction = tf.clip_by_value(y_pred, 0.0, 1.0)
    return tf.reduce_mean(tf.image.psnr(y_true, prediction, max_val=1.0))


def ssim(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    prediction = tf.clip_by_value(y_pred, 0.0, 1.0)
    return tf.reduce_mean(tf.image.ssim(y_true, prediction, max_val=1.0))


@dataclass(frozen=True)
class GradientDiagnostics:
    score: float
    trainable_variables: int
    connected_variables: int
    disconnected_variables: tuple[str, ...]
    non_finite_variables: tuple[str, ...]

    @property
    def coverage(self) -> float:
        if self.trainable_variables == 0:
            return 0.0
        return self.connected_variables / self.trainable_variables

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "trainable_variables": self.trainable_variables,
            "connected_variables": self.connected_variables,
            "gradient_coverage": self.coverage,
            "disconnected_variables": list(self.disconnected_variables),
            "non_finite_variables": list(self.non_finite_variables),
        }


def gradient_flow_diagnostics(
    model: tf.keras.Model,
    *,
    input_shape: tuple[int, int, int] = (64, 64, 3),
    strict: bool = True,
) -> GradientDiagnostics:
    """Compute the repository baseline and expose every gradient failure.

    This is intentionally named ``gradient_flow``. It is not canonical SynFlow
    and must not be presented as the intended AZ-score pipeline.
    """

    sample = tf.ones((1, *input_shape), dtype=tf.float32)
    with tf.GradientTape() as tape:
        output = model(sample, training=False)
        objective = tf.reduce_sum(output)
    variables = tuple(model.trainable_variables)
    gradients = tape.gradient(objective, variables)

    def variable_name(variable: tf.Variable) -> str:
        # Keras 3's short ``name`` (for example, ``kernel``) is not unique.
        return str(getattr(variable, "path", variable.name))

    disconnected = tuple(
        variable_name(variable)
        for variable, gradient in zip(variables, gradients)
        if gradient is None
    )
    non_finite = tuple(
        variable_name(variable)
        for variable, gradient in zip(variables, gradients)
        if gradient is not None and not bool(tf.reduce_all(tf.math.is_finite(gradient)))
    )
    terms = [
        tf.reduce_sum(tf.abs(variable * gradient))
        for variable, gradient in zip(variables, gradients)
        if gradient is not None and variable_name(variable) not in non_finite
    ]
    value = float(tf.add_n(terms).numpy()) if terms else float("nan")
    if not math.isfinite(value):
        non_finite = (*non_finite, "aggregate_score")
    diagnostics = GradientDiagnostics(
        score=value,
        trainable_variables=len(variables),
        connected_variables=len(variables) - len(disconnected),
        disconnected_variables=disconnected,
        non_finite_variables=non_finite,
    )
    if strict and (disconnected or non_finite):
        raise ValueError(
            "Invalid gradient-flow evaluation: "
            f"disconnected={list(disconnected)}, non_finite={list(non_finite)}"
        )
    return diagnostics


def gradient_flow_score(
    model: tf.keras.Model,
    *,
    input_shape: tuple[int, int, int] = (64, 64, 3),
) -> float:
    return gradient_flow_diagnostics(model, input_shape=input_shape, strict=True).score


def count_flops(
    model: tf.keras.Model,
    *,
    input_shape: tuple[int, int, int] = (64, 64, 3),
) -> int:
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
    metric: str = "gradient_flow",
    upscale_factor: int = 2,
    input_shape: tuple[int, int, int] = (64, 64, 3),
    train_dataset: Iterable | None = None,
    validation_dataset: Iterable | None = None,
    epochs: int = 5,
    include_flops: bool = True,
    evaluation_seed: int = DEFAULT_SEED,
    head_mode: str = DEFAULT_HEAD_MODE,
) -> EvaluationResult:
    metric_key = metric.lower()
    if metric_key == "synflow":
        raise ValueError("V2 does not implement canonical SynFlow; use 'gradient_flow'")
    if metric_key not in {"gradient_flow", "psnr"}:
        raise ValueError("metric must be 'gradient_flow' or 'psnr'")
    if metric_key == "psnr" and (train_dataset is None or validation_dataset is None):
        raise ValueError("train_dataset and validation_dataset are required for PSNR")

    tf.keras.utils.set_random_seed(evaluation_seed)
    model = build_model(
        architecture,
        upscale_factor=upscale_factor,
        input_channels=input_shape[-1],
        input_shape=input_shape,
        head_mode=head_mode,
    )
    try:
        params = int(model.count_params())
        flops = count_flops(model, input_shape=input_shape) if include_flops else 0
        if metric_key == "gradient_flow":
            diagnostics = gradient_flow_diagnostics(
                model, input_shape=input_shape, strict=True
            )
            score = diagnostics.score
            details = {"metric": "gradient_flow", **diagnostics.to_dict()}
        elif metric_key == "psnr":
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
        return EvaluationResult(float(score), params, flops, details)
    finally:
        del model
        tf.keras.backend.clear_session()


def evaluate_model(architecture: ArchitectureSpec, **kwargs) -> float:
    return -evaluate_architecture(architecture, **kwargs).score


calculate_model_flops = count_flops


def synflow_metric_nas(*args, **kwargs):
    """Reject the retired, scientifically misleading V2 compatibility name."""

    del args, kwargs
    raise RuntimeError(
        "bass.v2.evaluation.synflow_metric_nas is retired: the implemented "
        "quantity is gradient_flow_score, not canonical SynFlow"
    )


def count_params(model: tf.keras.Model) -> int:
    return int(model.count_params())
