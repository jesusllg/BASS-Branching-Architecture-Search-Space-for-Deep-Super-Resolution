"""Evaluation and proxy diagnostics for interaction-aware BASS V3."""

from __future__ import annotations

import math
from collections.abc import Iterable

import tensorflow as tf

from bass.v2.evaluation import (
    EvaluationResult,
    GradientDiagnostics,
    count_flops,
    count_params,
    gradient_flow_diagnostics,
    gradient_flow_score,
    psnr,
    ssim,
)

from .config import DEFAULT_HEAD_MODE, DEFAULT_SEED
from .genotype import ArchitectureSpec
from .model_builder import build_model


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
        raise ValueError("V3 does not implement canonical SynFlow; use 'gradient_flow'")
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
            details = {
                "metric": "gradient_flow",
                "interaction_aware": True,
                **diagnostics.to_dict(),
            }
        else:
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
                "interaction_aware": True,
                "val_ssim": float(history.history["val_ssim"][-1]),
            }
        if not math.isfinite(float(score)):
            raise ValueError("V3 evaluation produced a non-finite score")
        return EvaluationResult(float(score), params, flops, details)
    finally:
        del model
        tf.keras.backend.clear_session()


def evaluate_model(architecture: ArchitectureSpec, **kwargs) -> float:
    return -evaluate_architecture(architecture, **kwargs).score


calculate_model_flops = count_flops


def synflow_metric_nas(*args, **kwargs):
    """Reject the retired, scientifically misleading V3 compatibility name."""

    del args, kwargs
    raise RuntimeError(
        "bass.v3.evaluation.synflow_metric_nas is retired: the implemented "
        "quantity is gradient_flow_score, not canonical SynFlow"
    )


__all__ = [
    "EvaluationResult",
    "GradientDiagnostics",
    "calculate_model_flops",
    "count_flops",
    "count_params",
    "evaluate_architecture",
    "evaluate_model",
    "gradient_flow_diagnostics",
    "gradient_flow_score",
    "psnr",
    "ssim",
    "synflow_metric_nas",
]
