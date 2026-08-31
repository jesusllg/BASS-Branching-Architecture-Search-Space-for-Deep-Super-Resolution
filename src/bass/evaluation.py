"""Compatibility evaluation dispatcher for BASS V1 and V2."""

from .v1.evaluation import (
    EvaluationResult,
    count_flops,
    count_params,
    gradient_flow_score,
    psnr,
    ssim,
    synflow_metric_nas,
)
from .v1.genotype import ArchitectureSpec as V1ArchitectureSpec
from .v2.genotype import ArchitectureSpec as V2ArchitectureSpec


def evaluate_architecture(architecture, **kwargs):
    if isinstance(architecture, V1ArchitectureSpec):
        from .v1.evaluation import evaluate_architecture as evaluate_v1

        return evaluate_v1(architecture, **kwargs)
    if isinstance(architecture, V2ArchitectureSpec):
        from .v2.evaluation import evaluate_architecture as evaluate_v2

        return evaluate_v2(architecture, **kwargs)
    raise TypeError("Use a bass.v1 or bass.v2 ArchitectureSpec")


def evaluate_model(architecture, **kwargs) -> float:
    return -evaluate_architecture(architecture, **kwargs).score


calculate_model_flops = count_flops

__all__ = [
    "EvaluationResult",
    "calculate_model_flops",
    "count_flops",
    "count_params",
    "evaluate_architecture",
    "evaluate_model",
    "gradient_flow_score",
    "psnr",
    "ssim",
    "synflow_metric_nas",
]
