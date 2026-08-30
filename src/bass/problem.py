"""Optimization-problem adapter for bitstring-based search algorithms."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np

from .config import DEFAULT_SEED, V1_GENOME_BITS, V2_GENOME_BITS
from .encoding import decode
from .evaluation import evaluate_architecture


class BASSProblem:
    def __init__(
        self,
        *,
        genome_version: int = 1,
        metric: str = "synflow",
        upscale_factor: int = 2,
        input_shape: tuple[int, int, int] = (64, 64, 3),
        include_flops: bool = True,
        evaluation_seed: int = DEFAULT_SEED,
        objective_fn: Callable[[np.ndarray], Sequence[float]] | None = None,
    ):
        if genome_version not in {1, 2}:
            raise ValueError("genome_version must be 1 or 2")
        self.genome_version = genome_version
        self.n_var = V1_GENOME_BITS if genome_version == 1 else V2_GENOME_BITS
        self.n_obj = 3 if include_flops else 2
        self.xl = np.zeros(self.n_var, dtype=np.int8)
        self.xu = np.ones(self.n_var, dtype=np.int8)
        self.metric = metric
        self.upscale_factor = upscale_factor
        self.input_shape = input_shape
        self.include_flops = include_flops
        self.evaluation_seed = int(evaluation_seed)
        self.objective_fn = objective_fn
        self._cache: dict[str | tuple[int, ...], list[float]] = {}

    def evaluate(
        self, individual: np.ndarray, n_eval: int | None = None
    ) -> list[float]:
        del n_eval
        raw = np.asarray(individual)
        if raw.shape != (self.n_var,):
            raise ValueError(f"Expected an individual with shape ({self.n_var},)")
        if np.any((raw != 0) & (raw != 1)):
            raise ValueError("Individuals may only contain binary values")
        normalized = raw.astype(np.int8, copy=False)
        genome_key = tuple(int(value) for value in normalized)

        if self.objective_fn is not None:
            cache_key: str | tuple[int, ...] = genome_key
            architecture = None
        else:
            architecture = decode(genome_key)
            cache_key = architecture.canonical_hash()

        if cache_key in self._cache:
            return list(self._cache[cache_key])

        if self.objective_fn is not None:
            objectives = [
                float(value) for value in self.objective_fn(normalized.copy())
            ]
        else:
            assert architecture is not None
            result = evaluate_architecture(
                architecture,
                metric=self.metric,
                upscale_factor=self.upscale_factor,
                input_shape=self.input_shape,
                include_flops=self.include_flops,
                evaluation_seed=self.evaluation_seed,
            )
            if not math.isfinite(result.score):
                objectives = [float("inf")] * self.n_obj
            else:
                objectives = [-result.score, float(result.params)]
                if self.include_flops:
                    objectives.append(float(result.flops))

        if len(objectives) != self.n_obj:
            raise ValueError(
                f"Objective function returned {len(objectives)} values; expected {self.n_obj}"
            )
        self._cache[cache_key] = list(objectives)
        return objectives

    @property
    def cache_size(self) -> int:
        return len(self._cache)
