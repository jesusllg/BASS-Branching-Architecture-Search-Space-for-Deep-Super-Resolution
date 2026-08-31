"""Canonical semantic optimization problem for BASS V2."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np

from .config import (
    DEFAULT_HEAD_MODE,
    DEFAULT_SEED,
    SEMANTIC_GENOME_LENGTH,
    UNIT_STATE_COUNT,
)
from .encoding import canonicalize_genome, decode, encode, sample_genome
from .evaluation import evaluate_architecture
from .genotype import canonicalize_architecture


class BASSProblem:
    """V2 problem with architecture-aware sampling, crossover, and mutation."""

    genome_version = 2
    genome_kind = "canonical-semantic"

    def __init__(
        self,
        *,
        metric: str = "gradient_flow",
        upscale_factor: int = 2,
        input_shape: tuple[int, int, int] = (64, 64, 3),
        include_flops: bool = True,
        evaluation_seed: int = DEFAULT_SEED,
        head_mode: str = DEFAULT_HEAD_MODE,
        objective_fn: Callable[[np.ndarray], Sequence[float]] | None = None,
    ):
        metric_key = metric.lower()
        if metric_key == "synflow":
            raise ValueError(
                "V2 does not implement canonical SynFlow; use 'gradient_flow'"
            )
        if metric_key not in {"gradient_flow", "psnr"}:
            raise ValueError("metric must be 'gradient_flow' or 'psnr'")
        self.n_var = SEMANTIC_GENOME_LENGTH
        self.n_obj = 3 if include_flops else 2
        self.xl = np.zeros(self.n_var, dtype=np.int16)
        self.xu = np.asarray(
            [3] + [UNIT_STATE_COUNT - 1] * (self.n_var - 1), dtype=np.int16
        )
        self.metric = metric_key
        self.upscale_factor = upscale_factor
        self.input_shape = input_shape
        self.include_flops = include_flops
        self.evaluation_seed = int(evaluation_seed)
        self.head_mode = head_mode
        self.objective_fn = objective_fn
        self._cache: dict[str | tuple[int, ...], list[float]] = {}

    def canonicalize_individual(self, individual: np.ndarray) -> np.ndarray:
        raw = np.asarray(individual)
        if raw.shape != (self.n_var,):
            raise ValueError(f"Expected a V2 individual with shape ({self.n_var},)")
        if not np.all(np.equal(raw, np.floor(raw))):
            raise ValueError("V2 semantic genes must be integers")
        return np.asarray(canonicalize_genome(raw.tolist()), dtype=np.int16)

    def canonical_key(self, individual: np.ndarray) -> str:
        return decode(self.canonicalize_individual(individual)).canonical_hash()

    def sample_individual(self, rng: np.random.Generator) -> np.ndarray:
        seed = int(rng.integers(0, np.iinfo(np.int32).max))
        return np.asarray(sample_genome(seed=seed), dtype=np.int16)

    def crossover(
        self,
        left: np.ndarray,
        right: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        left_spec = decode(self.canonicalize_individual(left))
        right_spec = decode(self.canonicalize_individual(right))
        channels = left_spec.channels if rng.random() < 0.5 else right_spec.channels
        branches = tuple(
            left_branch if rng.random() < 0.5 else right_branch
            for left_branch, right_branch in zip(
                left_spec.branches, right_spec.branches
            )
        )
        return np.asarray(
            encode(canonicalize_architecture(channels, branches)), dtype=np.int16
        )

    def mutate(
        self,
        individual: np.ndarray,
        rng: np.random.Generator,
        probability: float,
    ) -> np.ndarray:
        child = self.canonicalize_individual(individual).copy()
        mask = rng.random(self.n_var) < probability
        if not np.any(mask):
            return child
        if mask[0]:
            choices = [value for value in range(4) if value != int(child[0])]
            child[0] = int(rng.choice(choices))
        for index in np.flatnonzero(mask[1:]) + 1:
            choices = np.arange(UNIT_STATE_COUNT, dtype=np.int16)
            choices = choices[choices != child[index]]
            child[index] = int(rng.choice(choices))
        return self.canonicalize_individual(child)

    def evaluate(
        self, individual: np.ndarray, n_eval: int | None = None
    ) -> list[float]:
        del n_eval
        normalized = self.canonicalize_individual(individual)
        if not np.array_equal(normalized, individual):
            raise ValueError("V2 population may contain only canonical individuals")
        architecture = decode(normalized)
        cache_key: str | tuple[int, ...]
        if self.objective_fn is None:
            cache_key = architecture.canonical_hash()
        else:
            cache_key = tuple(int(value) for value in normalized)
        if cache_key in self._cache:
            return list(self._cache[cache_key])

        if self.objective_fn is not None:
            objectives = [
                float(value) for value in self.objective_fn(normalized.copy())
            ]
        else:
            result = evaluate_architecture(
                architecture,
                metric=self.metric,
                upscale_factor=self.upscale_factor,
                input_shape=self.input_shape,
                include_flops=self.include_flops,
                evaluation_seed=self.evaluation_seed,
                head_mode=self.head_mode,
            )
            if not math.isfinite(result.score):
                objectives = [float("inf")] * self.n_obj
            else:
                objectives = [-result.score, float(result.params)]
                if self.include_flops:
                    objectives.append(float(result.flops))
        if len(objectives) != self.n_obj:
            raise ValueError(
                f"Objective function returned {len(objectives)} values; "
                f"expected {self.n_obj}"
            )
        self._cache[cache_key] = list(objectives)
        return objectives

    @property
    def cache_size(self) -> int:
        return len(self._cache)
