"""Semantic multi-objective optimization problem for BASS V3."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Sequence

import numpy as np

from .config import (
    BRANCH_COUNT,
    DEFAULT_HEAD_MODE,
    DEFAULT_SEED,
    EXCHANGE_SITES,
    EXCHANGE_STATE_COUNT,
    SEMANTIC_GENOME_LENGTH,
    UNIT_STATE_COUNT,
    UNITS_PER_BRANCH,
)
from .encoding import (
    block_to_state,
    canonicalize_genome,
    decode,
    encode,
    exchange_to_state,
    sample_canonical_genome,
    state_to_block,
    state_to_exchange,
)
from .evaluation import evaluate_architecture
from .genotype import canonicalize_architecture
from .variation import mutate_block, mutate_exchange


class BASSProblem:
    """IBASS problem with semantic branch and exchange variation."""

    genome_version = 3
    genome_kind = "interaction-semantic"

    def __init__(
        self,
        *,
        metric: str = "gradient_flow",
        upscale_factor: int = 2,
        input_shape: tuple[int, int, int] = (64, 64, 3),
        include_flops: bool = True,
        evaluation_seed: int = DEFAULT_SEED,
        head_mode: str = DEFAULT_HEAD_MODE,
        exchange_probability: float | None = None,
        objective_fn: Callable[[np.ndarray], Sequence[float]] | None = None,
    ):
        metric_key = metric.lower()
        if metric_key == "synflow":
            raise ValueError(
                "V3 does not implement canonical SynFlow; use 'gradient_flow'"
            )
        if metric_key not in {"gradient_flow", "psnr"}:
            raise ValueError("metric must be 'gradient_flow' or 'psnr'")
        if exchange_probability is not None and not 0.0 <= exchange_probability <= 1.0:
            raise ValueError("exchange_probability must lie in [0, 1] or be None")
        self.n_var = SEMANTIC_GENOME_LENGTH
        self.unit_gene_end = 1 + BRANCH_COUNT * UNITS_PER_BRANCH
        self.n_obj = 3 if include_flops else 2
        self.xl = np.zeros(self.n_var, dtype=np.int16)
        self.xu = np.asarray(
            [3]
            + [UNIT_STATE_COUNT - 1] * (BRANCH_COUNT * UNITS_PER_BRANCH)
            + [EXCHANGE_STATE_COUNT - 1] * EXCHANGE_SITES,
            dtype=np.int16,
        )
        self.metric = metric_key
        self.upscale_factor = upscale_factor
        self.input_shape = input_shape
        self.include_flops = include_flops
        self.evaluation_seed = int(evaluation_seed)
        self.head_mode = head_mode
        self.exchange_probability = (
            None if exchange_probability is None else float(exchange_probability)
        )
        self.objective_fn = objective_fn
        self._cache: dict[str | tuple[int, ...], list[float]] = {}
        self._mutation_transition_counts: Counter[str] = Counter()

    def canonicalize_individual(self, individual: np.ndarray) -> np.ndarray:
        raw = np.asarray(individual)
        if raw.shape != (self.n_var,):
            raise ValueError(f"Expected a V3 individual with shape ({self.n_var},)")
        if not np.all(np.equal(raw, np.floor(raw))):
            raise ValueError("V3 semantic genes must be integers")
        return np.asarray(canonicalize_genome(raw.tolist()), dtype=np.int16)

    def canonical_key(self, individual: np.ndarray) -> str:
        return decode(self.canonicalize_individual(individual)).canonical_hash()

    def sample_individual(self, rng: np.random.Generator) -> np.ndarray:
        seed = int(rng.integers(0, np.iinfo(np.int32).max))
        return np.asarray(
            sample_canonical_genome(
                seed=seed,
                exchange_probability=self.exchange_probability,
            ),
            dtype=np.int16,
        )

    def crossover(
        self,
        left: np.ndarray,
        right: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        left_spec = decode(self.canonicalize_individual(left))
        right_spec = decode(self.canonicalize_individual(right))
        channels = left_spec.channels if rng.random() < 0.5 else right_spec.channels
        branch_pool = left_spec.branches + right_spec.branches
        selected = rng.choice(len(branch_pool), size=3, replace=False)
        branches = tuple(branch_pool[int(index)] for index in selected)
        exchanges = tuple(
            left_exchange if rng.random() < 0.5 else right_exchange
            for left_exchange, right_exchange in zip(
                left_spec.exchanges, right_spec.exchanges
            )
        )
        return np.asarray(
            encode(canonicalize_architecture(channels, branches, exchanges)),
            dtype=np.int16,
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
            self._mutation_transition_counts["channel"] += 1
        for index in np.flatnonzero(mask[1 : self.unit_gene_end]) + 1:
            block, transition = mutate_block(state_to_block(int(child[index])), rng)
            child[index] = block_to_state(block)
            self._mutation_transition_counts[transition] += 1
        for index in np.flatnonzero(mask[self.unit_gene_end :]) + self.unit_gene_end:
            exchange, transition = mutate_exchange(
                state_to_exchange(int(child[index])), rng
            )
            child[index] = exchange_to_state(exchange)
            self._mutation_transition_counts[transition] += 1
        return self.canonicalize_individual(child)

    def evaluate(
        self, individual: np.ndarray, n_eval: int | None = None
    ) -> list[float]:
        del n_eval
        normalized = self.canonicalize_individual(individual)
        if not np.array_equal(normalized, individual):
            raise ValueError("V3 population may contain only canonical individuals")
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

    @property
    def mutation_transition_counts(self) -> dict[str, int]:
        """Attempted semantic moves, including duplicate-rejected offspring."""

        return dict(sorted(self._mutation_transition_counts.items()))


IBASSProblem = BASSProblem
