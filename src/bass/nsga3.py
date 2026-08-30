"""Compact NSGA-III implementation for the original BASS demonstration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReferencePoint:
    position: np.ndarray


class NSGA3:
    """Three-objective evolutionary search with reference-direction niching."""

    def __init__(
        self,
        problem,
        *,
        pop_size: int = 20,
        n_gen: int = 10,
        crossover_probability: float = 0.9,
        mutation_probability: float | None = None,
        divisions: int = 4,
        seed: int = 42,
        verbose: bool = False,
    ):
        if pop_size < 2:
            raise ValueError("pop_size must be at least 2")
        if n_gen < 0:
            raise ValueError("n_gen cannot be negative")
        if problem.n_var < 3:
            raise ValueError("The problem must expose at least three variables")
        if problem.n_obj < 2:
            raise ValueError("The problem must expose at least two objectives")
        if not 0.0 <= crossover_probability <= 1.0:
            raise ValueError("crossover_probability must lie in [0, 1]")
        if mutation_probability is not None and not 0.0 <= mutation_probability <= 1.0:
            raise ValueError("mutation_probability must lie in [0, 1]")
        if divisions < 1:
            raise ValueError("divisions must be positive")
        self.problem = problem
        self.pop_size = int(pop_size)
        self.n_gen = int(n_gen)
        self.crossover_probability = float(crossover_probability)
        self.mutation_probability = (
            1.0 / problem.n_var
            if mutation_probability is None
            else float(mutation_probability)
        )
        self.divisions = int(divisions)
        self.verbose = bool(verbose)
        self.rng = np.random.default_rng(seed)
        self.n_eval = 0
        self.ref_points = self._reference_points(problem.n_obj, self.divisions)

    @staticmethod
    def _dominates(left: np.ndarray, right: np.ndarray) -> bool:
        return bool(np.all(left <= right) and np.any(left < right))

    def _non_dominated_fronts(self, objectives: np.ndarray) -> list[list[int]]:
        size = len(objectives)
        dominates = [[] for _ in range(size)]
        domination_count = np.zeros(size, dtype=int)
        first = []
        for left in range(size):
            for right in range(size):
                if left == right:
                    continue
                if self._dominates(objectives[left], objectives[right]):
                    dominates[left].append(right)
                elif self._dominates(objectives[right], objectives[left]):
                    domination_count[left] += 1
            if domination_count[left] == 0:
                first.append(left)

        fronts = [first]
        while fronts[-1]:
            following = []
            for left in fronts[-1]:
                for right in dominates[left]:
                    domination_count[right] -= 1
                    if domination_count[right] == 0:
                        following.append(right)
            if following:
                fronts.append(following)
            else:
                break
        return fronts

    @staticmethod
    def _reference_points(objectives: int, divisions: int) -> np.ndarray:
        if objectives < 2 or divisions < 1:
            raise ValueError("At least two objectives and one division are required")
        points = []

        def populate(prefix: list[int], remaining: int, dimensions: int) -> None:
            if dimensions == 1:
                points.append(prefix + [remaining])
                return
            for value in range(remaining + 1):
                populate(prefix + [value], remaining - value, dimensions - 1)

        populate([], divisions, objectives)
        return np.asarray(points, dtype=float) / divisions

    def _evaluate(self, individual: np.ndarray) -> np.ndarray:
        values = np.asarray(self.problem.evaluate(individual), dtype=float)
        self.n_eval += 1
        if values.shape != (self.problem.n_obj,):
            raise ValueError(
                f"Expected {self.problem.n_obj} objective values, got {values.shape}"
            )
        values[~np.isfinite(values)] = np.inf
        return values

    def _initialize(self) -> tuple[np.ndarray, np.ndarray]:
        genomes = self.rng.integers(
            0, 2, size=(self.pop_size, self.problem.n_var), dtype=np.int8
        )
        objectives = np.asarray([self._evaluate(item) for item in genomes])
        return genomes, objectives

    def _rank_map(self, objectives: np.ndarray) -> np.ndarray:
        ranks = np.empty(len(objectives), dtype=int)
        for rank, front in enumerate(self._non_dominated_fronts(objectives)):
            ranks[front] = rank
        return ranks

    def _tournament(self, genomes: np.ndarray, ranks: np.ndarray) -> np.ndarray:
        left, right = self.rng.choice(len(genomes), size=2, replace=False)
        if ranks[left] < ranks[right]:
            return genomes[left]
        if ranks[right] < ranks[left]:
            return genomes[right]
        return genomes[left if self.rng.random() < 0.5 else right]

    def _crossover(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        child = left.copy()
        if self.rng.random() >= self.crossover_probability:
            return child
        point_a, point_b = sorted(
            self.rng.choice(np.arange(1, self.problem.n_var), size=2, replace=False)
        )
        child[point_a:point_b] = right[point_a:point_b]
        return child

    def _mutate(self, individual: np.ndarray) -> np.ndarray:
        mask = self.rng.random(self.problem.n_var) < self.mutation_probability
        individual[mask] = 1 - individual[mask]
        return individual

    @staticmethod
    def _normalize(objectives: np.ndarray) -> np.ndarray:
        normalized = np.full(objectives.shape, 1e12, dtype=float)
        for column_index in range(objectives.shape[1]):
            column = objectives[:, column_index]
            finite = np.isfinite(column)
            if not np.any(finite):
                continue
            minimum = np.min(column[finite])
            maximum = np.max(column[finite])
            span = maximum - minimum if maximum > minimum else 1.0
            normalized[finite, column_index] = (column[finite] - minimum) / span
        return normalized

    def _associate(self, normalized: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        associations = np.empty(len(normalized), dtype=int)
        distances = np.empty(len(normalized), dtype=float)
        for index, point in enumerate(normalized):
            best_reference, best_distance = 0, np.inf
            for ref_index, direction in enumerate(self.ref_points):
                norm = np.linalg.norm(direction)
                projection = np.dot(point, direction) / (norm * norm)
                distance = np.linalg.norm(point - projection * direction)
                if distance < best_distance:
                    best_reference, best_distance = ref_index, distance
            associations[index] = best_reference
            distances[index] = best_distance
        return associations, distances

    def _environmental_selection(
        self, genomes: np.ndarray, objectives: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        fronts = self._non_dominated_fronts(objectives)
        selected: list[int] = []
        last_front: list[int] = []
        for front in fronts:
            if len(selected) + len(front) <= self.pop_size:
                selected.extend(front)
            else:
                last_front = list(front)
                break

        if len(selected) < self.pop_size and last_front:
            normalized = self._normalize(objectives)
            associations, distances = self._associate(normalized)
            niche_count = np.zeros(len(self.ref_points), dtype=int)
            for index in selected:
                niche_count[associations[index]] += 1

            remaining = set(last_front)
            while len(selected) < self.pop_size and remaining:
                available_refs = sorted({associations[index] for index in remaining})
                minimum_niche = min(niche_count[ref] for ref in available_refs)
                ref_candidates = [
                    ref for ref in available_refs if niche_count[ref] == minimum_niche
                ]
                ref = int(self.rng.choice(ref_candidates))
                candidates = [
                    index for index in remaining if associations[index] == ref
                ]
                if niche_count[ref] == 0:
                    chosen = min(candidates, key=lambda index: distances[index])
                else:
                    chosen = int(self.rng.choice(candidates))
                selected.append(chosen)
                remaining.remove(chosen)
                niche_count[ref] += 1

        selected_array = np.asarray(selected[: self.pop_size], dtype=int)
        return genomes[selected_array], objectives[selected_array]

    def run(self) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        genomes, objectives = self._initialize()
        for generation in range(self.n_gen):
            ranks = self._rank_map(objectives)
            children = []
            child_objectives = []
            for _ in range(self.pop_size):
                left = self._tournament(genomes, ranks)
                right = self._tournament(genomes, ranks)
                child = self._mutate(self._crossover(left, right))
                children.append(child)
                child_objectives.append(self._evaluate(child))
            combined_x = np.vstack([genomes, np.asarray(children)])
            combined_f = np.vstack([objectives, np.asarray(child_objectives)])
            genomes, objectives = self._environmental_selection(combined_x, combined_f)
            if self.verbose:
                print(
                    f"generation={generation + 1}/{self.n_gen} "
                    f"evaluations={self.n_eval}"
                )

        first_front = self._non_dominated_fronts(objectives)[0]
        population = {"X": genomes, "F": objectives}
        non_dominated = {
            "X": genomes[first_front],
            "F": objectives[first_front],
        }
        return population, non_dominated
