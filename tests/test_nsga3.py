import numpy as np
import pytest

import bass.problem as problem_module
from bass.evaluation import EvaluationResult
from bass.nsga3 import NSGA3
from bass.problem import BASSProblem


class ToyProblem:
    n_var = 12
    n_obj = 3

    @staticmethod
    def evaluate(individual):
        ones = float(np.sum(individual))
        transitions = float(np.sum(individual[1:] != individual[:-1]))
        weighted = float(np.dot(individual, np.arange(1, 13)))
        return [ones, 12.0 - ones, weighted + transitions]


def test_nsga3_runs_and_preserves_population_size():
    optimizer = NSGA3(ToyProblem(), pop_size=12, n_gen=3, seed=42)
    population, non_dominated = optimizer.run()
    assert population["X"].shape == (12, 12)
    assert population["F"].shape == (12, 3)
    assert len(non_dominated["X"]) >= 1
    assert optimizer.n_eval == 48


def test_nsga3_is_reproducible():
    first = NSGA3(ToyProblem(), pop_size=8, n_gen=2, seed=7).run()[0]
    second = NSGA3(ToyProblem(), pop_size=8, n_gen=2, seed=7).run()[0]
    np.testing.assert_array_equal(first["X"], second["X"])
    np.testing.assert_allclose(first["F"], second["F"])


def test_problem_evaluation_is_cached():
    calls = []

    def objective(individual):
        calls.append(individual.copy())
        return [float(np.sum(individual)), 1.0]

    problem = BASSProblem(
        genome_version=1,
        include_flops=False,
        objective_fn=objective,
    )
    genome = np.zeros(problem.n_var, dtype=np.int8)
    assert problem.evaluate(genome) == problem.evaluate(genome)
    assert len(calls) == 1
    assert problem.cache_size == 1


def test_real_evaluations_cache_gray_aliases_by_phenotype(monkeypatch):
    calls = []

    def fake_evaluation(architecture, **kwargs):
        calls.append((architecture, kwargs))
        return EvaluationResult(score=3.0, params=2, flops=1, details={})

    monkeypatch.setattr(problem_module, "evaluate_architecture", fake_evaluation)
    problem = BASSProblem(genome_version=1)
    first = np.zeros(problem.n_var, dtype=np.int8)
    alias = first.copy()
    alias[:3] = [1, 1, 0]  # Gray 4; 4 mod 4 selects the same 16 channels.

    assert problem.evaluate(first) == problem.evaluate(alias)
    assert len(calls) == 1
    assert problem.cache_size == 1


def test_problem_rejects_malformed_individual():
    problem = BASSProblem(
        genome_version=1,
        include_flops=False,
        objective_fn=lambda _: [1.0, 2.0],
    )
    with pytest.raises(ValueError, match="shape"):
        problem.evaluate(np.zeros((1, problem.n_var), dtype=np.int8))
    malformed = np.zeros(problem.n_var, dtype=np.int8)
    malformed[0] = 2
    with pytest.raises(ValueError, match="binary"):
        problem.evaluate(malformed)
    fractional = np.zeros(problem.n_var, dtype=float)
    fractional[0] = 0.5
    with pytest.raises(ValueError, match="binary"):
        problem.evaluate(fractional)


def test_nsga3_handles_all_non_finite_objectives():
    class NonFiniteProblem:
        n_var = 6
        n_obj = 2

        @staticmethod
        def evaluate(_):
            return [float("nan"), float("inf")]

    population, _ = NSGA3(NonFiniteProblem(), pop_size=4, n_gen=1, seed=3).run()
    assert population["X"].shape == (4, 6)
    assert np.all(np.isinf(population["F"]))
