import numpy as np

from bass.shared.nsga3 import NSGA3
from bass.v2 import decode
from bass.v2.problem import BASSProblem


def make_problem():
    return BASSProblem(
        include_flops=False,
        objective_fn=lambda genome: [
            float(np.sum(genome)),
            float(np.sum(genome[1:] > 21)),
        ],
    )


def test_v2_search_population_is_canonical_and_unique():
    problem = make_problem()
    optimizer = NSGA3(problem, pop_size=12, n_gen=2, seed=17)
    population, _ = optimizer.run()

    assert population["X"].shape == (12, 10)
    specs = [decode(individual) for individual in population["X"]]
    hashes = [spec.canonical_hash() for spec in specs]
    assert len(set(hashes)) == len(hashes)


def test_v2_semantic_search_is_reproducible():
    first_optimizer = NSGA3(make_problem(), pop_size=8, n_gen=2, seed=23)
    second_optimizer = NSGA3(make_problem(), pop_size=8, n_gen=2, seed=23)
    first = first_optimizer.run()[0]
    second = second_optimizer.run()[0]
    np.testing.assert_array_equal(first["X"], second["X"])
    np.testing.assert_allclose(first["F"], second["F"])
    assert first_optimizer.history == second_optimizer.history
    assert [record["generation"] for record in first_optimizer.history] == [1, 2]


def test_v2_rejects_noncanonical_population_members():
    problem = make_problem()
    invalid = np.zeros(10, dtype=np.int16)
    invalid[2] = 1  # active unit after an empty slot
    canonical = problem.canonicalize_individual(invalid)
    assert not np.array_equal(invalid, canonical)
    try:
        problem.evaluate(invalid)
    except ValueError as error:
        assert "canonical" in str(error)
    else:
        raise AssertionError("Noncanonical genome was silently accepted")
