import numpy as np

from bass.shared.nsga3 import ReferenceDirectionEA
from bass.v3 import decode
from bass.v3.problem import BASSProblem


def make_problem():
    return BASSProblem(
        include_flops=False,
        objective_fn=lambda genome: [
            float(np.sum(genome)),
            float(np.count_nonzero(genome[-2:])),
        ],
    )


def test_v3_search_population_is_canonical_unique_and_12_integer():
    problem = make_problem()
    optimizer = ReferenceDirectionEA(problem, pop_size=12, n_gen=2, seed=17)
    population, _ = optimizer.run()

    assert population["X"].shape == (12, 12)
    specs = [decode(individual) for individual in population["X"]]
    hashes = [spec.canonical_hash() for spec in specs]
    assert len(set(hashes)) == len(hashes)


def test_v3_semantic_search_is_reproducible():
    first_optimizer = ReferenceDirectionEA(make_problem(), pop_size=8, n_gen=2, seed=23)
    second_optimizer = ReferenceDirectionEA(
        make_problem(), pop_size=8, n_gen=2, seed=23
    )
    first = first_optimizer.run()[0]
    second = second_optimizer.run()[0]
    np.testing.assert_array_equal(first["X"], second["X"])
    np.testing.assert_allclose(first["F"], second["F"])
    assert first_optimizer.history == second_optimizer.history
    assert all(
        "attempted_mutation_transitions" in record for record in first_optimizer.history
    )


def test_v3_mutation_treats_exchange_states_as_complete_genes():
    problem = make_problem()
    rng = np.random.default_rng(9)
    original = problem.sample_individual(rng)
    mutated = problem.mutate(original, rng, probability=1.0)

    assert mutated.shape == (12,)
    assert np.all(mutated[-2:] != original[-2:])
    assert np.array_equal(mutated, problem.canonicalize_individual(mutated))


def test_v3_problem_bounds_distinguish_units_and_exchanges():
    problem = make_problem()
    assert problem.n_var == 12
    assert problem.xu.tolist() == [3] + [42] * 9 + [2, 2]
