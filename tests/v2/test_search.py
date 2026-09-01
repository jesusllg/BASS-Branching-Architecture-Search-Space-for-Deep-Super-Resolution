import numpy as np

from bass.shared.nsga3 import NSGA3
from bass.v2 import decode, encode
from bass.v2.genotype import BlockGene, canonicalize_architecture
from bass.v2.problem import BASSProblem
from bass.v2.variation import mutate_block


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
    assert all(
        "attempted_mutation_transitions" in record for record in first_optimizer.history
    )


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


def test_crossover_recombines_the_unordered_six_branch_multiset():
    skip = BlockGene.skip()
    blocks = (
        BlockGene("cnn", "res_conv", 3, 1),
        BlockGene("cnn", "res_conv", 5, 1),
        BlockGene("cnn", "res_dilated_d2", 3, 1),
        BlockGene("attention", "channel_attention_residual", 0, 1),
        BlockGene("attention", "window_transformer", 4, 1),
        BlockGene("attention", "window_transformer", 8, 1),
    )
    left = canonicalize_architecture(
        16, tuple((block, skip, skip) for block in blocks[:3])
    )
    right = canonicalize_architecture(
        64, tuple((block, skip, skip) for block in blocks[3:])
    )
    pool = left.branches + right.branches
    expected = canonicalize_architecture(64, (pool[1], pool[5], pool[2]))

    child = make_problem().crossover(
        np.asarray(encode(left), dtype=np.int16),
        np.asarray(encode(right), dtype=np.int16),
        np.random.default_rng(0),
    )
    assert decode(child) == expected


def test_semantic_mutation_moves_are_local_and_auditable():
    original = BlockGene("attention", "window_transformer", 4, 2)
    rng = np.random.default_rng(44)
    observed = {}
    for _ in range(500):
        mutated, transition = mutate_block(original, rng)
        observed.setdefault(transition, mutated)

    assert set(observed) == {
        "repeat",
        "argument",
        "operation",
        "family_flip",
        "delete",
    }
    assert observed["repeat"].operation_key == original.operation_key
    assert observed["repeat"].repeat != original.repeat
    assert observed["argument"].family == original.family
    assert observed["argument"].op == original.op
    assert observed["argument"].arg != original.arg
    assert observed["operation"].family == original.family
    assert observed["operation"].op != original.op
    assert observed["family_flip"].family != original.family
    assert observed["delete"].is_skip
