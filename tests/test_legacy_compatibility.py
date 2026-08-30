def test_original_encoding_module_exports_working_decoder():
    from Implementation.encoding import PRIMITIVES, REPEAT, decode

    spec = decode([0] * 84)
    assert spec.channels == 16
    assert len(spec.flat_blocks) == 9
    assert len(PRIMITIVES) == 8
    assert REPEAT == [1, 2, 3, 4]


def test_original_dominance_contract():
    from Implementation.utils import Dominance

    assert Dominance([1, 1], [2, 2]) == 1
    assert Dominance([2, 2], [1, 1]) == -1
    assert Dominance([1, 2], [2, 1]) == 0


def test_original_nested_conversion_shape():
    from Implementation.encoding import convert

    branches = convert([0] * 81)
    assert len(branches) == 3
    assert all(len(branch) == 3 for branch in branches)
    assert all(len(unit) == 3 for branch in branches for unit in branch)


def test_original_problem_name_uses_repaired_contract():
    from Implementation.main import OptimizationProblem

    problem = OptimizationProblem(n_var=93, n_obj=2, objective_fn=lambda _: [1, 2])
    assert problem.genome_version == 2
    assert problem.n_var == 93
    assert problem.n_obj == 2
