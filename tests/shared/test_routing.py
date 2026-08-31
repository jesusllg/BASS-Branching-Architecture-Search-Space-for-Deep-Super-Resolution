import pytest

from bass.problem import BASSProblem
from bass.v1.problem import BASSProblem as V1Problem
from bass.v2.problem import BASSProblem as V2Problem


def test_compatibility_problem_routes_to_explicit_version():
    assert isinstance(BASSProblem(genome_version=1), V1Problem)
    assert isinstance(BASSProblem(genome_version=2), V2Problem)
    with pytest.raises(ValueError, match="1 or 2"):
        BASSProblem(genome_version=3)
