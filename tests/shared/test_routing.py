import pytest

from bass.cli import build_parser
from bass.problem import BASSProblem
from bass.v1.problem import BASSProblem as V1Problem
from bass.v2.problem import BASSProblem as V2Problem
from bass.v3.problem import BASSProblem as V3Problem


def test_compatibility_problem_routes_to_explicit_version():
    assert isinstance(BASSProblem(genome_version=1), V1Problem)
    assert isinstance(BASSProblem(genome_version=2), V2Problem)
    assert isinstance(BASSProblem(genome_version=3), V3Problem)
    with pytest.raises(ValueError, match="1, 2, or 3"):
        BASSProblem(genome_version=4)


def test_cli_exposes_v3_explicitly():
    args = build_parser().parse_args(["--genome-version", "3"])
    assert args.genome_version == 3
    assert args.exchange_probability is None
