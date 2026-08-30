import pytest

tf = pytest.importorskip("tensorflow")

from bass.encoding import decode_v1_bits
from bass.evaluation import evaluate_architecture


def test_gradient_flow_evaluation_is_seeded_and_reproducible():
    architecture = decode_v1_bits([0] * 84)
    first = evaluate_architecture(
        architecture,
        input_shape=(4, 4, 3),
        include_flops=False,
        evaluation_seed=19,
    )
    second = evaluate_architecture(
        architecture,
        input_shape=(4, 4, 3),
        include_flops=False,
        evaluation_seed=19,
    )
    assert first.params == second.params
    assert first.score == pytest.approx(second.score, rel=0.0, abs=0.0)
