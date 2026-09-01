import pytest

from bass.v2.evaluation import synflow_metric_nas


def test_v2_retired_synflow_alias_cannot_silently_run_gradient_flow():
    with pytest.raises(RuntimeError, match="not canonical SynFlow"):
        synflow_metric_nas(None)
