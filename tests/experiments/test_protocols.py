import json
from dataclasses import asdict

import pytest

from bass.experiments.contracts import (
    load_hardware_profiles,
    load_protocol,
    protocol_digest,
)
from bass.experiments.results import validate_result
from bass.experiments.work_orders import (
    create_work_order,
    render_slurm,
    validate_work_order,
    work_order_digest,
)


def test_v2_and_v3_protocols_are_strict_dependency_ordered_contracts():
    v2 = load_protocol(2)
    v3 = load_protocol("3")

    assert v2.protocol_id == "bass-v2-gates-1.0"
    assert v3.protocol_id == "bass-v3-gates-1.0"
    assert len(v2.gates) == 14
    assert len(v3.gates) == 13
    assert v2.gates[-1].id == "V2-G13"
    assert v3.gates[-1].id == "V3-G12"
    assert len(protocol_digest(v2)) == 64
    assert len(protocol_digest(v3)) == 64


def test_every_gate_maps_to_a_qualifying_hardware_profile():
    profiles = load_hardware_profiles()
    for version in (2, 3):
        for gate in load_protocol(version).gates:
            assert gate.hardware in profiles
            assert profiles[gate.hardware].qualifying
            assert gate.outputs
            assert gate.pass_criteria


def test_work_order_preserves_protocol_and_smoke_can_never_qualify(monkeypatch):
    protocol = load_protocol(3)
    gate = protocol.gate("V3-G06")
    hardware = load_hardware_profiles()[gate.hardware]
    monkeypatch.setattr("bass.experiments.work_orders._git_revision", lambda: "a" * 40)
    monkeypatch.setattr("bass.experiments.work_orders._git_dirty", lambda: False)

    payload = create_work_order(
        protocol,
        gate,
        hardware,
        parameters={"target_device": "test-gpu"},
        smoke=True,
    )

    assert payload["state"] == "NOT_RUN"
    assert payload["smoke"] is True
    assert payload["qualifying"] is False
    assert payload["protocol_digest"] == protocol_digest(protocol)
    assert validate_work_order(payload) == []
    slurm = render_slurm(payload)
    assert "#SBATCH --gres=gpu:1" in slurm
    assert "V3-G06" in json.dumps(payload)


def test_dirty_tree_work_order_is_explicitly_nonqualifying(monkeypatch):
    protocol = load_protocol(2)
    gate = protocol.gate("V2-G03")
    hardware = load_hardware_profiles()[gate.hardware]
    monkeypatch.setattr("bass.experiments.work_orders._git_dirty", lambda: True)

    payload = create_work_order(protocol, gate, hardware)

    assert payload["source_dirty"] is True
    assert payload["qualifying"] is False
    assert validate_work_order(payload) == []


def test_result_validator_rejects_nonqualifying_pass():
    protocol = load_protocol(3)
    gate = protocol.gate("V3-G02")
    hardware = load_hardware_profiles()[gate.hardware]
    work_order = create_work_order(protocol, gate, hardware, smoke=True)
    result = {
        "schema_version": 1,
        "protocol_id": protocol.protocol_id,
        "protocol_digest": protocol_digest(protocol),
        "gate_id": gate.id,
        "source_revision": "b" * 40,
        "work_order_digest": work_order_digest(work_order),
        "status": "PASS",
        "qualifying": False,
        "smoke": True,
        "hardware_observed": asdict(load_hardware_profiles()[gate.hardware]),
        "command_expanded": "smoke-command",
        "environment_observed": {},
        "started_at": "2026-09-01T00:00:00Z",
        "finished_at": "2026-09-01T00:01:00Z",
        "criteria": [
            {"criterion": criterion, "passed": True} for criterion in gate.pass_criteria
        ],
        "artifacts": [],
        "deviations": [],
        "notes": "smoke only",
    }

    errors = validate_result(result, protocol, gate, work_order)
    assert any("smoke" in error for error in errors)
    assert any("qualifying" in error for error in errors)


def test_result_validator_accepts_complete_automatic_pass(monkeypatch):
    protocol = load_protocol(2)
    gate = protocol.gate("V2-G00")
    hardware = load_hardware_profiles()[gate.hardware]
    revision = "c" * 40
    monkeypatch.setattr("bass.experiments.work_orders._git_revision", lambda: revision)
    monkeypatch.setattr("bass.experiments.work_orders._git_dirty", lambda: False)
    work_order = create_work_order(protocol, gate, hardware)
    result = {
        "schema_version": 1,
        "protocol_id": protocol.protocol_id,
        "protocol_digest": protocol_digest(protocol),
        "gate_id": gate.id,
        "source_revision": revision,
        "work_order_digest": work_order_digest(work_order),
        "status": "PASS",
        "qualifying": True,
        "smoke": False,
        "hardware_observed": asdict(hardware),
        "command_expanded": "python -m pytest --junitxml=run/pytest-junit.xml",
        "environment_observed": {"python": "3.11"},
        "started_at": "2026-09-01T00:00:00Z",
        "finished_at": "2026-09-01T00:01:00Z",
        "criteria": [
            {"criterion": criterion, "passed": True} for criterion in gate.pass_criteria
        ],
        "artifacts": [
            {"name": name, "uri": name, "sha256": "d" * 64} for name in gate.outputs
        ],
        "deviations": [],
        "notes": "qualifying software gate",
    }

    assert validate_result(result, protocol, gate, work_order) == []


def test_invalid_version_is_rejected():
    with pytest.raises(ValueError, match="2 or 3"):
        load_protocol(1)
