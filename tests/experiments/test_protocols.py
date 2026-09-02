import json
from hashlib import sha256

import pytest

from bass.experiments.contracts import (
    load_hardware_profiles,
    load_protocol,
    protocol_digest,
    runtime_tree_digest,
)
from bass.experiments.ledger import validate_gate_ledger
from bass.experiments.results import validate_result_schema, verify_result_evidence
from bass.experiments.work_orders import (
    create_work_order,
    render_slurm,
    validate_work_order,
    work_order_digest,
)


def _observed_hardware(hardware):
    return {
        "profile_id": hardware.id,
        "accelerator": hardware.accelerator,
        "device_name": "test-device",
        "gpus": hardware.gpus,
        "vram_gib_per_gpu": hardware.minimum_vram_gib,
        "cpus": hardware.cpus,
        "memory_gib": hardware.memory_gib,
    }


def test_v2_and_v3_protocols_are_strict_dependency_ordered_contracts():
    v2 = load_protocol(2)
    v3 = load_protocol("3")

    assert v2.protocol_id == "bass-v2-gates-1.1"
    assert v3.protocol_id == "bass-v3-gates-1.1"
    assert len(v2.gates) == 14
    assert len(v3.gates) == 14
    assert v2.gates[-1].id == "V2-G13"
    assert v3.gates[-1].id == "V3-G13"
    assert v3.gate("V3-G03").title == "Stage-aware canonical equivalence gate"
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
    assert payload["runtime_tree_sha256"] == protocol.runtime_contract.tree_sha256
    assert payload["runtime_contract_matches"] is True
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
        "source_revision": work_order["source_revision"],
        "runtime_tree_sha256": work_order["runtime_tree_sha256"],
        "work_order_digest": work_order_digest(work_order),
        "status": "PASS",
        "qualifying": False,
        "smoke": True,
        "hardware_observed": _observed_hardware(hardware),
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

    errors = validate_result_schema(result, protocol, gate, work_order)
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
        "runtime_tree_sha256": work_order["runtime_tree_sha256"],
        "work_order_digest": work_order_digest(work_order),
        "status": "PASS",
        "qualifying": True,
        "smoke": False,
        "hardware_observed": _observed_hardware(hardware),
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

    assert validate_result_schema(result, protocol, gate, work_order) == []

    result["qualifying"] = "yes"
    result["artifacts"].append(dict(result["artifacts"][0]))
    errors = validate_result_schema(result, protocol, gate, work_order)
    assert "qualifying must be a boolean" in errors
    assert any("duplicate artifact name" in error for error in errors)


def test_runtime_contract_is_content_bound():
    protocol = load_protocol(3)
    assert protocol.runtime_contract.audit_base_revision == (
        "d1b624feb98d28caaf439d0d2ee9d919fede2516"
    )
    assert runtime_tree_digest(protocol.runtime_contract.paths) == (
        protocol.runtime_contract.tree_sha256
    )


def test_evidence_verifier_recomputes_artifact_bytes(monkeypatch, tmp_path):
    protocol = load_protocol(2)
    gate = protocol.gate("V2-G00")
    hardware = load_hardware_profiles()[gate.hardware]
    revision = "e" * 40
    monkeypatch.setattr("bass.experiments.work_orders._git_revision", lambda: revision)
    monkeypatch.setattr("bass.experiments.work_orders._git_dirty", lambda: False)
    work_order = create_work_order(protocol, gate, hardware)
    artifact = tmp_path / gate.outputs[0]
    artifact.write_bytes(b"real evidence")
    result = {
        "schema_version": 1,
        "protocol_id": protocol.protocol_id,
        "protocol_digest": protocol_digest(protocol),
        "gate_id": gate.id,
        "source_revision": revision,
        "runtime_tree_sha256": work_order["runtime_tree_sha256"],
        "work_order_digest": work_order_digest(work_order),
        "status": "PASS",
        "qualifying": True,
        "smoke": False,
        "hardware_observed": _observed_hardware(hardware),
        "command_expanded": "python -m pytest --junitxml=pytest-junit.xml",
        "environment_observed": {"python": "3.11"},
        "started_at": "2026-09-01T00:00:00Z",
        "finished_at": "2026-09-01T00:01:00Z",
        "criteria": [
            {"criterion": criterion, "passed": True} for criterion in gate.pass_criteria
        ],
        "artifacts": [
            {
                "name": gate.outputs[0],
                "uri": artifact.name,
                "sha256": sha256(artifact.read_bytes()).hexdigest(),
            }
        ],
        "deviations": [],
        "notes": "verified fixture",
    }

    assert (
        verify_result_evidence(
            result, protocol, gate, work_order, artifact_root=tmp_path
        )
        == []
    )
    result["artifacts"][0]["sha256"] = "f" * 64
    errors = verify_result_evidence(
        result, protocol, gate, work_order, artifact_root=tmp_path
    )
    assert any("does not match its bytes" in error for error in errors)

    result["artifacts"][0]["sha256"] = sha256(artifact.read_bytes()).hexdigest()
    result["finished_at"] = "2025-12-31T23:59:59Z"
    result["hardware_observed"]["cpus"] = hardware.cpus - 1
    errors = verify_result_evidence(
        result, protocol, gate, work_order, artifact_root=tmp_path
    )
    assert "finished_at precedes started_at" in errors
    assert any("hardware_observed.cpus" in error for error in errors)

    result["finished_at"] = "2026-09-01T00:01:00Z"
    result["hardware_observed"]["cpus"] = hardware.cpus
    result["artifacts"][0]["uri"] = "../outside.json"
    errors = verify_result_evidence(
        result, protocol, gate, work_order, artifact_root=tmp_path
    )
    assert any("escapes the declared artifact root" in error for error in errors)


def test_justified_skip_exists_only_in_the_signed_ledger():
    protocol = load_protocol(2)
    conditional = protocol.gate("V2-G09")
    entries = [
        {
            "gate_id": gate.id,
            "disposition": "PASS",
            "result_digest": "a" * 64,
            "justification": "",
        }
        for gate in protocol.gates
    ]
    skipped = next(entry for entry in entries if entry["gate_id"] == conditional.id)
    skipped.update(
        disposition="JUSTIFIED_SKIP",
        result_digest=None,
        justification="No aggregate proxy was preregistered after calibration.",
    )
    ledger = {
        "schema_version": 1,
        "protocol_id": protocol.protocol_id,
        "protocol_digest": protocol_digest(protocol),
        "entries": entries,
        "decision": "GO",
        "signed_by": "independent-reviewer",
        "signed_at": "2026-09-01T12:00:00Z",
        "notes": "fixture",
    }

    assert validate_gate_ledger(ledger, protocol) == []
    entries[0]["disposition"] = "JUSTIFIED_SKIP"
    entries[0]["result_digest"] = None
    entries[0]["justification"] = "This mandatory gate was incorrectly skipped."
    errors = validate_gate_ledger(ledger, protocol)
    assert any("only a conditional gate" in error for error in errors)

    entries[0]["gate_id"] = []
    errors = validate_gate_ledger(ledger, protocol)
    assert any("requires a non-empty gate_id" in error for error in errors)


def test_invalid_version_is_rejected():
    with pytest.raises(ValueError, match="2 or 3"):
        load_protocol(1)
