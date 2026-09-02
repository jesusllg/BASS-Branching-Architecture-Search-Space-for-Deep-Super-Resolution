"""Validate signed gate-ledger decisions without inventing experimental results."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from .contracts import Protocol, protocol_digest

_DISPOSITIONS = frozenset({"NOT_RUN", "PASS", "FAIL", "ERROR", "JUSTIFIED_SKIP"})
_DECISIONS = frozenset({"PENDING", "GO", "REVISE", "NO-GO"})


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _valid_timestamp(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None


def validate_gate_ledger(payload: dict[str, Any], protocol: Protocol) -> list[str]:
    """Validate execution dispositions separately from result-envelope status."""

    errors: list[str] = []
    required = {
        "schema_version",
        "protocol_id",
        "protocol_digest",
        "entries",
        "decision",
        "signed_by",
        "signed_at",
        "notes",
    }
    missing = required - set(payload)
    if missing:
        return [f"missing fields: {sorted(missing)}"]
    extra = set(payload) - required
    if extra:
        errors.append(f"unexpected fields: {sorted(extra)}")
    if payload["schema_version"] != 1:
        errors.append("schema_version must be 1")
    if payload["protocol_id"] != protocol.protocol_id:
        errors.append("protocol_id does not match")
    if payload["protocol_digest"] != protocol_digest(protocol):
        errors.append("protocol_digest does not match")
    if (
        not isinstance(payload["decision"], str)
        or payload["decision"] not in _DECISIONS
    ):
        errors.append(f"unknown ledger decision {payload['decision']}")
    if not isinstance(payload["notes"], str):
        errors.append("notes must be a string")

    entries = payload["entries"]
    if not isinstance(entries, list):
        return errors + ["entries must be a list"]
    by_gate: dict[str, dict[str, Any]] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            errors.append(f"ledger entry {index} must be an object")
            continue
        if set(entry) != {
            "gate_id",
            "disposition",
            "result_digest",
            "justification",
        }:
            errors.append(f"ledger entry {index} has incorrect fields")
            continue
        gate_id = entry["gate_id"]
        if not isinstance(gate_id, str) or not gate_id:
            errors.append(f"ledger entry {index} requires a non-empty gate_id")
            continue
        if gate_id in by_gate:
            errors.append(f"duplicate ledger entry for {gate_id}")
            continue
        by_gate[gate_id] = entry

    expected_ids = {gate.id for gate in protocol.gates}
    missing_gates = expected_ids - set(by_gate)
    extra_gates = set(by_gate) - expected_ids
    if missing_gates:
        errors.append(f"ledger is missing gates: {sorted(missing_gates)}")
    if extra_gates:
        errors.append(f"ledger contains unknown gates: {sorted(extra_gates)}")

    for gate in protocol.gates:
        entry = by_gate.get(gate.id)
        if entry is None:
            continue
        disposition = entry["disposition"]
        if not isinstance(disposition, str) or disposition not in _DISPOSITIONS:
            errors.append(f"unknown disposition {disposition} for {gate.id}")
            continue
        result_digest = entry["result_digest"]
        justification = entry["justification"]
        if disposition in {"PASS", "FAIL", "ERROR"}:
            if not _is_sha256(result_digest):
                errors.append(f"{gate.id} {disposition} requires a result digest")
        elif result_digest is not None:
            errors.append(f"{gate.id} {disposition} cannot claim a result digest")
        if disposition == "JUSTIFIED_SKIP":
            if gate.decision != "conditional":
                errors.append(
                    f"only a conditional gate may be JUSTIFIED_SKIP: {gate.id}"
                )
            if not isinstance(justification, str) or len(justification.strip()) < 20:
                errors.append(f"{gate.id} JUSTIFIED_SKIP requires a justification")
        elif not isinstance(justification, str):
            errors.append(f"{gate.id} justification must be a string")

    if payload["decision"] == "GO":
        for gate in protocol.gates:
            disposition = by_gate.get(gate.id, {}).get("disposition")
            allowed = (
                {"PASS", "JUSTIFIED_SKIP"}
                if gate.decision == "conditional"
                else {"PASS"}
            )
            if disposition not in allowed:
                errors.append(f"GO requires an acceptable disposition for {gate.id}")

    if payload["decision"] == "PENDING":
        if payload["signed_by"] is not None or payload["signed_at"] is not None:
            errors.append("a PENDING ledger must remain unsigned")
    else:
        if (
            not isinstance(payload["signed_by"], str)
            or not payload["signed_by"].strip()
        ):
            errors.append("a final ledger decision requires signed_by")
        if not _valid_timestamp(payload["signed_at"]):
            errors.append("a final ledger decision requires a timezone-aware signed_at")
    return errors
