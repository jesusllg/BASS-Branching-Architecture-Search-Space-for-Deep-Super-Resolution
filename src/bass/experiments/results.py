"""Validate result envelopes without manufacturing scientific conclusions."""

from __future__ import annotations

from typing import Any

from .contracts import GateSpec, Protocol, protocol_digest
from .work_orders import work_order_digest

_STATUSES = frozenset({"PASS", "FAIL", "ERROR", "NOT_RUN"})


def validate_result(
    payload: dict[str, Any],
    protocol: Protocol,
    gate: GateSpec,
    work_order: dict[str, Any] | None = None,
) -> list[str]:
    errors: list[str] = []
    required = {
        "schema_version",
        "protocol_id",
        "protocol_digest",
        "gate_id",
        "source_revision",
        "work_order_digest",
        "status",
        "qualifying",
        "smoke",
        "hardware_observed",
        "command_expanded",
        "environment_observed",
        "started_at",
        "finished_at",
        "criteria",
        "artifacts",
        "deviations",
        "notes",
    }
    missing = required - set(payload)
    if missing:
        errors.append(f"missing fields: {sorted(missing)}")
        return errors
    if payload["schema_version"] != 1:
        errors.append("schema_version must be 1")
    if payload["protocol_id"] != protocol.protocol_id:
        errors.append("protocol_id does not match")
    if payload["protocol_digest"] != protocol_digest(protocol):
        errors.append("protocol_digest does not match the frozen protocol")
    if payload["gate_id"] != gate.id:
        errors.append("gate_id does not match")
    if work_order is None:
        errors.append("the originating work order is required for validation")
    else:
        if payload["work_order_digest"] != work_order_digest(work_order):
            errors.append("work_order_digest does not match")
        if payload["source_revision"] != work_order.get("source_revision"):
            errors.append("source_revision does not match the work order")
        if payload["protocol_digest"] != work_order.get("protocol_digest"):
            errors.append("protocol_digest does not match the work order")
        if payload["gate_id"] != work_order.get("gate_id"):
            errors.append("gate_id does not match the work order")
        if payload["qualifying"] and not work_order.get("qualifying"):
            errors.append(
                "a nonqualifying work order cannot produce a qualifying result"
            )
    if payload["status"] not in _STATUSES:
        errors.append(f"unknown status {payload['status']}")
    if payload["smoke"] and payload["qualifying"]:
        errors.append("smoke results cannot be qualifying")
    criteria = payload["criteria"]
    if not isinstance(criteria, list) or len(criteria) != len(gate.pass_criteria):
        errors.append("criteria must contain one disposition per frozen criterion")
    else:
        for index, (item, expected) in enumerate(zip(criteria, gate.pass_criteria)):
            if not isinstance(item, dict):
                errors.append(f"criterion {index} must be an object")
                continue
            if item.get("criterion") != expected:
                errors.append(f"criterion {index} text does not match the protocol")
            if item.get("passed") not in {True, False}:
                errors.append(f"criterion {index} must have a boolean passed value")
    if payload["status"] == "PASS":
        if payload["smoke"]:
            errors.append("PASS cannot come from a smoke run")
        if not payload["qualifying"]:
            errors.append("PASS requires a qualifying run")
        if isinstance(criteria, list) and any(
            item.get("passed") is not True
            for item in criteria
            if isinstance(item, dict)
        ):
            errors.append("PASS requires every frozen criterion to pass")
        if gate.decision == "review" and not payload.get("reviewed_by"):
            errors.append("review gates require reviewed_by before PASS")
        if "${" in payload["command_expanded"]:
            errors.append("PASS requires a fully expanded command")
        artifacts = payload["artifacts"]
        if not isinstance(artifacts, list):
            errors.append("artifacts must be a list")
        else:
            by_name = {
                item.get("name"): item
                for item in artifacts
                if isinstance(item, dict) and item.get("name")
            }
            missing_outputs = set(gate.outputs) - set(by_name)
            if missing_outputs:
                errors.append(
                    f"PASS is missing expected artifacts: {sorted(missing_outputs)}"
                )
            for name, item in by_name.items():
                digest = item.get("sha256", "")
                if len(digest) != 64:
                    errors.append(f"artifact {name} lacks a SHA-256 digest")
    return errors
