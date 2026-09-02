"""Validate result envelopes without manufacturing scientific conclusions."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .contracts import (
    GateSpec,
    Protocol,
    load_hardware_profiles,
    protocol_digest,
)
from .work_orders import validate_work_order, work_order_digest

_STATUSES = frozenset({"PASS", "FAIL", "ERROR", "NOT_RUN"})


def validate_result_schema(
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
        "runtime_tree_sha256",
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
    for field in ("qualifying", "smoke"):
        if type(payload[field]) is not bool:
            errors.append(f"{field} must be a boolean")
    for field in (
        "protocol_id",
        "gate_id",
        "source_revision",
        "runtime_tree_sha256",
        "work_order_digest",
        "status",
        "command_expanded",
        "started_at",
        "finished_at",
        "notes",
    ):
        if not isinstance(payload[field], str):
            errors.append(f"{field} must be a string")
    if not _is_git_oid(payload["source_revision"]):
        errors.append("source_revision must be a full Git OID")
    for field in ("protocol_digest", "runtime_tree_sha256", "work_order_digest"):
        if not _is_sha256(payload[field]):
            errors.append(f"{field} must be a SHA-256 digest")
    if not isinstance(payload["hardware_observed"], dict):
        errors.append("hardware_observed must be an object")
    if not isinstance(payload["environment_observed"], dict):
        errors.append("environment_observed must be an object")
    if not isinstance(payload["deviations"], list):
        errors.append("deviations must be a list")
    if payload["protocol_id"] != protocol.protocol_id:
        errors.append("protocol_id does not match")
    if payload["protocol_digest"] != protocol_digest(protocol):
        errors.append("protocol_digest does not match the frozen protocol")
    if payload["gate_id"] != gate.id:
        errors.append("gate_id does not match")
    if work_order is None:
        errors.append("the originating work order is required for validation")
    else:
        hardware = load_hardware_profiles()[gate.hardware]
        for error in validate_work_order(
            work_order, protocol=protocol, gate=gate, hardware=hardware
        ):
            errors.append(f"invalid work order: {error}")
        if payload["work_order_digest"] != work_order_digest(work_order):
            errors.append("work_order_digest does not match")
        if payload["source_revision"] != work_order.get("source_revision"):
            errors.append("source_revision does not match the work order")
        if payload["runtime_tree_sha256"] != work_order.get("runtime_tree_sha256"):
            errors.append("runtime_tree_sha256 does not match the work order")
        if payload["protocol_digest"] != work_order.get("protocol_digest"):
            errors.append("protocol_digest does not match the work order")
        if payload["gate_id"] != work_order.get("gate_id"):
            errors.append("gate_id does not match the work order")
        if payload["qualifying"] and not work_order.get("qualifying"):
            errors.append(
                "a nonqualifying work order cannot produce a qualifying result"
            )
    if not isinstance(payload["status"], str) or payload["status"] not in _STATUSES:
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
            if type(item.get("passed")) is not bool:
                errors.append(f"criterion {index} must have a boolean passed value")
    artifacts = payload["artifacts"]
    artifacts_by_name: dict[str, dict[str, Any]] = {}
    if not isinstance(artifacts, list):
        errors.append("artifacts must be a list")
    else:
        for index, item in enumerate(artifacts):
            if not isinstance(item, dict):
                errors.append(f"artifact {index} must be an object")
                continue
            name = item.get("name")
            uri = item.get("uri")
            digest = item.get("sha256")
            if not isinstance(name, str) or not name:
                errors.append(f"artifact {index} requires a non-empty name")
                continue
            if name in artifacts_by_name:
                errors.append(f"duplicate artifact name: {name}")
            else:
                artifacts_by_name[name] = item
            if not isinstance(uri, str) or not uri:
                errors.append(f"artifact {name} requires a non-empty URI")
            if not _is_sha256(digest):
                errors.append(f"artifact {name} lacks a SHA-256 digest")

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
        if gate.decision == "review" and (
            not isinstance(payload.get("reviewed_by"), str)
            or not payload["reviewed_by"].strip()
        ):
            errors.append("review gates require reviewed_by before PASS")
        if (
            not isinstance(payload["command_expanded"], str)
            or "${" in payload["command_expanded"]
        ):
            errors.append("PASS requires a fully expanded command")
        missing_outputs = set(gate.outputs) - set(artifacts_by_name)
        if missing_outputs:
            errors.append(
                f"PASS is missing expected artifacts: {sorted(missing_outputs)}"
            )
    return errors


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _is_git_oid(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) in {40, 64}
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _parse_timestamp(value: object, field: str, errors: list[str]) -> datetime | None:
    if not isinstance(value, str):
        errors.append(f"{field} must be an ISO-8601 timestamp")
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        errors.append(f"{field} must be an ISO-8601 timestamp")
        return None
    if parsed.tzinfo is None:
        errors.append(f"{field} must include a timezone")
        return None
    return parsed


def _verify_hardware(payload: dict[str, Any], gate: GateSpec) -> list[str]:
    errors: list[str] = []
    observed = payload.get("hardware_observed")
    expected = load_hardware_profiles()[gate.hardware]
    if not isinstance(observed, dict):
        return ["hardware_observed must be an object"]
    if observed.get("profile_id") != expected.id:
        errors.append("observed hardware profile_id does not match the gate")
    if observed.get("accelerator") != expected.accelerator:
        errors.append("observed accelerator does not match the gate profile")
    for field, minimum in (
        ("gpus", expected.gpus),
        ("cpus", expected.cpus),
        ("memory_gib", expected.memory_gib),
        ("vram_gib_per_gpu", expected.minimum_vram_gib),
    ):
        value = observed.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            errors.append(f"hardware_observed.{field} must be finite numeric")
        elif value < minimum:
            errors.append(
                f"hardware_observed.{field}={value} is below required {minimum}"
            )
    if expected.accelerator != "cpu" and (
        not isinstance(observed.get("device_name"), str)
        or not observed["device_name"].strip()
    ):
        errors.append("accelerator evidence requires hardware_observed.device_name")
    return errors


def _local_artifact_path(uri: str, artifact_root: Path) -> Path | None:
    parsed = urlparse(uri)
    if parsed.scheme not in {"", "file"}:
        return None
    if parsed.query or parsed.fragment or (parsed.scheme == "file" and parsed.netloc):
        raise ValueError("local artifact URI contains unsupported URI components")
    raw = Path(parsed.path if parsed.scheme == "file" else uri)
    candidate = raw if raw.is_absolute() else artifact_root / raw
    resolved = candidate.resolve()
    root = artifact_root.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("artifact URI escapes the declared artifact root")
    return resolved


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_result_evidence(
    payload: dict[str, Any],
    protocol: Protocol,
    gate: GateSpec,
    work_order: dict[str, Any],
    *,
    artifact_root: Path,
    artifact_resolver: Callable[[str], bytes] | None = None,
) -> list[str]:
    """Verify the result's work order, hardware, timestamps, and artifact bytes."""

    errors = validate_result_schema(payload, protocol, gate, work_order)
    started = _parse_timestamp(payload.get("started_at"), "started_at", errors)
    finished = _parse_timestamp(payload.get("finished_at"), "finished_at", errors)
    if started is not None and finished is not None and finished < started:
        errors.append("finished_at precedes started_at")
    errors.extend(_verify_hardware(payload, gate))

    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list):
        return errors
    for item in artifacts:
        if not isinstance(item, dict):
            errors.append("every artifact must be an object")
            continue
        name = item.get("name", "<unnamed>")
        uri = item.get("uri")
        expected_digest = item.get("sha256")
        if not isinstance(uri, str) or not uri:
            errors.append(f"artifact {name} lacks a URI")
            continue
        if not _is_sha256(expected_digest):
            errors.append(f"artifact {name} lacks a valid SHA-256 digest")
            continue
        try:
            path = _local_artifact_path(uri, artifact_root)
        except ValueError as error:
            errors.append(f"artifact {name}: {error}")
            continue
        if path is not None:
            if not path.is_file():
                errors.append(f"artifact {name} does not exist: {uri}")
                continue
            observed_digest = _file_sha256(path)
        elif artifact_resolver is not None:
            try:
                observed_digest = hashlib.sha256(artifact_resolver(uri)).hexdigest()
            except Exception as error:  # noqa: BLE001 - external adapter boundary
                errors.append(f"artifact {name} could not be resolved: {error}")
                continue
        else:
            errors.append(
                f"artifact {name} uses a remote URI without an artifact resolver"
            )
            continue
        if observed_digest != expected_digest:
            errors.append(f"artifact {name} SHA-256 does not match its bytes")
    return errors


# Backward-compatible name: callers that only need manifest validation keep the
# previous API, while publication authorization must call verify_result_evidence.
validate_result = validate_result_schema
