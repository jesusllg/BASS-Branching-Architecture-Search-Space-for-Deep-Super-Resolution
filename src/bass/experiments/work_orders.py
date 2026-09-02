"""Create immutable, auditable work orders for external hardware runs."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .contracts import (
    GateSpec,
    HardwareProfile,
    Protocol,
    protocol_digest,
    runtime_tree_digest,
)


def _is_hex_digest(value: object, lengths: set[int]) -> bool:
    return (
        isinstance(value, str)
        and len(value) in lengths
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _is_timezone_timestamp(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None


def _git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _git_dirty() -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        check=False,
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip()) if result.returncode == 0 else True


def create_work_order(
    protocol: Protocol,
    gate: GateSpec,
    hardware: HardwareProfile,
    *,
    parameters: dict[str, str] | None = None,
    smoke: bool = False,
    repository_root: Path | None = None,
) -> dict[str, Any]:
    parameters = dict(parameters or {})
    source_dirty = _git_dirty()
    current_runtime_digest = runtime_tree_digest(
        protocol.runtime_contract.paths, root=repository_root
    )
    runtime_contract_matches = (
        current_runtime_digest == protocol.runtime_contract.tree_sha256
    )
    return {
        "schema_version": 1,
        "state": "NOT_RUN",
        "protocol_id": protocol.protocol_id,
        "protocol_digest": protocol_digest(protocol),
        "gate_id": gate.id,
        "source_revision": _git_revision(),
        "source_dirty": source_dirty,
        "runtime_tree_sha256": current_runtime_digest,
        "runtime_contract_matches": runtime_contract_matches,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "smoke": bool(smoke),
        "qualifying": bool(
            hardware.qualifying
            and not smoke
            and not source_dirty
            and runtime_contract_matches
        ),
        "hardware": asdict(hardware),
        "parameters": parameters,
        "command_template": gate.command,
        "expected_outputs": list(gate.outputs),
        "pass_criteria": list(gate.pass_criteria),
        "decision": gate.decision,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
        },
    }


def write_work_order(payload: dict[str, Any], destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    destination.with_suffix(destination.suffix + ".sha256").write_text(
        work_order_digest(payload) + "\n", encoding="utf-8"
    )
    return destination


def work_order_digest(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def render_slurm(work_order: dict[str, Any]) -> str:
    hardware = work_order["hardware"]
    accelerator = hardware["accelerator"]
    gpu_line = ""
    if accelerator != "cpu":
        gpu_line = f"#SBATCH --gres=gpu:{hardware['gpus']}\n"
    return (
        "#!/usr/bin/env bash\n"
        f"#SBATCH --job-name={work_order['gate_id'].lower()}\n"
        f"#SBATCH --cpus-per-task={hardware['cpus']}\n"
        f"#SBATCH --mem={hardware['memory_gib']}G\n"
        f"#SBATCH --time={hardware['walltime']}\n"
        f"{gpu_line}"
        "set -euo pipefail\n\n"
        "WORK_ORDER=${1:?pass the work-order.json path}\n"
        'python scripts/run_experimental_gate.py validate-work-order "$WORK_ORDER"\n'
        f"# Protocol command template:\n# {work_order['command_template']}\n"
    )


def validate_work_order(
    payload: dict[str, Any],
    protocol: Protocol | None = None,
    gate: GateSpec | None = None,
    hardware: HardwareProfile | None = None,
) -> list[str]:
    errors: list[str] = []
    required = {
        "schema_version",
        "state",
        "protocol_id",
        "protocol_digest",
        "gate_id",
        "source_revision",
        "source_dirty",
        "runtime_tree_sha256",
        "runtime_contract_matches",
        "created_at",
        "smoke",
        "qualifying",
        "hardware",
        "parameters",
        "command_template",
        "expected_outputs",
        "pass_criteria",
        "decision",
        "runtime",
    }
    missing = required - set(payload)
    extra = set(payload) - required
    if missing:
        errors.append(f"missing fields: {sorted(missing)}")
    if extra:
        errors.append(f"unexpected fields: {sorted(extra)}")
    if payload.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if payload.get("state") != "NOT_RUN":
        errors.append("a prepared work order must have state=NOT_RUN")
    for field in (
        "source_dirty",
        "runtime_contract_matches",
        "smoke",
        "qualifying",
    ):
        if type(payload.get(field)) is not bool:
            errors.append(f"{field} must be a boolean")
    if not _is_hex_digest(payload.get("source_revision"), {40, 64}):
        errors.append("source_revision must be a full Git OID")
    if not _is_hex_digest(payload.get("protocol_digest"), {64}):
        errors.append("protocol_digest must be a SHA-256 digest")
    if not _is_hex_digest(payload.get("runtime_tree_sha256"), {64}):
        errors.append("runtime_tree_sha256 must be a SHA-256 digest")
    if not _is_timezone_timestamp(payload.get("created_at")):
        errors.append("created_at must be a timezone-aware ISO-8601 timestamp")
    if not isinstance(payload.get("hardware"), dict):
        errors.append("hardware must be an object")
    parameters = payload.get("parameters")
    if not isinstance(parameters, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in parameters.items()
    ):
        errors.append("parameters must map strings to strings")
    if not isinstance(payload.get("runtime"), dict):
        errors.append("runtime must be an object")
    for field in ("expected_outputs", "pass_criteria"):
        value = payload.get(field)
        if not isinstance(value, list) or any(
            not isinstance(item, str) or not item for item in value
        ):
            errors.append(f"{field} must contain non-empty strings")
    for field in ("protocol_id", "gate_id", "command_template", "decision"):
        if not isinstance(payload.get(field), str) or not payload[field]:
            errors.append(f"{field} must be a non-empty string")
    if payload.get("smoke") and payload.get("qualifying"):
        errors.append("a smoke work order can never be qualifying")
    if payload.get("source_dirty") and payload.get("qualifying"):
        errors.append("a dirty source tree can never produce a qualifying work order")
    if not payload.get("runtime_contract_matches") and payload.get("qualifying"):
        errors.append("a runtime-contract mismatch can never qualify")
    if protocol is not None:
        if payload.get("protocol_id") != protocol.protocol_id:
            errors.append("protocol_id does not match the selected protocol")
        if payload.get("protocol_digest") != protocol_digest(protocol):
            errors.append("protocol_digest does not match the selected protocol")
        if payload.get("runtime_tree_sha256") != protocol.runtime_contract.tree_sha256:
            errors.append("runtime tree does not match the frozen runtime contract")
        if payload.get("runtime_contract_matches") is not True:
            errors.append("runtime_contract_matches must be true")
    if gate is not None:
        if payload.get("gate_id") != gate.id:
            errors.append("gate_id does not match the selected gate")
        if payload.get("command_template") != gate.command:
            errors.append("command_template does not match the selected gate")
        if payload.get("expected_outputs") != list(gate.outputs):
            errors.append("expected_outputs do not match the selected gate")
        if payload.get("pass_criteria") != list(gate.pass_criteria):
            errors.append("pass_criteria do not match the selected gate")
        if payload.get("decision") != gate.decision:
            errors.append("decision mode does not match the selected gate")
    if hardware is not None:
        if payload.get("hardware") != asdict(hardware):
            errors.append(
                "work-order hardware does not match the gate hardware profile"
            )
        expected_qualification = bool(
            hardware.qualifying
            and not payload.get("smoke")
            and not payload.get("source_dirty")
            and payload.get("runtime_contract_matches") is True
        )
        if payload.get("qualifying") is not expected_qualification:
            errors.append("qualifying does not match the frozen work-order state")
    return errors
