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

from .contracts import GateSpec, HardwareProfile, Protocol, protocol_digest


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
) -> dict[str, Any]:
    parameters = dict(parameters or {})
    source_dirty = _git_dirty()
    return {
        "schema_version": 1,
        "state": "NOT_RUN",
        "protocol_id": protocol.protocol_id,
        "protocol_digest": protocol_digest(protocol),
        "gate_id": gate.id,
        "source_revision": _git_revision(),
        "source_dirty": source_dirty,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "smoke": bool(smoke),
        "qualifying": bool(hardware.qualifying and not smoke and not source_dirty),
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


def validate_work_order(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = {
        "schema_version",
        "state",
        "protocol_id",
        "protocol_digest",
        "gate_id",
        "source_revision",
        "source_dirty",
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
    if missing:
        errors.append(f"missing fields: {sorted(missing)}")
    if payload.get("state") != "NOT_RUN":
        errors.append("a prepared work order must have state=NOT_RUN")
    if payload.get("smoke") and payload.get("qualifying"):
        errors.append("a smoke work order can never be qualifying")
    if payload.get("source_dirty") and payload.get("qualifying"):
        errors.append("a dirty source tree can never produce a qualifying work order")
    return errors
