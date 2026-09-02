"""Strict loaders for the machine-readable BASS gate protocols."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

_PROTOCOL_PACKAGE = "bass.experiments.protocols"
_DECISIONS = frozenset({"automatic", "review", "conditional"})


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


def runtime_tree_digest(paths: tuple[str, ...], *, root: Path | None = None) -> str:
    """Hash the versioned runtime files independently of the enclosing Git commit."""

    base = (Path.cwd() if root is None else Path(root)).resolve()
    selected: list[Path] = []
    for item in paths:
        relative = Path(item)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Runtime path must stay inside the repository: {item}")
        candidate = (base / relative).resolve()
        if not candidate.is_relative_to(base):
            raise ValueError(f"Runtime path escapes the repository: {item}")
        if (base / relative).is_symlink():
            raise ValueError(f"Runtime contract paths cannot be symlinks: {item}")
        if candidate.is_file():
            selected.append(candidate)
        elif candidate.is_dir():
            for path in candidate.rglob("*"):
                if path.is_symlink():
                    raise ValueError(
                        f"Runtime contract trees cannot contain symlinks: {path}"
                    )
                if (
                    path.is_file()
                    and "__pycache__" not in path.parts
                    and path.suffix not in {".pyc", ".pyo"}
                ):
                    selected.append(path)
        else:
            raise FileNotFoundError(f"Runtime contract path does not exist: {item}")

    if not selected:
        raise ValueError("Runtime contract must select at least one file")

    digest = hashlib.sha256()
    for path in sorted(set(selected)):
        relative = path.relative_to(base).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _read_json(name: str) -> dict[str, Any]:
    resource = files(_PROTOCOL_PACKAGE).joinpath(name)
    return json.loads(resource.read_text(encoding="utf-8"))


def _require_keys(payload: dict[str, Any], expected: set[str], context: str) -> None:
    missing = expected - set(payload)
    extra = set(payload) - expected
    if missing or extra:
        raise ValueError(
            f"{context} fields mismatch; missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )


@dataclass(frozen=True, slots=True)
class HardwareProfile:
    id: str
    purpose: str
    accelerator: str
    gpus: int
    minimum_vram_gib: int
    cpus: int
    memory_gib: int
    walltime: str
    qualifying: bool
    notes: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> HardwareProfile:
        _require_keys(
            payload,
            {
                "id",
                "purpose",
                "accelerator",
                "gpus",
                "minimum_vram_gib",
                "cpus",
                "memory_gib",
                "walltime",
                "qualifying",
                "notes",
            },
            "hardware profile",
        )
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class GateSpec:
    id: str
    title: str
    phase: str
    hardware: str
    depends_on: tuple[str, ...]
    cohort: dict[str, Any]
    command: str
    outputs: tuple[str, ...]
    pass_criteria: tuple[str, ...]
    decision: str
    rationale: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> GateSpec:
        _require_keys(
            payload,
            {
                "id",
                "title",
                "phase",
                "hardware",
                "depends_on",
                "cohort",
                "command",
                "outputs",
                "pass_criteria",
                "decision",
                "rationale",
            },
            "gate",
        )
        if payload["decision"] not in _DECISIONS:
            raise ValueError(f"Unknown decision mode: {payload['decision']}")
        return cls(
            id=payload["id"],
            title=payload["title"],
            phase=payload["phase"],
            hardware=payload["hardware"],
            depends_on=tuple(payload["depends_on"]),
            cohort=dict(payload["cohort"]),
            command=payload["command"],
            outputs=tuple(payload["outputs"]),
            pass_criteria=tuple(payload["pass_criteria"]),
            decision=payload["decision"],
            rationale=payload["rationale"],
        )


@dataclass(frozen=True, slots=True)
class RuntimeContract:
    audit_base_revision: str
    paths: tuple[str, ...]
    tree_sha256: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RuntimeContract:
        _require_keys(
            payload,
            {"audit_base_revision", "paths", "tree_sha256"},
            "runtime contract",
        )
        if not _is_git_oid(payload["audit_base_revision"]):
            raise ValueError("runtime audit_base_revision must be a full Git OID")
        if not isinstance(payload["paths"], list):
            raise TypeError("runtime paths must be a JSON array")
        paths = tuple(payload["paths"])
        if not paths or any(not isinstance(path, str) or not path for path in paths):
            raise ValueError("runtime paths must contain non-empty strings")
        if len(set(paths)) != len(paths):
            raise ValueError("runtime paths must be unique")
        if not _is_sha256(payload["tree_sha256"]):
            raise ValueError("runtime tree_sha256 must be a lowercase SHA-256")
        return cls(
            audit_base_revision=payload["audit_base_revision"],
            paths=paths,
            tree_sha256=payload["tree_sha256"].lower(),
        )


@dataclass(frozen=True, slots=True)
class Protocol:
    schema_version: int
    protocol_id: str
    bass_version: int
    status: str
    runtime_contract: RuntimeContract
    qualification_rule: str
    gates: tuple[GateSpec, ...]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Protocol:
        _require_keys(
            payload,
            {
                "schema_version",
                "protocol_id",
                "bass_version",
                "status",
                "runtime_contract",
                "qualification_rule",
                "gates",
            },
            "protocol",
        )
        gates = tuple(GateSpec.from_dict(item) for item in payload["gates"])
        protocol = cls(
            schema_version=payload["schema_version"],
            protocol_id=payload["protocol_id"],
            bass_version=payload["bass_version"],
            status=payload["status"],
            runtime_contract=RuntimeContract.from_dict(payload["runtime_contract"]),
            qualification_rule=payload["qualification_rule"],
            gates=gates,
        )
        protocol.validate()
        return protocol

    def validate(self) -> None:
        if self.schema_version != 2:
            raise ValueError("Only experimental protocol schema_version=2 is supported")
        if self.bass_version not in {2, 3}:
            raise ValueError("Experimental protocols exist only for BASS V2 and V3")
        hardware = load_hardware_profiles()
        seen: set[str] = set()
        for gate in self.gates:
            if gate.id in seen:
                raise ValueError(f"Duplicate gate ID: {gate.id}")
            if not gate.id.startswith(f"V{self.bass_version}-G"):
                raise ValueError(
                    f"Gate {gate.id} does not belong to V{self.bass_version}"
                )
            unknown_dependencies = set(gate.depends_on) - seen
            if unknown_dependencies:
                raise ValueError(
                    f"Gate {gate.id} has unresolved or forward dependencies: "
                    f"{sorted(unknown_dependencies)}"
                )
            if gate.hardware not in hardware:
                raise ValueError(
                    f"Gate {gate.id} uses unknown hardware profile {gate.hardware}"
                )
            if not gate.outputs or not gate.pass_criteria:
                raise ValueError(
                    f"Gate {gate.id} must define outputs and pass criteria"
                )
            seen.add(gate.id)

    def gate(self, gate_id: str) -> GateSpec:
        for gate in self.gates:
            if gate.id == gate_id:
                return gate
        raise KeyError(f"Unknown gate {gate_id} in {self.protocol_id}")


def load_hardware_profiles() -> dict[str, HardwareProfile]:
    payload = _read_json("hardware.json")
    _require_keys(payload, {"schema_version", "profiles"}, "hardware manifest")
    if payload["schema_version"] != 1:
        raise ValueError("Only hardware manifest schema_version=1 is supported")
    profiles = tuple(HardwareProfile.from_dict(item) for item in payload["profiles"])
    result = {profile.id: profile for profile in profiles}
    if len(result) != len(profiles):
        raise ValueError("Hardware profile IDs must be unique")
    return result


def load_protocol(version: int | str) -> Protocol:
    value = int(version)
    if value not in {2, 3}:
        raise ValueError("version must be 2 or 3")
    return Protocol.from_dict(_read_json(f"gates-v{value}.json"))


def protocol_digest(protocol: Protocol) -> str:
    payload = _read_json(f"gates-v{protocol.bass_version}.json")
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
