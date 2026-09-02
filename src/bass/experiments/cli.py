"""Inspect and prepare the frozen BASS V2/V3 experimental gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contracts import load_hardware_profiles, load_protocol
from .ledger import validate_gate_ledger
from .results import validate_result_schema, verify_result_evidence
from .work_orders import (
    create_work_order,
    render_slurm,
    validate_work_order,
    write_work_order,
)


def _version(value: str) -> int:
    normalized = value.lower().removeprefix("v")
    version = int(normalized)
    if version not in {2, 3}:
        raise argparse.ArgumentTypeError("version must be v2 or v3")
    return version


def _parameters(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"parameter must be KEY=VALUE: {value}")
        key, item = value.split("=", 1)
        if not key or key in result:
            raise SystemExit(f"invalid or duplicate parameter: {key}")
        result[key] = item
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="list every V2 and V3 gate")

    show = subparsers.add_parser("show", help="show a protocol or one gate")
    show.add_argument("version", type=_version)
    show.add_argument("gate", nargs="?")

    prepare = subparsers.add_parser("prepare", help="write an immutable work order")
    prepare.add_argument("version", type=_version)
    prepare.add_argument("gate")
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--parameter", action="append", default=[])
    prepare.add_argument("--smoke", action="store_true")
    prepare.add_argument("--slurm", type=Path)

    validate_order = subparsers.add_parser(
        "validate-work-order", help="validate a prepared work order"
    )
    validate_order.add_argument("path", type=Path)

    validate = subparsers.add_parser(
        "validate-result", help="validate a result envelope against its gate"
    )
    validate.add_argument("version", type=_version)
    validate.add_argument("gate")
    validate.add_argument("path", type=Path)
    validate.add_argument("--work-order", type=Path, required=True)

    verify = subparsers.add_parser(
        "verify-result", help="verify a result envelope and its artifact bytes"
    )
    verify.add_argument("version", type=_version)
    verify.add_argument("gate")
    verify.add_argument("path", type=Path)
    verify.add_argument("--work-order", type=Path, required=True)
    verify.add_argument("--artifact-root", type=Path, required=True)

    ledger = subparsers.add_parser(
        "validate-ledger", help="validate a signed final gate ledger"
    )
    ledger.add_argument("version", type=_version)
    ledger.add_argument("path", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "list":
        for version in (2, 3):
            protocol = load_protocol(version)
            for gate in protocol.gates:
                print(f"{gate.id}\t{gate.hardware}\t{gate.decision}\t{gate.title}")
        return 0
    if args.command == "validate-work-order":
        payload = json.loads(args.path.read_text(encoding="utf-8"))
        protocols = [load_protocol(version) for version in (2, 3)]
        matching = [
            protocol
            for protocol in protocols
            if protocol.protocol_id == payload.get("protocol_id")
        ]
        if len(matching) != 1:
            errors = ["work order does not identify a current V2/V3 protocol"]
        else:
            protocol = matching[0]
            try:
                gate = protocol.gate(payload.get("gate_id"))
            except KeyError as error:
                errors = [str(error)]
            else:
                hardware = load_hardware_profiles()[gate.hardware]
                errors = validate_work_order(
                    payload, protocol=protocol, gate=gate, hardware=hardware
                )
        print(json.dumps({"valid": not errors, "errors": errors}, indent=2))
        return int(bool(errors))

    protocol = load_protocol(args.version)
    gate = protocol.gate(args.gate) if getattr(args, "gate", None) else None
    if args.command == "show":
        value = gate if gate is not None else protocol
        from dataclasses import asdict

        print(json.dumps(asdict(value), indent=2, sort_keys=True))
        return 0
    if args.command == "prepare":
        assert gate is not None
        hardware = load_hardware_profiles()[gate.hardware]
        payload = create_work_order(
            protocol,
            gate,
            hardware,
            parameters=_parameters(args.parameter),
            smoke=args.smoke,
        )
        write_work_order(payload, args.output)
        if args.slurm is not None:
            args.slurm.parent.mkdir(parents=True, exist_ok=True)
            args.slurm.write_text(render_slurm(payload), encoding="utf-8")
        print(args.output)
        return 0
    if args.command == "validate-result":
        assert gate is not None
        payload = json.loads(args.path.read_text(encoding="utf-8"))
        work_order = json.loads(args.work_order.read_text(encoding="utf-8"))
        errors = validate_result_schema(payload, protocol, gate, work_order)
        print(
            json.dumps(
                {"valid": not errors, "verification": "schema", "errors": errors},
                indent=2,
            )
        )
        return int(bool(errors))
    if args.command == "verify-result":
        assert gate is not None
        payload = json.loads(args.path.read_text(encoding="utf-8"))
        work_order = json.loads(args.work_order.read_text(encoding="utf-8"))
        errors = verify_result_evidence(
            payload,
            protocol,
            gate,
            work_order,
            artifact_root=args.artifact_root,
        )
        print(
            json.dumps(
                {"valid": not errors, "verification": "evidence", "errors": errors},
                indent=2,
            )
        )
        return int(bool(errors))
    if args.command == "validate-ledger":
        payload = json.loads(args.path.read_text(encoding="utf-8"))
        errors = validate_gate_ledger(payload, protocol)
        print(json.dumps({"valid": not errors, "errors": errors}, indent=2))
        return int(bool(errors))
    raise AssertionError(f"Unhandled command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
