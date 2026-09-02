#!/usr/bin/env python3
"""Exhaustively audit the stage-aware V3 canonical equivalence relation."""

from __future__ import annotations

import argparse
import json
from itertools import pairwise, product
from pathlib import Path

from bass import v3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    return parser


def _exchanges(barriers: tuple[bool, bool]):
    return tuple(
        v3.ExchangeGene.cimex(8) if enabled else v3.ExchangeGene.none()
        for enabled in barriers
    )


def _segments(barriers: tuple[bool, bool]) -> tuple[tuple[int, int], ...]:
    boundaries = [0]
    boundaries.extend(site + 1 for site, enabled in enumerate(barriers) if enabled)
    boundaries.append(3)
    return tuple(pairwise(boundaries))


def _expanded_trace(blocks) -> tuple[tuple[str, str, int], ...]:
    return tuple(
        block.operation_key
        for block in blocks
        if not block.is_skip
        for _ in range(block.repeat)
    )


def _audit_branch_catalog(barriers: tuple[bool, bool]) -> dict[str, int]:
    exchanges = _exchanges(barriers)
    canonical_states: set[tuple[int, ...]] = set()
    raw_count = 0
    for raw_states in product(range(v3.UNIT_STATE_COUNT), repeat=3):
        raw = tuple(v3.state_to_block(state) for state in raw_states)
        canonical = v3.canonicalize_branch(raw, exchanges)
        for start, stop in _segments(barriers):
            if _expanded_trace(raw[start:stop]) != _expanded_trace(
                canonical[start:stop]
            ):
                raise AssertionError(
                    "V3 canonicalization changed execution order across a "
                    f"barrier-delimited segment: {barriers=} {raw_states=}"
                )
        if v3.canonicalize_branch(canonical, exchanges) != canonical:
            raise AssertionError("V3 branch canonicalization is not idempotent")
        canonical_states.add(tuple(v3.block_to_state(block) for block in canonical))
        raw_count += 1
    return {"raw": raw_count, "canonical": len(canonical_states)}


def _audit_counterexamples() -> None:
    skip = v3.BlockGene.skip()
    a1 = v3.BlockGene("cnn", "res_conv", 3, 1)
    a2 = v3.BlockGene("cnn", "res_conv", 3, 2)
    enabled = (v3.ExchangeGene.cimex(8), v3.ExchangeGene.none())

    if v3.canonicalize_branch((skip, a1, skip), enabled) != (skip, a1, skip):
        raise AssertionError("An internal skip moved a transform across CIMEX")
    if v3.canonicalize_branch((a1, a2, skip), enabled) != (a1, a2, skip):
        raise AssertionError("Repeat runs merged across CIMEX")

    disabled = (v3.ExchangeGene.none(), v3.ExchangeGene.none())
    compact = v3.BlockGene("cnn", "res_conv", 3, 3)
    if v3.canonicalize_branch((skip, a1, a2), disabled) != (
        compact,
        skip,
        skip,
    ):
        raise AssertionError("A disabled boundary did not preserve V2 compression")


def run_audit(samples: int, seed: int) -> dict:
    if samples <= 0:
        raise ValueError("samples must be positive")
    _audit_counterexamples()
    catalogs = {
        "none_none": _audit_branch_catalog((False, False)),
        "cimex_none": _audit_branch_catalog((True, False)),
        "none_cimex": _audit_branch_catalog((False, True)),
        "cimex_cimex": _audit_branch_catalog((True, True)),
    }
    expected = {
        "none_none": 68_923,
        "cimex_none": 74_089,
        "none_cimex": 74_089,
        "cimex_cimex": 79_507,
    }
    observed = {name: values["canonical"] for name, values in catalogs.items()}
    if observed != expected:
        raise AssertionError(f"Unexpected V3 branch catalogs: {observed}")

    for index in range(samples):
        genome = v3.sample_canonical_genome(seed=seed + index)
        spec = v3.decode(genome)
        if v3.encode(spec) != genome:
            raise AssertionError("A sampled V3 genome failed its strict round trip")
        for site, exchange in enumerate(spec.exchanges):
            if exchange.is_enabled and not any(
                any(not block.is_skip for block in branch[site + 1 :])
                for branch in spec.branches
            ):
                raise AssertionError("A canonical exchange has no downstream transform")

    return {
        "schema_version": 1,
        "representation": "interaction-semantic-v2",
        "status": "PASS",
        "exhaustive_branch_catalogs": catalogs,
        "sampled_complete_architectures": samples,
        "exact_architecture_count": v3.canonical_architecture_count(),
        "counterexamples": {
            "internal_skip_barrier": "PASS",
            "repeat_barrier": "PASS",
            "none_boundary_compression": "PASS",
        },
    }


def main() -> int:
    args = build_parser().parse_args()
    payload = json.dumps(run_audit(args.samples, args.seed), indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
