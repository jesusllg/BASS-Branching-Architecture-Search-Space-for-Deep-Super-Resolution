#!/usr/bin/env python3
"""Structural audit for the canonical 12-gene IBASS V3 search space."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from bass import v3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sampling-prior",
        choices=["canonical", "conditioned"],
        default="canonical",
        help="Compare exact canonical-uniform and conditioned construction priors",
    )
    parser.add_argument("--output", type=Path)
    return parser


def run_audit(samples: int, seed: int, sampling_prior: str = "canonical") -> dict:
    if samples <= 0:
        raise ValueError("samples must be positive")
    if sampling_prior not in {"canonical", "conditioned"}:
        raise ValueError("sampling_prior must be 'canonical' or 'conditioned'")

    channel_counts: Counter[int] = Counter()
    exchange_counts: Counter[str] = Counter()
    enabled_site_counts: Counter[int] = Counter()
    family_counts: Counter[str] = Counter()
    operation_counts: Counter[str] = Counter()
    attention_units = []
    active_units = []
    attention_repeat_depth = []
    cnn_repeat_depth = []
    repeat_depth = []
    hash_to_genome: dict[str, tuple[int, ...]] = {}
    genome_to_hash: dict[tuple[int, ...], str] = {}
    aliases = []

    for index in range(samples):
        genome = tuple(
            v3.sample_canonical_genome(seed=seed + index)
            if sampling_prior == "canonical"
            else v3.sample_genome(seed=seed + index)
        )
        spec = v3.decode(genome)
        if tuple(v3.encode(spec)) != genome:
            raise AssertionError("decode/encode changed a canonical V3 genome")
        if tuple(v3.canonicalize_genome(genome)) != genome:
            raise AssertionError("V3 canonicalization is not idempotent")

        digest = spec.canonical_hash()
        previous_genome = hash_to_genome.setdefault(digest, genome)
        previous_hash = genome_to_hash.setdefault(genome, digest)
        if previous_genome != genome or previous_hash != digest:
            aliases.append(
                {
                    "hash": digest,
                    "first_genome": list(previous_genome),
                    "second_genome": list(genome),
                }
            )

        channel_counts[spec.channels] += 1
        enabled_site_counts[spec.exchange_count] += 1
        for exchange in spec.exchanges:
            key = "none" if not exchange.is_enabled else f"cimex_k{exchange.prototypes}"
            exchange_counts[key] += 1
        for block in spec.flat_blocks:
            family_counts[block.family] += 1
            operation_counts[block.op] += 1
        active = spec.active_blocks
        active_units.append(len(active))
        attention_units.append(sum(block.family == "attention" for block in active))
        attention_repeat_depth.append(
            sum(block.repeat for block in active if block.family == "attention")
        )
        cnn_repeat_depth.append(
            sum(block.repeat for block in active if block.family == "cnn")
        )
        repeat_depth.append(sum(block.repeat for block in active))

    if aliases:
        raise AssertionError(
            f"Found {len(aliases)} structural aliases in the V3 semantic codec"
        )

    unique = len(genome_to_hash)
    return {
        "schema_version": 3,
        "representation": "interaction-semantic-v2",
        "sampling_prior": (
            "exact_uniform_canonical_architectures"
            if sampling_prior == "canonical"
            else "conditioned_raw_grid_then_canonicalized"
        ),
        "requested_samples": samples,
        "unique_samples": unique,
        "duplicate_draws": samples - unique,
        "canonical_aliases": 0,
        "semantic_genome_length": v3.SEMANTIC_GENOME_LENGTH,
        "semantic_grid_size_before_canonical_quotient": 4 * (43**9) * (3**2),
        "effective_stage_aware_architecture_count": v3.canonical_architecture_count(),
        "canonical_branch_catalog_sizes": {
            "none_none": len(v3.canonical_branch_genomes((False, False))),
            "cimex_none": len(v3.canonical_branch_genomes((True, False))),
            "none_cimex": len(v3.canonical_branch_genomes((False, True))),
            "cimex_cimex": len(v3.canonical_branch_genomes((True, True))),
        },
        "channel_counts": dict(sorted(channel_counts.items())),
        "enabled_exchange_site_counts": dict(sorted(enabled_site_counts.items())),
        "exchange_state_counts": dict(sorted(exchange_counts.items())),
        "unit_family_counts": dict(sorted(family_counts.items())),
        "operation_counts": dict(sorted(operation_counts.items())),
        "attention_units": _summary(attention_units),
        "active_units": _summary(active_units),
        "attention_repeat_depth": _summary(attention_repeat_depth),
        "cnn_repeat_depth": _summary(cnn_repeat_depth),
        "repeat_depth": _summary(repeat_depth),
    }


def _summary(values: list[int]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "min": float(np.min(array)),
        "q1": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "q3": float(np.quantile(array, 0.75)),
        "max": float(np.max(array)),
    }


def main() -> int:
    args = build_parser().parse_args()
    report = run_audit(args.samples, args.seed, args.sampling_prior)
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
