"""Sample canonical V2 architectures and report structural distributions."""

from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np

from bass.v2 import decode, sample_genome


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.samples < 1:
        raise SystemExit("--samples must be positive")
    rng = np.random.default_rng(args.seed)
    hashes = set()
    channels = Counter()
    configurations = Counter()
    operator_units = Counter()
    operator_repeats = Counter()
    attention_units = []
    active_units = []
    effective_depth = []
    for _ in range(args.samples):
        genome = sample_genome(seed=int(rng.integers(0, np.iinfo(np.int32).max)))
        spec = decode(genome)
        hashes.add(spec.canonical_hash())
        channels[spec.channels] += 1
        active = spec.active_blocks
        configurations.update(
            f"{block.family}/{block.op}/{block.arg}" for block in active
        )
        operator_units.update(block.op for block in active)
        for block in active:
            operator_repeats[block.op] += block.repeat
        active_units.append(len(active))
        attention_units.append(sum(block.family == "attention" for block in active))
        effective_depth.append(sum(block.repeat for block in active))

    payload = {
        "samples": args.samples,
        "seed": args.seed,
        "unique_canonical_hashes": len(hashes),
        "duplicate_rate": 1.0 - len(hashes) / args.samples,
        "channels": dict(sorted(channels.items())),
        "primitive_configurations": dict(sorted(configurations.items())),
        "operator_units": dict(sorted(operator_units.items())),
        "operator_repeats": dict(sorted(operator_repeats.items())),
        "attention_units": _summary(attention_units),
        "active_units": _summary(active_units),
        "repeat_sum": _summary(effective_depth),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as stream:
            stream.write(rendered + "\n")
    return 0


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


if __name__ == "__main__":
    raise SystemExit(main())
