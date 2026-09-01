#!/usr/bin/env python3
"""Build/forward/backward validation for family-balanced IBASS V3 models."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import tensorflow as tf

from bass import v3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input-size", type=int, default=16)
    parser.add_argument("--scale", type=int, choices=[2, 3, 4], default=2)
    parser.add_argument("--output", type=Path)
    return parser


def _sampling_stratum(index: int) -> tuple[float, float, str]:
    strata = (
        (0.2, 0.0, "cnn_dominant_no_exchange"),
        (0.2, 1.0, "cnn_dominant_exchange"),
        (0.5, 0.5, "hybrid_mixed_exchange"),
        (0.8, 0.0, "attention_dominant_no_exchange"),
        (0.8, 1.0, "attention_dominant_exchange"),
    )
    return strata[index % len(strata)]


def run_validation(samples: int, seed: int, input_size: int, scale: int) -> dict:
    if samples <= 0:
        raise ValueError("samples must be positive")
    if input_size <= 0:
        raise ValueError("input-size must be positive")

    failures = []
    strata_counts: Counter[str] = Counter()
    exchange_counts: Counter[int] = Counter()
    parameters = []
    attention_unit_fraction = []
    attention_repeat_fraction = []
    repeat_depth = []
    durations = []
    expected_shape = (1, input_size * scale, input_size * scale, 3)

    for index in range(samples):
        attention_probability, exchange_probability, stratum = _sampling_stratum(index)
        strata_counts[stratum] += 1
        tf.keras.backend.clear_session()
        started = time.perf_counter()
        try:
            spec = v3.sample(
                seed=seed + index,
                attention_probability=attention_probability,
                exchange_probability=exchange_probability,
            )
            exchange_counts[spec.exchange_count] += 1
            active = spec.active_blocks
            total_repeat = sum(block.repeat for block in active)
            attention_unit_fraction.append(spec.attention_fraction)
            attention_repeat_fraction.append(
                sum(block.repeat for block in active if block.family == "attention")
                / total_repeat
                if total_repeat
                else 0.0
            )
            repeat_depth.append(total_repeat)
            model = v3.build_model(
                spec,
                upscale_factor=scale,
                input_shape=(input_size, input_size, 3),
            )
            sample = tf.random.stateless_uniform(
                (1, input_size, input_size, 3), seed=(seed, index)
            )
            with tf.GradientTape() as tape:
                output = model(sample, training=True)
                loss = tf.reduce_mean(tf.square(output))
            gradients = tape.gradient(loss, model.trainable_variables)
            if tuple(output.shape) != expected_shape:
                raise ValueError(
                    f"wrong output shape {tuple(output.shape)} != {expected_shape}"
                )
            if not bool(tf.reduce_all(tf.math.is_finite(output))):
                raise ValueError("non-finite model output")
            if any(gradient is None for gradient in gradients):
                raise ValueError("disconnected trainable variable")
            if not all(
                bool(tf.reduce_all(tf.math.is_finite(gradient)))
                for gradient in gradients
            ):
                raise ValueError("non-finite gradient")
            parameters.append(int(model.count_params()))
        except Exception as error:  # noqa: BLE001 - audit every failing seed
            failures.append(
                {
                    "index": index,
                    "seed": seed + index,
                    "stratum": stratum,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
        finally:
            durations.append(time.perf_counter() - started)
            tf.keras.backend.clear_session()

    report = {
        "schema_version": 3,
        "requested_samples": samples,
        "successful_models": samples - len(failures),
        "failed_models": len(failures),
        "input_size": input_size,
        "scale": scale,
        "strata_counts": dict(sorted(strata_counts.items())),
        "enabled_exchange_site_counts": dict(sorted(exchange_counts.items())),
        "parameter_min": min(parameters) if parameters else None,
        "parameter_max": max(parameters) if parameters else None,
        "attention_unit_fraction": _summary(attention_unit_fraction),
        "attention_repeat_fraction": _summary(attention_repeat_fraction),
        "repeat_depth": _summary(repeat_depth),
        "mean_seconds_per_model": sum(durations) / len(durations),
        "failures": failures,
    }
    return report


def _summary(values: list[float | int]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    count = len(ordered)

    def quantile(fraction: float) -> float:
        position = (count - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, count - 1)
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "min": ordered[0],
        "q1": quantile(0.25),
        "median": quantile(0.5),
        "mean": sum(ordered) / count,
        "q3": quantile(0.75),
        "max": ordered[-1],
    }


def main() -> int:
    args = build_parser().parse_args()
    report = run_validation(args.samples, args.seed, args.input_size, args.scale)
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return int(bool(report["failed_models"]))


if __name__ == "__main__":
    raise SystemExit(main())
