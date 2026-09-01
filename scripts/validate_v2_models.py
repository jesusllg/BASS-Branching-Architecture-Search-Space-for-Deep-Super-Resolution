"""Run the audit's executable family-balanced V2 validation stage."""

from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np
import tensorflow as tf

from bass.v2 import build_model, sample
from bass.v2.evaluation import gradient_flow_diagnostics

STRATA = {
    "cnn_heavy": (0.15, lambda fraction: fraction < 1.0 / 3.0),
    "balanced": (0.5, lambda fraction: 1.0 / 3.0 <= fraction <= 2.0 / 3.0),
    "attention_heavy": (0.85, lambda fraction: fraction > 2.0 / 3.0),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input-size", type=int, default=32)
    parser.add_argument("--scale", type=int, choices=[2, 3, 4], default=2)
    parser.add_argument("--output", type=str)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.samples < 1 or args.input_size < 1:
        raise SystemExit("--samples and --input-size must be positive")
    rng = np.random.default_rng(args.seed)
    stratum_names = tuple(STRATA)
    failures = []
    records = []
    for index in range(args.samples):
        stratum = stratum_names[index % len(stratum_names)]
        probability, accepts = STRATA[stratum]
        for _ in range(10_000):
            seed = int(rng.integers(0, np.iinfo(np.int32).max))
            spec = sample(
                seed=seed,
                attention_probability=probability,
                skip_probability=1.0 / 43.0,
            )
            if accepts(spec.attention_fraction):
                break
        else:
            raise RuntimeError(f"Unable to sample the {stratum} stratum")
        try:
            model = build_model(
                spec,
                input_shape=(args.input_size, args.input_size + index % 2, 3),
                upscale_factor=args.scale,
            )
            width = args.input_size + index % 2
            inputs = tf.ones((1, args.input_size, width, 3))
            with tf.GradientTape() as tape:
                output = model(inputs, training=True)
                loss = tf.reduce_mean(output)
            if not bool(tf.reduce_all(tf.math.is_finite(output))):
                raise ValueError("forward pass produced non-finite activations")
            gradients = tape.gradient(loss, model.trainable_variables)
            if any(gradient is None for gradient in gradients):
                raise ValueError("forward/backward produced disconnected gradients")
            diagnostics = gradient_flow_diagnostics(
                model,
                input_shape=(args.input_size, width, 3),
                strict=True,
            )
            records.append(
                {
                    "hash": spec.canonical_hash(),
                    "stratum": stratum,
                    "channels": spec.channels,
                    "attention_fraction": spec.attention_fraction,
                    "attention_repeat_fraction": (
                        sum(
                            block.repeat
                            for block in spec.active_blocks
                            if block.family == "attention"
                        )
                        / sum(block.repeat for block in spec.active_blocks)
                        if spec.active_blocks
                        else 0.0
                    ),
                    "repeat_depth": sum(block.repeat for block in spec.active_blocks),
                    "params": int(model.count_params()),
                    "gradient_coverage": diagnostics.coverage,
                }
            )
        except Exception as error:  # noqa: BLE001 - retain every research failure
            failures.append(
                {"index": index, "hash": spec.canonical_hash(), "error": repr(error)}
            )
        finally:
            tf.keras.backend.clear_session()

    payload = {
        "samples": args.samples,
        "passed": len(records),
        "failed": len(failures),
        "strata_passed": dict(
            sorted(Counter(record["stratum"] for record in records).items())
        ),
        "failures": failures,
        "records": records,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as stream:
            stream.write(rendered + "\n")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
