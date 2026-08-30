"""Command-line entry point for the repaired original BASS demonstration."""

from __future__ import annotations

import argparse

from .nsga3 import NSGA3
from .problem import BASSProblem


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--genome-version", type=int, choices=[1, 2], default=1)
    parser.add_argument("--population", type=int, default=20)
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--metric", choices=["synflow"], default="synflow")
    parser.add_argument("--scale", type=int, choices=[2, 3, 4], default=2)
    parser.add_argument("--input-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-flops", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    problem = BASSProblem(
        genome_version=args.genome_version,
        metric=args.metric,
        upscale_factor=args.scale,
        input_shape=(args.input_size, args.input_size, 3),
        include_flops=not args.skip_flops,
        evaluation_seed=args.seed,
    )
    optimizer = NSGA3(
        problem,
        pop_size=args.population,
        n_gen=args.generations,
        seed=args.seed,
        verbose=True,
    )
    _, non_dominated = optimizer.run()
    print("Non-dominated solutions:")
    for genome, objectives in zip(non_dominated["X"], non_dominated["F"]):
        print(f"genome={genome.tolist()} objectives={objectives.tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
