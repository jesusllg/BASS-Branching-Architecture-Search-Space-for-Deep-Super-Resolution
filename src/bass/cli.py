"""Command-line entry point for the separate BASS V1, V2, and V3 spaces."""

from __future__ import annotations

import argparse

from .shared.nsga3 import ReferenceDirectionEA


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--genome-version", type=int, choices=[1, 2, 3], default=1)
    parser.add_argument("--population", type=int, default=20)
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument(
        "--metric", choices=["gradient_flow", "synflow"], default="gradient_flow"
    )
    parser.add_argument(
        "--head-mode",
        choices=["residual", "direct"],
        default="residual",
        help="V2/V3 fixed SR head; direct is retained only for ablation",
    )
    parser.add_argument(
        "--exchange-probability",
        type=float,
        default=None,
        help=(
            "V3 conditioned initialization probability per active CIMEX site; "
            "omit for an exactly uniform canonical V3 prior"
        ),
    )
    parser.add_argument("--scale", type=int, choices=[2, 3, 4], default=2)
    parser.add_argument("--input-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-flops", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    problem_args = {
        "metric": args.metric,
        "upscale_factor": args.scale,
        "input_shape": (args.input_size, args.input_size, 3),
        "include_flops": not args.skip_flops,
        "evaluation_seed": args.seed,
    }
    if args.genome_version == 1:
        from .v1.problem import BASSProblem
    elif args.genome_version == 2:
        if args.metric == "synflow":
            raise SystemExit(
                "BASS V2 does not implement canonical SynFlow; "
                "use --metric gradient_flow"
            )
        from .v2.problem import BASSProblem

        problem_args["head_mode"] = args.head_mode
    else:
        if args.metric == "synflow":
            raise SystemExit(
                "BASS V3 does not implement canonical SynFlow; "
                "use --metric gradient_flow"
            )
        from .v3.problem import BASSProblem

        problem_args["head_mode"] = args.head_mode
        problem_args["exchange_probability"] = args.exchange_probability

    problem = BASSProblem(**problem_args)
    optimizer = ReferenceDirectionEA(
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
    print(f"Canonical duplicates rejected: {optimizer.duplicate_rejections}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
