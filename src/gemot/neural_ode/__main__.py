"""Fit the neural ODE to one PEtab problem and score it, from the command line.

    python -m petab_harness.neural_ode fit examples/fujita/problem --out runs/fujita \\
        [--windows 5 --steps 1600 --restarts 2 --aug 0 --width 32 --depth 2 --seed 0] \\
        [--dosed S] [--shared-ics] [--clock] \\
        [--heldout PATH | --no-heldout] [--reference PATH]

Writes ``<out>/result.json`` -- the field's training nll, its ``k`` (the scalars PEtab estimates,
never the weights) and BIC, the held-out block in the shape every run's ``result.json`` carries,
and the training record -- and ``<out>/model/``, the trained field as a complete PEtab SciML
problem. For an ``examples/<name>/problem`` the held-out set and the reference are found where
the harness writes them. Exit 0 when a field trained, 1 when every restart failed.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

METHOD = "neural_ode"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m petab_harness.neural_ode", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("fit", help="train the field on one problem and score it")
    fit.add_argument("problem", type=Path, help="a PEtab v1 problem yaml or its directory")
    fit.add_argument("--out", type=Path, required=True)
    fit.add_argument("--dosed", default="", help="condition columns that dose a state, comma-separated")
    fit.add_argument("--windows", type=int, default=1)
    fit.add_argument("--continuity", type=float, default=1.0)
    fit.add_argument("--steps", type=int, default=1500)
    fit.add_argument("--restarts", type=int, default=4)
    fit.add_argument("--aug", type=int, default=0)
    fit.add_argument("--width", type=int, default=32)
    fit.add_argument("--depth", type=int, default=2)
    fit.add_argument("--seed", type=int, default=0)
    fit.add_argument("--shared-ics", dest="shared_ics", action="store_true")
    fit.add_argument("--clock", action="store_true")
    fit.add_argument("--heldout", type=Path, default=None, help="held-out measurement table")
    fit.add_argument("--no-heldout", dest="no_heldout", action="store_true")
    fit.add_argument("--reference", type=Path, default=None, help="reference.json to compare against")
    args = parser.parse_args(argv)
    return _fit(args)


def _fit(args) -> int:
    from . import heldout as H  # noqa: PLC0415
    from . import problem as P  # noqa: PLC0415
    from . import train as T  # noqa: PLC0415

    yaml_path = args.problem if args.problem.is_file() else args.problem / "problem.yaml"
    problem_dir = yaml_path.parent
    problem = P.load(yaml_path, dosed=tuple(x for x in args.dosed.split(",") if x))
    config = T.TrainConfig(n_augmented=args.aug, width=args.width, depth=args.depth, clock=args.clock,
                           shared_initial_conditions=args.shared_ics,
                           n_windows=args.windows,
                           continuity_weight=args.continuity, steps=args.steps, n_restarts=args.restarts,
                           seed=args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    result = T.train(problem, args.out / "train", config)
    T.save_bundle(result, args.out / "model")

    n = problem.n_measurements
    record = {
        "method": METHOD, "problem": problem.name, "problem_dir": str(problem_dir),
        "config": asdict(config), "config_key": config.key(),
        "nll": None if not math.isfinite(result.nll) else result.nll,
        "k": result.n_ode_parameters, "n": n,
        "bic": H.score_of(result.nll, result.n_ode_parameters, n),
        "n_network_weights": result.n_network_weights, "solver_failed": result.solver_failed,
        "steps_run": result.steps_run, "restart_failures": list(result.restart_failures),
        "stage_objectives": list(result.stage_objectives), "notes": list(result.notes),
        "parameters": result.parameters, "problem_warnings": list(problem.warnings),
        "wall_seconds": result.wall_seconds, "model_dir": str(args.out / "model"),
    }

    held_out_path = None if args.no_heldout else (args.heldout or _default_held_out(problem_dir))
    reference_path = args.reference or (problem_dir / "reference.json")
    reference = json.loads(reference_path.read_text()) if reference_path.is_file() else None
    payload = None
    if held_out_path is not None and not result.solver_failed:
        from petab_mcp.validation_data import read_held_out  # noqa: PLC0415

        payload = H.score(result, read_held_out(held_out_path), args.out / "heldout",
                          training=problem)
    from ..runner.scoring import _validation_fields  # noqa: PLC0415

    record["validation"] = _validation_fields(payload, reference, asked=held_out_path is not None)
    record["held_out_path"] = None if held_out_path is None else str(held_out_path)
    record["reference"] = None if reference is None else {k: reference.get(k) for k in ("nll", "k", "n")}
    (args.out / "result.json").write_text(json.dumps(record, indent=1, default=str) + "\n")

    validation = record["validation"] or {}
    print(f"{problem.name}: nll {record['nll']} k {record['k']} n {n} bic {record['bic']}; "
          f"held-out {validation.get('status', 'none')} {validation.get('mode', '')} "
          f"nll {validation.get('nll')} delta {validation.get('delta_nll')}; "
          f"{result.wall_seconds:.0f}s -> {args.out / 'result.json'}")
    return 1 if result.solver_failed else 0


def _default_held_out(problem_dir: Path) -> Path | None:
    """``examples/<name>/validation/measurements.tsv`` for an example's ``problem/``, where the
    harness materializes the held-out set; ``None`` when there is none."""
    if problem_dir.name != "problem":
        return None
    path = problem_dir.parent / "validation" / "measurements.tsv"
    return path if path.is_file() else None


if __name__ == "__main__":
    sys.exit(main())
