"""The neural route from the command line: fit and score one field, or propose models.

    python -m petab_harness.neural_ode fit examples/fujita/problem --out runs/fujita \\
        [--windows 5 --steps 1600 --restarts 2 --aug 0 --width 32 --depth 2 --seed 0] \\
        [--dosed S] [--shared-ics] [--clock] \\
        [--heldout PATH | --no-heldout] [--reference PATH]
    python -m petab_harness.neural_ode propose examples/fujita/problem --out runs/fujita \\
        [--config neural.json] [--interpreter PY] [--pysr-interpreter PY] [--seconds 8400] \\
        [--threads 16] [--dosed S]
    python -m petab_harness.neural_ode score runs/fujita/proposals.json --example examples/fujita \\
        --out runs/fujita/judge [--n-starts 50] [--only selected,c7]

``fit`` writes ``<out>/result.json`` -- the field's training nll, its ``k`` (the scalars PEtab
estimates, never the weights) and its score, the held-out block in the shape every run's
``result.json`` carries, and the training record -- and ``<out>/model/``, the trained field as a
complete PEtab SciML problem. For an ``examples/<name>/problem`` the held-out set and the
reference are found where the harness writes them. Exit 0 when a field trained, 1 when every
restart failed.

``propose`` runs the route (:mod:`.route`): one training and one distillation at the settings
given, and writes ``<out>/proposals.json``; flags override the config file. Exit 0 for proposals,
1 for a refusal.

``score`` commits each proposal through the server and judges it as a panel's model would be
(:mod:`.judge`), one private copy of the example per proposal; ``<out>/scores.json`` collects
the rows. Exit 0 when every proposal was judged, 1 otherwise.
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
    propose = commands.add_parser("propose", help="train, distil and write proposals")
    propose.add_argument("problem", type=Path, help="a PEtab v1 problem yaml or its directory")
    propose.add_argument("--out", type=Path, required=True)
    propose.add_argument("--config", type=Path, default=None)
    propose.add_argument("--interpreter", default=None)
    propose.add_argument("--pysr-interpreter", dest="pysr_interpreter", default=None)
    propose.add_argument("--seconds", type=float, default=None)
    propose.add_argument("--threads", type=int, default=None)
    propose.add_argument("--dosed", default=None, help="condition columns that dose a state, comma-separated")
    score = commands.add_parser("score", help="judge the proposals through the server")
    score.add_argument("proposals", type=Path, help="the route's proposals.json")
    score.add_argument("--example", type=Path, required=True, help="the example directory (holds problem/)")
    score.add_argument("--out", type=Path, required=True)
    score.add_argument("--n-starts", dest="n_starts", type=int, default=50)
    score.add_argument("--only", default="", help="tags to judge, comma-separated (default: all)")
    args = parser.parse_args(argv)
    if args.command == "score":
        from . import judge  # noqa: PLC0415

        rows = judge.score_proposals(args.proposals, args.example, args.out, n_starts=args.n_starts,
                                     only=tuple(x for x in args.only.split(",") if x))
        return 0 if rows and all(r["exit_code"] == 0 for r in rows) else 1
    return _fit(args) if args.command == "fit" else _propose(args)


def _propose(args) -> int:
    from . import route  # noqa: PLC0415
    from .config import Config  # noqa: PLC0415

    config = Config.load(args.config) if args.config else Config()
    overrides = {k: getattr(args, k) for k in ("interpreter", "pysr_interpreter", "seconds", "threads")
                 if getattr(args, k) is not None}
    if args.dosed is not None:
        overrides["dosed"] = [x for x in args.dosed.split(",") if x]
    config = Config.from_dict({**config.as_dict(), **overrides})
    args.out.mkdir(parents=True, exist_ok=True)
    problem_dir = args.problem.parent if args.problem.is_file() else args.problem
    outcome = route.propose(problem_dir, config, work_dir=args.out / "work")
    path = args.out / "proposals.json"
    path.write_text(json.dumps({**outcome.as_dict(), "problem_dir": str(problem_dir),
                                "config": config.as_dict()}, indent=1, default=str) + "\n")
    if outcome.refusal is not None:
        print(f"refused: {outcome.refusal.reason}", file=sys.stderr)
        return 1
    print(f"wrote {path}: {len(outcome.proposals)} proposal(s)")
    return 0


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
    record["held_out"] = payload  # the scorer's own payload, k and bic of a transfer included
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
