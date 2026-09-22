"""The JAX-side process: run the route on one problem and write its outcome as JSON.

    $JAX_PYTHON -m petab_harness.neural_ode.worker <problem_dir> --out <json> --config <json>
        [--work <dir>] [--seconds <s>]

Exit 0 whenever the file was written (a refusal is a result), 2 when the route could not be
imported here, 3 when it raised. Both failures still write ``ok: false`` with a reason and a
traceback, so the parent can tell an import problem from a crash mid-training.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("problem_dir")
    parser.add_argument("--out", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--work", default=None)
    parser.add_argument("--seconds", type=float, default=None)
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    def fail(code: int, reason: str) -> int:
        out.write_text(json.dumps({"ok": False, "reason": reason,
                                   "traceback": traceback.format_exc()}, indent=1))
        return code

    try:  # inside main, so an import error still leaves a payload the parent can read
        from . import route
        from .config import Config
    except Exception as exc:                                    # noqa: BLE001 -- reported
        return fail(2, f"could not import the route in {sys.executable}: {exc}")
    try:
        config = Config.load(args.config) if args.config else Config()
        config = Config.from_dict({**config.as_dict(), "interpreter": None})  # never re-delegate
        outcome = route.propose(args.problem_dir, config, work_dir=args.work, seconds=args.seconds)
    except Exception as exc:                                    # noqa: BLE001 -- reported
        return fail(3, f"the route raised {type(exc).__name__}: {exc}")
    out.write_text(json.dumps({"ok": True, "interpreter": sys.executable,
                               "outcome": outcome.as_dict()}, indent=1, default=str))
    print(f"wrote {out}: {len(outcome.proposals)} proposal(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
