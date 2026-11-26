"""``python -m gemot.explorer``: serve the run explorer over a checkout's archived and live runs.

.. code-block:: bash

    python -m gemot.explorer                 # serve archived runs on :8000
    python -m gemot.explorer --port 8080
    python -m gemot.explorer --archive examples/_runs_archive

Each harness run archives its on-disk half (:mod:`.archive`); this serves it in petab-mcp's viewer
(:mod:`.app`), from the archive alone.

The entry point the sweep's explorer job runs (``gemot/sweep/explorer_server.sh``).
"""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m gemot.explorer", description="gemot run explorer")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument(
        "--archive", default=None, help="runs archive dir (default: examples/_runs_archive)"
    )
    p.add_argument(
        "--examples-root",
        default=None,
        help="dir holding each example's live work/ (default: the archive's parent, "
        "i.e. examples/) — where in-flight runs are discovered for live watching",
    )
    args = p.parse_args(argv)
    try:
        from .app import serve
    except ImportError as exc:
        raise SystemExit(
            f"the explorer needs gemot's `explorer` extra, petab-mcp's viewer with it: {exc}"
        ) from exc
    print(
        f"[explorer] serving runs from "
        f"{args.archive or 'examples/_runs_archive'} on http://{args.host}:{args.port}"
    )
    serve(args.archive, args.examples_root, args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
