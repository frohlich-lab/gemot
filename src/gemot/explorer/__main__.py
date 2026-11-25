"""``python -m gemot.explorer``: serve the run explorer over a checkout's archived and live runs.

.. code-block:: bash

    python -m gemot.explorer                 # serve archived runs on :8000
    python -m gemot.explorer --port 8080
    python -m gemot.explorer --phoenix http://localhost:6006 --archive examples/_runs_archive

Each harness run archives its on-disk half (:mod:`.archive`); this serves it with the run's Phoenix
trace in one scroll-synced viewer. The conversation panels need a persistent Phoenix (the collector
the runs exported to) reachable at ``--phoenix``; the model-variant and nllh/BIC panels work from
the archive alone.

The entry point the sweep's explorer job runs (``gemot/sweep/explorer_server.sh``), in place of
the ``scripts/explore_ui.py`` each checkout used to carry.
"""

from __future__ import annotations

import argparse

from .server import create_app


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
    p.add_argument(
        "--phoenix",
        default=None,
        help="Phoenix base URL (default: $PHOENIX_COLLECTOR_ENDPOINT or :6006)",
    )
    args = p.parse_args(argv)

    import uvicorn

    app = create_app(
        archive_root=args.archive, phoenix_base_url=args.phoenix, examples_root=args.examples_root
    )
    print(
        f"[explorer] serving runs from "
        f"{args.archive or 'examples/_runs_archive'} on http://{args.host}:{args.port}"
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
