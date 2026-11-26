"""The run explorer: a checkout's runs, archived or still running, each opened in petab-mcp's viewer.

``/`` lists the runs, the running ones first. ``/run?run=<id>`` is petab-mcp's viewer of the run's
version store (``petab_mcp.viz``: the multistart, the model, the fit, version by version) with the
agents' conversation as a tab of its own (:mod:`.conversation`). A running run is followed as the
server writes it.

The viewer reads two things: the run's store, and the PEtab problem the server was given, for the
design tables it does not store. A live run has both in its ``work/``. An archive keeps the store
(:mod:`.archive`, laid out as ``.petab_mcp/``, which the viewer reads as it is) but not the
problem, so an archived run is shown against its example's pristine ``problem/``: the server
never changes the design tables, so they are the ones it scored against.

Panel and the viewer come with the ``explorer`` extra, and are imported only here: nothing that
imports :mod:`.bundle` or :mod:`.archive` pays for them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import panel as pn

from . import archive
from .conversation import Conversation


@dataclass(frozen=True)
class Located:
    """A run as the viewer needs it: its store, the problem's YAML (``None`` where its example's
    problem is not on this machine), and whether the server is still writing the store."""

    store: Path
    problem: Path | None
    live: bool
    run_dir: Path
    meta: dict


def _problem_yaml(problem_dir: Path) -> Path | None:
    if (problem_dir / "problem.yaml").is_file():
        return problem_dir / "problem.yaml"
    found = sorted(problem_dir.glob("*.yaml")) if problem_dir.is_dir() else []
    return found[0] if len(found) == 1 else None


def locate(run_id: str, archive_root: Path, examples_root: Path) -> Located | None:
    """``run_id``'s store and problem: the live run's ``work/``, else the archive's store against
    its example's ``problem/``. ``None`` for no such run."""
    live = archive.load_live_run(run_id, examples_root)
    if live is not None:
        work, meta = live
        return Located(work / ".petab_mcp", _problem_yaml(work), True, work, meta)
    loaded = archive.load_run_meta(run_id, archive_root)
    if loaded is None:
        return None
    run_dir, meta = loaded
    scenario = str(meta.get("scenario") or "")
    problem = _problem_yaml(examples_root / scenario / "problem") if scenario else None
    return Located(run_dir / ".petab_mcp", problem, False, run_dir, meta)


#: The run list's columns, as the page names them.
COLUMNS = {
    "run_id": "run",
    "status": "status",
    "scenario": "scenario",
    "model": "model",
    "driver": "driver",
    "delta_bic": "ΔBIC",
    "n_versions": "versions",
    "run_started_at": "started",
}


def runs(archive_root: Path, examples_root: Path) -> pd.DataFrame:
    """The runs, the running ones first, then the archived newest first."""
    rows = []
    for r in archive.list_live_runs(examples_root):
        rows.append({**r, "status": "running"})
    for r in archive.list_runs(archive_root):
        status = {True: "succeeded", False: "failed"}.get(r.get("success"), "finished")
        rows.append(
            {**r, "status": f"{status}: {r['stop_reason']}" if r.get("stop_reason") else status}
        )
    df = pd.DataFrame(rows, columns=list(COLUMNS))
    df["delta_bic"] = pd.to_numeric(df["delta_bic"], errors="coerce").round(1)
    return df.rename(columns=COLUMNS)


def index(archive_root: Path, examples_root: Path) -> pn.viewable.Viewable:
    table = pn.widgets.Tabulator(
        runs(archive_root, examples_root),
        disabled=True,
        show_index=False,
        sizing_mode="stretch_width",
        formatters={
            "run": {
                "type": "link",
                "urlPrefix": "/run?run=",
                "target": "_blank",
            }
        },
        header_filters=True,
        page_size=50,
        pagination="local",
    )

    def refresh():
        table.value = runs(archive_root, examples_root)

    pn.state.add_periodic_callback(refresh, period=30_000)
    return pn.template.FastListTemplate(
        title="gemot runs",
        main=[
            pn.pane.Markdown(
                f"Runs in `{archive_root}`, and those running under `{examples_root}`. Open a run "
                "to see its versions in petab-mcp's viewer, with the agents' conversation.",
                sizing_mode="stretch_width",
            ),
            table,
        ],
    )


def run_page(
    run_id: str, archive_root: Path, examples_root: Path, phoenix_base_url: str
) -> pn.viewable.Viewable:
    from petab_mcp.viz.app import build

    found = locate(run_id, archive_root, examples_root)
    if found is None or not (found.store / "versions").is_dir():
        return pn.pane.Markdown(f"No version store for run `{run_id}`.")
    if found.problem is None:
        return pn.pane.Markdown(
            f"Run `{run_id}`'s example has no problem on this machine, which the viewer reads the "
            "design tables from."
        )
    conversation = Conversation(run_id, found.run_dir, found.meta, phoenix_base_url, found.live)
    scenario = found.meta.get("scenario") or run_id
    return build(
        found.store,
        found.problem,
        poll_seconds=3.0 if found.live else 0,
        extra_tabs=[("Conversation", conversation.view)],
        title=f"gemot · {scenario} · {run_id}",
    )


def phoenix_base_url(explicit: str | None) -> str:
    """The Phoenix the conversations are read from: ``explicit``, then the standard environment
    variables, then the local default, normalised to the base the Phoenix client wants."""
    from ..tracing import phoenix_base_url as base

    return base(
        explicit
        or os.environ.get("PHOENIX_COLLECTOR_ENDPOINT")
        or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
        or "http://localhost:6006"
    )


def serve(
    archive_root: Path | None = None,
    examples_root: Path | None = None,
    phoenix: str | None = None,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    archive_root = Path(archive_root) if archive_root else archive.default_archive_root()
    examples_root = Path(examples_root) if examples_root else archive_root.parent
    px = phoenix_base_url(phoenix)

    def run():
        run_id = (pn.state.session_args.get("run") or [b""])[0].decode()
        return run_page(run_id, archive_root, examples_root, px)

    # Bokeh checks a page's websocket against the address it is browsed at: bound to every
    # interface on a compute node, it is browsed through an SSH tunnel at an address it cannot
    # know (gemot.sweep's explorer job).
    origins = ["*"] if host == "0.0.0.0" else [f"{host}:{port}", f"localhost:{port}"]
    pn.serve(
        {"/": lambda: index(archive_root, examples_root), "/run": run},
        address=host,
        port=port,
        show=False,
        websocket_origin=origins,
        title="gemot runs",
    )
