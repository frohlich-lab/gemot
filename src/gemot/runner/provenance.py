"""What a run records about itself: timestamps, code version, reference score, live marker."""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
from pathlib import Path

from ..scenario import Scenario

# --------------------------------------------------------------------------- provenance
def _now_iso() -> str:
    """Current UTC time as an ISO-8601 string (for the metrics history + trace join)."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _git_sha(path: Path) -> str | None:
    """Short HEAD sha (+``+dirty`` if the tree has uncommitted changes), or ``None``.

    Provenance for the metrics history so a run is attributable to a commit; best-effort so
    a detached/absent git checkout never breaks the run.
    """
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(path),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return None
    try:
        dirty = subprocess.call(
            ["git", "diff", "--quiet", "--ignore-submodules"], cwd=str(path),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except Exception:
        dirty = 0
    return f"{sha}+dirty" if dirty else sha


def _bic_reference(ref) -> float | None:
    """BIC of the true-model reference, or ``None`` when it is absent or malformed.

    ``scenario.reference()`` catches JSON *parse* errors but not a ``reference.json`` that parses
    yet lacks ``nll``/``k``/``n`` (or carries ``null``). Guard here so an author-side reference
    problem degrades to "no BIC gate" rather than raising a ``KeyError``/``TypeError`` that
    ``_drive``'s broad handler would misreport as an aborted (provider) run — and so the live
    marker and the judge compute the reference identically."""
    if not ref:
        return None
    from ..scenario import bic
    try:
        return bic(ref["nll"], ref["k"], ref["n"])
    except (KeyError, TypeError, ValueError):
        return None


def _write_live_marker(work: Path, meta: dict, scenario: Scenario) -> None:
    """Write ``work/run_live.json`` so the run explorer can watch this run WHILE it is still
    in flight — before ``result.json`` (written only at run end) exists.

    ``result.json`` carries the ``phoenix_trace_id`` that joins a run to its conversation, but
    it isn't written until the verdict; mid-run the explorer would have no way to find the
    in-progress trace. This marker publishes that join key (trace id + project) the moment the
    run span opens, plus the BIC reference, so the live view's ΔBIC panel works exactly like the
    archived one. Its presence-without-a-sibling-``result.json`` is also how the explorer tells
    "in progress" from "finished". Best-effort — telemetry must never break a run; the next
    run's ``cleanup()`` wipes ``work/`` and with it any stale marker.
    """
    try:
        ref = scenario.reference()
        bic_ref = _bic_reference(ref)
        marker = {
            **meta,
            "run_id": meta.get("phoenix_trace_id") or f"live-{scenario.name}",
            "scenario": scenario.name,
            "example_version": scenario.version,
            "reference": ref,
            "bic_reference": bic_ref,
            "bic_threshold": scenario.bic_threshold,
        }
        (work / "run_live.json").write_text(json.dumps(marker, indent=2, default=str))
    except Exception:
        pass


def _base_meta(scenario: Scenario, args: argparse.Namespace,
               run_started_at: str | None = None) -> dict:
    """The result-json fields known before ever touching the server — enough to write a
    well-formed degraded result from ANY failure: preflight, baseline scoring, the agent, the
    judge (see ``_drive``, which enriches its own copy with Phoenix ids once its span is open),
    or a teardown-time failure after the agent already finished (see ``_main_async``'s own
    guard). Kept as one function so both call sites can never drift out of sync on the shape
    ``collect_metrics.py`` expects.

    ``run_started_at`` is stamped fresh when not given; ``_main_async`` passes the REAL start
    time it captured before launching the server, so a teardown-time degraded result preserves
    the true start instead of re-stamping it at abort time (which made aborted runs show a
    zero-length duration)."""
    scripted = args.driver == "scripted"
    return {
        "driver": args.driver,
        "model": None if scripted else args.model,
        "git_sha": _git_sha(scenario.example_dir), "example_version": scenario.version,
        "recursion_limit": None if scripted else args.recursion_limit,
        "n_starts": args.n_starts,
        "temperature": None if scripted else args.temperature,
        "run_started_at": run_started_at or _now_iso(),
        "phoenix_project": None, "phoenix_trace_id": None,
    }
