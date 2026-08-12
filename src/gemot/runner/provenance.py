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


def _git_diff_digest(path: Path) -> dict | None:
    """Fingerprint of the UNCOMMITTED working tree, or ``None`` on a clean/absent checkout.

    ``git_sha`` alone does not pin the code. Every run of the Aug 8-10 sweep recorded
    ``732ee74+dirty``, so three array jobs a day apart were indistinguishable in the record even
    though their measured tool-failure rates differed materially (17% -> 7% within one model). A
    whole class of question -- "is this a model effect or a code change?" -- was unanswerable after
    the fact, and the ``+dirty`` suffix was the only hint that anything was missing.

    Stores a DIGEST rather than the diff: the patch can be megabytes and would land in every
    result.json, while a sha256 is enough to tell two runs apart and to match a run against a
    working tree you still have. ``lines``/``files`` make an accidental huge diff visible.
    Best-effort, like :func:`_git_sha` -- provenance must never break a run.
    """
    import hashlib

    try:
        diff = subprocess.check_output(
            ["git", "diff", "HEAD", "--ignore-submodules"], cwd=str(path),
            stderr=subprocess.DEVNULL,
        )
    except Exception:  # noqa: BLE001 - no git, detached tree, anything
        return None
    if not diff.strip():
        return None
    # File names are derived from the diff bytes ALREADY captured, not from a second `git diff`:
    # running it twice recomputes the whole patch, and if the tree changes between the two calls the
    # digest and the file list describe different states. Split on LINES -- `.split()` on whitespace
    # turned a path containing a space (`slurm/trace review/x.py`) into two entries, inflating
    # `n_files`, the one field whose job is making an accidental huge diff visible.
    names = set()
    for line in diff.decode(errors="replace").splitlines():
        for prefix in ("+++ b/", "--- a/"):
            if line.startswith(prefix):
                names.add(line[len(prefix):])
    files = sorted(names)
    return {
        "sha256": hashlib.sha256(diff).hexdigest(),
        "lines": diff.count(b"\n"),
        # Truncated for the record, but n_files counts them ALL -- so a 500-file diff cannot look
        # like a 50-file one.
        "files": files[:50],
        "n_files": len(files),
    }


def _omnipath_provenance() -> dict | None:
    """Fetch date + row counts of the local OmniPath snapshot, or ``None`` if it was not used.

    ``None`` means no snapshot was on disk, so OmniPath came from the hosted service and this run's
    biology is not pinned -- which is the thing worth being able to tell apart afterwards. The
    snapshot is deliberately NOT an ablation (it changes where an answer comes from, not what the
    answer is); this is pure provenance, and best-effort like the git helpers above, because
    provenance must never break a run.
    """
    try:
        from ..omnipath_local import available, snapshot_info
        if not available():
            return None
        info = snapshot_info()
        tables = info.get("tables") or {}
        return {"fetched_at": info.get("fetched_at"),
                "rows": {name: meta.get("rows") for name, meta in tables.items()}}
    except Exception:  # noqa: BLE001 - any import/IO problem degrades to "unrecorded", not a crash
        return None


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
    zero-length duration).

    ``prompt_version`` rides here, next to ``git_sha``, because the prompts are as much an input
    to a run as the code is. It used to reach only Phoenix's experiment metadata, so every run
    outside Phoenix — every ``run.py``, every slurm sweep — recorded no trace of which prompt
    text produced it. ``prompt_group_versions`` makes a bump attributable to lead vs analysts vs
    tools vs control instead of one opaque number moving. Imported from ``prompts`` rather than
    ``experiments`` to keep the runner free of the (heavier, Phoenix-facing) module."""
    from petab_mcp.skills import skills_version

    from .. import ablations

    from ..prompts import prompt_group_versions, prompt_version

    scripted = args.driver == "scripted"
    return {
        "driver": args.driver,
        "model": None if scripted else args.model,
        "git_sha": _git_sha(scenario.example_dir), "example_version": scenario.version,
        # What `git_sha`'s `+dirty` suffix only hinted at: see `_git_diff_digest`. None when clean.
        "git_diff": _git_diff_digest(scenario.example_dir),
        "prompt_version": prompt_version(),
        "prompt_group_versions": prompt_group_versions(),
        # the skills are instructions the panel reads (prompts step 0), so they are
        # part of "what produced this run" just as much as the prompts are.
        "skills_version": skills_version(),
        # WHICH ARM. An ablation changes what a run does while leaving `git_sha` and (for the
        # switches that touch no prompt text) `prompt_version` identical, so without these two
        # fields the halves of an A/B are indistinguishable in the record and pool into one cell
        # that looks like a result. `ablation_arm` is the flat, groupable name -- "baseline", or
        # the deviations -- and rides in the metrics CSV; `ablations` is the full resolved state,
        # every switch recorded rather than only the non-default ones, so a silent absence can
        # never have to be read as "probably the default". See petab_harness.ablations.
        "ablation_arm": ablations.arm(),
        "ablations": ablations.state(),
        # Which OmniPath answered, and from when. A pinned snapshot that nothing records is the
        # `git_sha +dirty` problem in a new costume; None when the run used the hosted service.
        "omnipath_snapshot": _omnipath_provenance(),
        "recursion_limit": None if scripted else args.recursion_limit,
        "n_starts": args.n_starts,
        "temperature": None if scripted else args.temperature,
        "run_started_at": run_started_at or _now_iso(),
        "phoenix_project": None, "phoenix_trace_id": None,
    }
