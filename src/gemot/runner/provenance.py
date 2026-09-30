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
        sha = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=str(path),
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return None
    try:
        dirty = subprocess.call(
            ["git", "diff", "--quiet", "--ignore-submodules"],
            cwd=str(path),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
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
            ["git", "diff", "HEAD", "--ignore-submodules"],
            cwd=str(path),
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
                names.add(line[len(prefix) :])
    files = sorted(names)
    return {
        "sha256": hashlib.sha256(diff).hexdigest(),
        "lines": diff.count(b"\n"),
        # Truncated for the record, but n_files counts them ALL -- so a 500-file diff cannot look
        # like a 50-file one.
        "files": files[:50],
        "n_files": len(files),
    }


def _server_provenance() -> dict | None:
    """Which petab-mcp code served the run. The server is installed from its own repository, so
    ``git_sha`` does not cover it: record the commit a git install was made from, or an editable
    checkout's own ``git_sha`` and ``git_diff``. ``None`` when the server is not installed.
    Best-effort, like the git helpers above."""
    import urllib.parse
    from importlib.metadata import PackageNotFoundError, distribution

    try:
        dist = distribution("petab-mcp")
    except PackageNotFoundError:
        return None
    try:
        direct = json.loads(dist.read_text("direct_url.json") or "{}")
    except ValueError:
        direct = {}
    out = {"version": dist.version}
    if "vcs_info" in direct:
        out["commit"] = direct["vcs_info"].get("commit_id")
    elif (direct.get("dir_info") or {}).get("editable"):
        checkout = Path(urllib.parse.unquote(urllib.parse.urlparse(direct["url"]).path))
        out["git_sha"] = _git_sha(checkout)
        out["git_diff"] = _git_diff_digest(checkout)
    return out


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
        return {
            "fetched_at": info.get("fetched_at"),
            "rows": {name: meta.get("rows") for name, meta in tables.items()},
        }
    except Exception:  # noqa: BLE001 - any import/IO problem degrades to "unrecorded", not a crash
        return None


def _bic_reference(ref) -> float | None:
    """BIC of the true-model reference, or ``None`` when it is absent or malformed.

    ``scenario.reference()`` catches JSON *parse* errors but not a ``reference.json`` that parses
    yet lacks ``nll``/``k``/``n`` (or carries ``null``). Guard here so an author-side reference
    problem degrades to "no BIC gate" rather than raising a ``KeyError``/``TypeError`` that
    ``_drive``'s broad handler would misreport as an aborted (provider) run.

    CANONICAL BIC, which is what the live marker publishes. The judge grades under the run's own
    criterion via :func:`_reference_score`, so the marker's live ΔBIC panel and the final verdict
    are on the same scale only when BIC is that criterion.
    """
    return _reference_score(ref, None)


def _reference_score(ref, criterion) -> float | None:
    """The true-model reference scored under ``criterion`` — canonical BIC when it is ``None``.

    The reference carries ``nll``/``k``/``n``, from which every criterion but TIC is computable,
    so the judge can grade a candidate under whatever criterion the run searched by; TIC also
    needs the reference fit's ``k_eff``, which :func:`simulate.write_reference_from_problem`
    records and an older ``reference.json`` lacks (``None`` then — no TIC gate, and the judge
    says so). Same fail-soft contract as above: an author-side reference problem degrades to "no
    gate", never an exception.
    """
    if not ref:
        return None
    from petab_mcp.criteria import CANONICAL

    try:
        return (criterion or CANONICAL["bic"]).value(
            ref["nll"], ref["k"], ref["n"], ref.get("k_eff")
        )
    except (KeyError, TypeError, ValueError):
        return None


def _write_live_marker(
    work: Path, meta: dict, scenario: Scenario, *, include_reference: bool = True
) -> None:
    """Write ``work/run_live.json`` so the run explorer can watch this run WHILE it is still
    in flight — before ``result.json`` (written only at run end) exists.

    ``result.json`` carries the ``phoenix_trace_id`` that joins a run to its conversation, but
    it isn't written until the verdict; mid-run the explorer would have no way to find the
    in-progress trace. This marker publishes that join key (trace id + project) the moment the
    run span opens, plus the BIC reference, so the live view's ΔBIC panel works exactly like the
    archived one. Its presence-without-a-sibling-``result.json`` is also how the explorer tells
    "in progress" from "finished". Best-effort — telemetry must never break a run; the next
    run's ``cleanup()`` wipes ``work/`` and with it any stale marker.

    ``include_reference=False`` omits the reference score: the barebones arm has no versions to
    score live, and its agent works beside this file.
    """
    try:
        from petab_mcp.criteria import CANONICAL

        ref = scenario.reference() if include_reference else None
        bic_ref = _bic_reference(ref)
        marker = {
            **meta,
            "run_id": meta.get("phoenix_trace_id") or f"live-{scenario.name}",
            "scenario": scenario.name,
            "example_version": scenario.version,
            "reference": ref,
            "bic_reference": bic_ref,
            "bic_threshold": scenario.bic_threshold,
            # CANONICAL BIC above (no fit exists yet to report a criterion), so the live panel
            # can say which scale it is showing rather than implying it is the gate's.
            "reference_criterion": "bic",
            # ...and the reference under EVERY criterion, so a live view can put the fits of a
            # run that ranks by another one against the right bar. `None` where the reference
            # has no value under it (aicc at n <= k+1; tic until the reference is regenerated
            # with a k_eff).
            "criteria_reference": {
                name: _reference_score(ref, crit) for name, crit in CANONICAL.items()
            },
        }
        (work / "run_live.json").write_text(json.dumps(marker, indent=2, default=str))
    except Exception:
        pass


def _base_meta(
    scenario: Scenario, args: argparse.Namespace, run_started_at: str | None = None
) -> dict:
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

    from ..prompts import BIOCURATOR_ENABLED, prompt_group_versions, prompt_version

    scripted = args.driver == "scripted"
    from ..biology_blinding import read_names

    blind_names = (
        read_names(scenario.example_dir / "work") if ablations.enabled("biology_blind") else None
    )
    uses_bio_tools = args.driver == "barebones" or (
        args.driver == "deepagents" and BIOCURATOR_ENABLED
    )
    return {
        "driver": args.driver,
        "model": None if scripted else args.model,
        "git_sha": _git_sha(scenario.example_dir),
        "example_version": scenario.version,
        # What `git_sha`'s `+dirty` suffix only hinted at: see `_git_diff_digest`. None when clean.
        "git_diff": _git_diff_digest(scenario.example_dir),
        "server_code": _server_provenance(),
        "prompt_version": prompt_version(),
        "prompt_group_versions": prompt_group_versions(),
        # the skills are instructions the panel reads (prompts step 0), so they are
        # part of "what produced this run" just as much as the prompts are.
        "skills_version": None if blind_names is not None else skills_version(),
        # WHICH ARM. An ablation changes what a run does while leaving `git_sha` and (for the
        # switches that touch no prompt text) `prompt_version` identical, so without these two
        # fields the halves of an A/B are indistinguishable in the record and pool into one cell
        # that looks like a result. `ablation_arm` is the flat, groupable name -- "baseline", or
        # the deviations -- and rides in the metrics CSV; `ablations` is the full resolved state,
        # every switch recorded rather than only the non-default ones, so a silent absence can
        # never have to be read as "probably the default". See petab_harness.ablations.
        # `ablation_state` is the third: `ablation_arm` names only the DEVIATIONS, so its meaning
        # moves whenever a default does (after 2026-08-15 "baseline" carries free_exponents=on
        # where it used to carry off). Grouping a sweep that spans such a commit on `arm` alone
        # pools opposite configurations; `ablation_state` spells every switch out and is safe.
        "ablation_arm": ablations.arm(),
        "ablation_state": ablations.state_key(),
        "ablations": ablations.state(),
        # Which OmniPath answered, and from when. A pinned snapshot that nothing records is the
        # `git_sha +dirty` problem in a new costume; None for hosted service or no bio access.
        "omnipath_snapshot": (
            _omnipath_provenance() if uses_bio_tools and blind_names is None else None
        ),
        # Operator-only metadata: persisted with the archive so its anonymous model can
        # still be interpreted after work/ is replaced by the next run.
        "biology_blinding": (
            {"version": 1, "identifiers": blind_names.forward} if blind_names is not None else None
        ),
        "recursion_limit": None if scripted else args.recursion_limit,
        "n_starts": args.n_starts,
        "temperature": None if scripted else args.temperature,
        "run_started_at": run_started_at or _now_iso(),
        "phoenix_project": None,
        "phoenix_trace_id": None,
    }


def _agent_meta(base_meta: dict, agent_out: dict, stop_reason: str) -> dict:
    """:func:`_base_meta` plus EVERYTHING :func:`petab_harness.agent.run_deepagents` reported.

    Spread wholesale rather than copied field by field, so a measurement the driver adds reaches
    ``result.json`` without a second edit here (docs/RATIONALE.md). ``stop_reason`` is restated
    because the caller resolves the driver's default before printing it.
    """
    return {**base_meta, **agent_out, "stop_reason": stop_reason}
