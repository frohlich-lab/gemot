"""Run and trace analysis: the shared primitives behind every sweep report and trace review.

One implementation of what every analysis script needs — load ``result.json`` + ``combo.env`` per
task, price a run, bucket why it stopped, read the version store and ``provenance.jsonl``, stream
spans out of the Phoenix trace DB — so that ``slurm/sweep_watch.py``, ``slurm/sweep_heatmap.py``,
``slurm/recover_run.py`` and the ``slurm/trace_review/`` extractors agree on every number.
``slurm/ANALYSIS.md`` maps each question to the script that answers it; findings live in
``slurm/trace_review/FINDINGS.md`` and ``examples/BENCHMARK_NOTES.md``.

Import-light by contract (``tests/test_import_lightness.py``): stdlib only at module scope, so
everything here runs on a login node with nothing but the harness installed.

**Usage.** For a sweep submitted by ``slurm/make_sweep.py``, from the repo root::

    python -m petab_harness.analysis summary <job> [<job> ...] [--by model|scenario] [--results-dir D]
    python -m petab_harness.analysis rows    <job> [<job> ...] [--results-dir D] > sweep.csv
    python -m petab_harness.analysis live    [<job>] [--work-root D]

* ``summary`` — how the sweep did and what it cost. One line per model (per example with
  ``--by scenario``): runs, how many are scoreable, how many passed, success rate, median and best
  Δ against the true model, mean and total $ as billed (:func:`run_cost`; ``-`` until the harness
  records it, #178), cache-read fraction, LLM calls. Below it, each line's stop buckets
  (:data:`STOP_CLASSES`) and the median wall-clock split — wall, llm, compute, queue, compile, fit —
  with how many runs waited on a fit queue. Success rate and Δ
  are over **scoreable** runs only (:func:`is_scoreable`): an infrastructure abort before any
  work, an unreachable server, or a task that left no ``result.json`` is counted in its bucket,
  never averaged in as a failure. The Δ is the criterion each run was GRADED on
  (:func:`score_delta`), which is ΔBIC unless the sweep set another. Several job ids merge into one
  table.
* ``rows`` — one CSV row per task, tasks that left only a ``combo.env`` included, for a spreadsheet
  or pandas: task, scenario, model, success, the deciding Δ and the ``criterion`` it is on beside
  canonical ``delta_bic``, stop bucket and the raw ``stop_reason``, cost,
  the llm / compute / queue split, where it ran (``server_mode``, ``workspace_id``), ``n_ops``, the
  #166 reasoning-runaway counters (a run that scored well with ``reasoning_runaways`` high was
  rescued by the retry ladder, not by the model), ``max_output_tokens`` and ``ablation_state``.
* ``live`` — the sweep right now, mid-flight. ``result.json`` only appears when a task exits, so a
  two-day array is otherwise opaque for two days; this reads every in-flight task's
  ``run_live.json`` and ``provenance.jsonl`` off shared storage and prints best-so-far BIC and ΔBIC
  per task, which are already under the bar, and the median. A pure file scan — no Phoenix, no
  network — so poll it as often as you like. Without a job id it lists every live run under every
  known work root (:func:`work_roots`); ``--work-root`` restricts the search to one.

``summary`` and ``live`` exit 1 when there is nothing to report yet.
"""

from __future__ import annotations

import csv
import glob
import json
import math
import os
import sqlite3
import statistics
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from . import paths

# ---------------------------------------------------------------------------------------------
# cost
# ---------------------------------------------------------------------------------------------

def run_cost(token_usage: dict | None) -> float | None:
    """Dollars for one run, as the provider billed it: ``token_usage["cost_usd"]``.

    Recorded, not estimated: the driver asks OpenRouter for the bill on every call and
    ``token_usage`` sums it (#178). A hand-kept price table used to sit here and price every run
    from its token counts; it carried no date, went stale unnoticed, and was removed in #170. A
    run archived before the figure was recorded, or one whose provider reported none, reads
    ``None`` — never 0.0, which would read as "free" in a mean — and its cost column prints ``-``.
    """
    cost = (token_usage or {}).get("cost_usd")
    try:
        return float(cost) if cost is not None else None
    except (TypeError, ValueError):
        return None


def cache_fraction(rows: Iterable[dict]) -> float | None:
    """Share of prompt tokens that were cache reads, pooled over ``rows``. ``None`` with no prompt."""
    prompt = cached = 0
    for r in rows:
        tu = r.get("token_usage") or {}
        prompt += tu.get("prompt_tokens") or 0
        cached += tu.get("cache_read_tokens") or 0
    return (cached / prompt) if prompt else None


# ---------------------------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------------------------

def default_results_root() -> Path:
    """Where a sweep's per-task results land when ``make_sweep.py`` is run with its defaults.

    ``make_sweep.py --out-dir`` moves it; pass the sweep's ``results/`` directory explicitly then.
    """
    return paths.repo_root() / "slurm" / "runs" / "results"


def default_work_root() -> Path:
    """Where sweep tasks stage their working directories.

    ``array_task.sh`` stages under ``$PETAB_MCP_SCRATCH`` when set, else ``slurm/runs/work``.
    ``PETAB_MCP_WORK_ROOT`` is honoured too: the shared-server refactor names it that.
    """
    override = os.environ.get("PETAB_MCP_SCRATCH") or os.environ.get("PETAB_MCP_WORK_ROOT")
    return Path(override) if override else paths.repo_root() / "slurm" / "runs" / "work"


def work_roots() -> tuple[Path, ...]:
    """Every root a task's working directory may sit under, most current first.

    A store is looked for in all of them because a sweep's trees outlive the layout they were
    written with: ``slurm/runs/work`` is the current default, ``slurm/work`` is where the
    shared-server refactor stages, and ``.slurm_scratch`` holds the pre-2026-08-16 trees. The
    version-store check in :func:`find_work_dir` is what makes searching all of them safe.
    """
    repo = paths.repo_root()
    candidates = [default_work_root(), repo / "slurm" / "runs" / "work",
                  repo / "slurm" / "work", repo / ".slurm_scratch"]
    seen: dict[Path, None] = {}
    for c in candidates:
        seen.setdefault(c, None)
    return tuple(seen)


def read_combo_env(path: str | Path) -> dict[str, str]:
    """The ``key=value`` lines of a task's ``combo.env`` as a dict (``{}`` if unreadable)."""
    out: dict[str, str] = {}
    try:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            key, sep, value = line.partition("=")
            if sep:
                out[key.strip()] = value.strip()
    except OSError:
        pass
    return out


def _load_json(path: str | Path) -> dict | None:
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def load_sweep_results(jobs: Iterable[str | int] | str | int, results_root: str | Path | None = None,
                       *, include_missing: bool = False) -> list[dict]:
    """One row per task of the given Slurm array job(s): ``result.json`` joined with ``combo.env``.

    Each row is the run's ``result.json`` dict plus ``task`` (``<job>_<index>``), ``job``, ``index``
    and ``combo`` (the parsed ``combo.env``, ``{}`` if absent). Rows are ordered by job then index,
    so a plot's columns are stable between calls.

    ``include_missing`` also yields a row for a task that left a ``combo.env`` but NO ``result.json``
    — a run killed before its verdict (Slurm TIMEOUT, OOM, a crash), or on shared-server sweeps one
    that could not reach the server at all (exit 97, ``reason=petab_server_unreachable``). Such a
    row has ``result_missing: True`` and a ``stop_reason`` synthesized from the combo's exit code,
    so a summary can count what a sweep lost instead of silently shrinking. :func:`stop_class`
    separates the two cases (``no_result`` vs ``server_unreachable``). Off by default because a
    chart of outcomes should not draw them as outcomes.
    """
    root = Path(results_root) if results_root is not None else default_results_root()
    job_list = [jobs] if isinstance(jobs, (str, int)) else list(jobs)
    rows: list[dict] = []
    for job in job_list:
        for task_dir in sorted(root.glob(f"{job}_*"), key=_task_sort_key):
            if not task_dir.is_dir():
                continue
            combo = read_combo_env(task_dir / "combo.env")
            result = _load_json(task_dir / "result.json")
            if result is None:
                if not include_missing or not combo:
                    continue
                result = {
                    "result_missing": True,
                    "scenario": combo.get("example"),
                    "model": combo.get("model"),
                    "success": False,
                    "delta_bic": None,
                    "stop_reason": combo.get("reason") or f"no result.json (exit {combo.get('exit_code', '?')})",
                }
            index = task_dir.name.rsplit("_", 1)[-1]
            rows.append({**result, "task": task_dir.name, "job": str(job), "index": index,
                         "combo": combo})
    return rows


def _task_sort_key(path: Path) -> tuple:
    head, _, tail = path.name.rpartition("_")
    return (head, int(tail)) if tail.isdigit() else (path.name, 0)


def load_archive_results(archive_root: str | Path | None = None) -> list[dict]:
    """Every archived run's ``result.json`` (``examples/_runs_archive/<run_id>/``), plus ``run_id``
    and ``run_dir``. Newest first by ``run_started_at``.

    The archive is keyed by the run's Phoenix trace id rather than by sweep task; ``result.json``
    carries ``archive_dir`` in the other direction, so the two can be joined either way.
    """
    root = Path(archive_root) if archive_root is not None else paths.runs_archive_root()
    rows: list[dict] = []
    if not root.is_dir():
        return rows
    for run_dir in root.iterdir():
        result = _load_json(run_dir / "result.json")
        if result is not None:
            rows.append({**result, "run_id": run_dir.name, "run_dir": str(run_dir)})
    rows.sort(key=lambda r: str(r.get("run_started_at") or ""), reverse=True)
    return rows


# ---------------------------------------------------------------------------------------------
# classifying outcomes
# ---------------------------------------------------------------------------------------------

#: The buckets :func:`stop_class` sorts a run into, in the order a report should list them.
STOP_CLASSES = (
    "completed",          # the agent finished through the gate
    "early_stop",         # the server's plateau / streak stop fired
    "budget",             # wall-time, token, or recursion budget — a genuine outcome, counted
    "no_model",           # never committed a fittable model
    "content_filter",     # the provider refused the prompt — deterministic, not retried
    "reconstructed",      # rebuilt from a killed run's version store (slurm/recover_run.py)
    "server_unreachable", # the task never reached the petab-mcp server (exit 97, shared-server sweeps)
    "no_result",          # the task left a combo.env but no result.json: killed before its verdict
    "infra_partial",      # infrastructure abort AFTER real modeling work — a valid partial result
    "infra_noop",         # infrastructure abort before any work — scoring noise, excluded
    "other",
)


def stop_class(row: dict) -> str:
    """Which :data:`STOP_CLASSES` bucket this run belongs in.

    Built on ``experiments._is_infra_abort`` / ``_did_modeling_work`` so this and the Phoenix
    experiment summary can never disagree about what counts as infrastructure noise. The one rule
    that matters for aggregation: an infrastructure abort that did modeling work is a PARTIAL
    result and stays in; only an abort before any work is dropped (:func:`is_scoreable`).
    """
    from .experiments import _did_modeling_work, _is_infra_abort

    reason = str(row.get("stop_reason") or "").strip()
    low = reason.lower()
    if "petab_server_unreachable" in low or row.get("combo", {}).get("exit_code") == "97":
        return "server_unreachable"
    if row.get("result_missing"):
        return "no_result"
    if row.get("reconstructed") or low.startswith("recovered_from_incomplete_run"):
        return "reconstructed"
    if low in ("", "completed"):
        return "completed"
    if low.startswith("early_stop"):
        return "early_stop"
    if "wall_time_budget" in low or "token_budget" in low or "recursion" in low:
        return "budget"
    if low == "no_model_built":
        return "no_model"
    if "content" in low and ("filter" in low or "flagged" in low or "refus" in low):
        return "content_filter"
    if _is_infra_abort(reason):
        # `_did_modeling_work` reads `n_ops`, which the Phoenix experiment rows carry; a judged
        # result.json records the same fact as `head_version` (-1 = nothing committed).
        work = dict(row)
        head = row.get("head_version")
        if "n_ops" not in work and isinstance(head, int) and not isinstance(head, bool):
            work["n_ops"] = max(0, head + 1)
        return "infra_partial" if _did_modeling_work(work) else "infra_noop"
    return "other"


def score_delta(row: dict) -> float | None:
    """The Δ that DECIDED this run's verdict: ``delta_criterion`` where the run recorded one, else
    ``delta_bic``.

    Since #167 a run can be graded under CAIC/AIC/AICc/HQC/TIC, and the judge records canonical
    ``delta_bic`` for every run whatever the gate — so on a non-BIC arm ``delta_bic`` is a number
    nobody was graded on. Results that predate #167 carry only ``delta_bic``, which was the gate
    then. ``None`` when neither is a finite number.
    """
    for key in ("delta_criterion", "delta_bic"):
        value = row.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            return float(value)
    return None


def is_scoreable(row: dict) -> bool:
    """Should this run count toward success rate and Δ? A scored verdict exists and the run was
    not pure infrastructure noise."""
    if score_delta(row) is None:
        return False
    # A DECLINED verdict, not a failed one: `slurm/recover_run.py` writes `success: null` for a
    # run graded under a criterion it cannot reproduce.
    if row.get("success") is None:
        return False
    return stop_class(row) not in ("infra_noop", "server_unreachable", "no_result")


# ---------------------------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------------------------

def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def summarize_by(rows: Iterable[dict], key: str = "model") -> list[dict]:
    """One summary row per distinct ``row[key]``: counts, success, ΔBIC, cost, and stop buckets.

    ``success_rate`` and ``median_delta`` are over the SCOREABLE runs (:func:`is_scoreable`) so an
    out-of-credits abort at n_ops=0 does not read as a modeling failure; ``n`` and ``stops`` cover
    every row, so the noise is still visible. Costs are summed over runs that recorded usage.

    The Δs are each run's DECIDING one (:func:`score_delta`), so a group whose runs were graded
    under different criteria has a median across scales — group by the arm before reading it.
    """
    groups: dict[Any, list[dict]] = {}
    for r in rows:
        groups.setdefault(r.get(key) or "?", []).append(r)
    out: list[dict] = []
    for group, runs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        scoreable = [r for r in runs if is_scoreable(r)]
        deltas = [d for d in (score_delta(r) for r in scoreable) if d is not None]
        costs = [c for c in (run_cost(r.get("token_usage")) for r in runs)
                 if c is not None]
        stops: dict[str, int] = {}
        for r in runs:
            stops[stop_class(r)] = stops.get(stop_class(r), 0) + 1
        out.append({
            key: group,
            "n": len(runs),
            "n_scoreable": len(scoreable),
            "n_success": sum(1 for r in scoreable if r.get("success")),
            "success_rate": (sum(1 for r in scoreable if r.get("success")) / len(scoreable)
                             if scoreable else None),
            "median_delta": _median(deltas),
            "best_delta": min(deltas) if deltas else None,
            "total_cost": sum(costs) if costs else None,
            "mean_cost": (sum(costs) / len(costs)) if costs else None,
            "cache_fraction": cache_fraction(runs),
            "llm_calls": sum((r.get("token_usage") or {}).get("n_llm_calls") or 0 for r in runs),
            "stops": dict(sorted(stops.items())),
        })
    return out


def timing_split(row: dict) -> dict[str, float | None]:
    """Where one run's wall clock went, as the judge accounted it.

    ``llm`` is the model thinking, ``compute`` the server working on its behalf (compile + fit),
    ``queue`` the part of ``compute``'s wall time the fit spent waiting for a cluster worker — a
    number that exists only since fits moved to the shared server, and one the client could not
    have measured itself. ``tool_wall`` is the raw client-side busy interval that ``compute``
    is derived from (``compute = tool_wall - queue``).
    """
    def _f(k: str) -> float | None:
        v = row.get(k)
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    return {
        "wall": _f("wall_seconds"), "llm": _f("llm_seconds"), "compute": _f("compute_seconds"),
        "tool_wall": _f("tool_wall_seconds"), "queue": _f("queue_seconds"),
        "compile": _f("compile_seconds"), "fit": _f("fit_seconds"),
        "n_queue_waits": _f("n_queue_waits"), "n_fits": _f("n_fits"), "n_compiles": _f("n_compiles"),
    }


def timing_summary(rows: Iterable[dict]) -> dict[str, float | None]:
    """Medians of :func:`timing_split` over ``rows`` (in seconds), ignoring runs without a value."""
    splits = [timing_split(r) for r in rows]
    out: dict[str, float | None] = {}
    for k in ("wall", "llm", "compute", "tool_wall", "queue", "compile", "fit"):
        out[f"median_{k}"] = _median([s[k] for s in splits if s[k] is not None])
    out["runs_with_queue_wait"] = sum(1 for s in splits if (s["queue"] or 0) > 0)
    return out


def format_table(rows: list[dict], columns: list[tuple[str, str, str]]) -> str:
    """A plain-text table. ``columns`` are ``(key, header, format_spec)``; ``None`` prints ``-``."""
    def cell(row: dict, key: str, spec: str) -> str:
        v = row.get(key)
        if v is None:
            return "-"
        if isinstance(v, dict):
            return " ".join(f"{k}={n}" for k, n in v.items())
        try:
            return format(v, spec) if spec else str(v)
        except (TypeError, ValueError):
            return str(v)

    table = [[h for _k, h, _s in columns]]
    table += [[cell(r, k, s) for k, _h, s in columns] for r in rows]
    widths = [max(len(line[i]) for line in table) for i in range(len(columns))]
    return "\n".join("  ".join(c.ljust(w) for c, w in zip(line, widths)).rstrip()
                     for line in table)


# ---------------------------------------------------------------------------------------------
# the version store and the provenance log
# ---------------------------------------------------------------------------------------------

def store_dir(run_dir: str | Path) -> Path:
    """The ``.petab_mcp`` store for a run directory, whether given the run dir or the store itself."""
    p = Path(run_dir)
    return p if p.name == ".petab_mcp" else p / ".petab_mcp"


def iter_versions(run_dir: str | Path) -> list[dict]:
    """Every version's ``meta.json`` under a run's store, sorted by version number."""
    metas: list[dict] = []
    for vdir in glob.glob(str(store_dir(run_dir) / "versions" / "v*")):
        meta = _load_json(Path(vdir) / "meta.json")
        if meta is not None:
            meta.setdefault("version", _vnum(Path(vdir).name))
            metas.append(meta)
    metas.sort(key=lambda m: _vnum(str(m.get("version"))))
    return metas


def _vnum(name: str) -> int:
    digits = "".join(ch for ch in str(name) if ch.isdigit())
    return int(digits) if digits else -1


def bic_of(meta: dict, n_data_points: int | None) -> float | None:
    """A version's BIC: its own stored value, else recomputed as ``2·nll + k·ln(n)`` from its
    negative log-likelihood and fitted-parameter count — the same formula the reference BIC is
    formed with, so the two are comparable. ``None`` when neither is possible."""
    bic = meta.get("bic")
    if bic is not None:
        try:
            return float(bic)
        except (TypeError, ValueError):
            pass
    nll = meta.get("neg_log_likelihood")
    if nll is None or not n_data_points:
        return None
    k = len(meta.get("fitted_parameters") or {})
    return 2.0 * float(nll) + k * math.log(n_data_points)


def best_fitted_version(run_dir: str | Path, n_data_points: int | None = None
                        ) -> tuple[float, dict, int] | None:
    """``(bic, meta, n_fitted)`` for the lowest-BIC REAL fit in a run's store, or ``None``.

    Only versions with ``fit_status == "fit"`` and a finite nll count: a simulate-only pass or a
    rollback snapshot records a likelihood too, and taking it would score a model nobody optimized.
    This is the arithmetic ``slurm/recover_run.py`` rebuilds a killed run's score from.
    """
    best: tuple[float, dict] | None = None
    n_fitted = 0
    for meta in iter_versions(run_dir):
        if meta.get("fit_status") != "fit" or meta.get("neg_log_likelihood") is None:
            continue
        bic = bic_of(meta, n_data_points)
        if bic is None or not math.isfinite(bic):
            continue
        n_fitted += 1
        if best is None or bic < best[0]:
            best = (bic, meta)
    return (best[0], best[1], n_fitted) if best else None


def provenance_events(run_dir: str | Path, types: Iterable[str] | None = ("fit", "alteration")
                      ) -> list[dict]:
    """The events of a run's ``provenance.jsonl``, optionally filtered by ``type``.

    ``fit`` is the versioned op since issue #81 made every version a fit; older traces recorded it
    as ``alteration`` (edit and fit were one step). The default keeps both so a digest spans eras.
    """
    wanted = set(types) if types is not None else None
    events: list[dict] = []
    try:
        lines = (store_dir(run_dir) / "provenance.jsonl").read_text(encoding="utf-8").splitlines()
    except OSError:
        return events
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if isinstance(rec, dict) and (wanted is None or rec.get("type") in wanted):
            events.append(rec)
    return events


def clip(value: Any, n: int = 280) -> str | None:
    """``value`` as a string of at most ``n`` characters (JSON for non-strings), ``None`` for None.

    The one truncation rule every digest and dossier uses, so an analyst reading two of them is
    reading the same thing.
    """
    if value is None:
        return None
    s = value if isinstance(value, str) else json.dumps(value, default=str)
    return s if len(s) <= n else s[:n] + f"…[+{len(s) - n} chars]"


def op_digest(event: dict) -> dict:
    """One versioned provenance event -> the compact op record the trace reviews read.

    Keeps the agent's stated ``justification``, whether the fit was a new best and by how much, the
    nll / χ² it reached, and a SHAPE summary of the parameter diff (counts, not contents).
    """
    pd = event.get("parameter_diff") or {}
    pd_summary: Any
    if isinstance(pd, dict):
        pd_summary = {k: len(v) if isinstance(v, (list, dict)) else v for k, v in pd.items()}
    else:
        pd_summary = clip(pd, 120)
    op = event.get("operation")
    if op is None:
        # A fit event has no single `operation`; it scores the edits it consumed.
        edits = event.get("staged_edits") or []
        op = "+".join(str(e.get("operation", "?")) for e in edits) or "fit"
    return {
        "version": event.get("version"),
        "op": op,
        "justification": clip(event.get("justification")),
        "is_new_best": event.get("is_new_best"),
        "improvement_vs_best_prior": event.get("improvement_vs_best_prior"),
        "nll": event.get("neg_log_likelihood"),
        "chi2": event.get("chi2"),
        "fit_status": event.get("fit_status"),
        "warnings": clip(event.get("warnings"), 200),
        "param_diff": pd_summary,
        "self_improvement": clip(event.get("self_improvement"), 200),
    }


def run_digest(run_dir: str | Path) -> dict:
    """A run's op sequence plus the waste signals the trace reviews key on.

    ``longest_non_improving_streak`` is the longest run of consecutive versioned ops that were not a
    new best — the number the early-stop chain was built to bound. ``n_fits`` counts ops whose
    ``fit_status`` is ``"fit"`` (a real optimizer run), not ops.
    """
    ops = [op_digest(e) for e in provenance_events(run_dir)]
    longest = current = 0
    for op in ops:
        if op.get("is_new_best"):
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return {
        "n_ops": len(ops),
        "n_new_best": sum(1 for o in ops if o.get("is_new_best")),
        "n_fits": sum(1 for o in ops if o.get("fit_status") == "fit"),
        "longest_non_improving_streak": longest,
        "ops": ops,
    }


#: The depths a task's working directory sits at, relative to a work root. Shared by
#: :func:`find_work_dir` and :func:`live_rows` so the two can never disagree about where a run is.
_WORK_DEPTHS = (("{tag}", "*", "work"), ("{tag}", "*", "*", "work"), ("*", "{tag}", "*", "work"))


def find_work_dir(tag: str, results_root: str | Path | None = None,
                  work_root: str | Path | None = None) -> Path | None:
    """Where a sweep task's version store survives, for a run that never wrote ``result.json``.

    Looks, in order, at: the archive directory its ``result.json`` names (if one exists and holds
    versions); a copy preserved under the task's results directory; then the working directory
    under each of :func:`work_roots`, at every depth a sweep layout puts it
    (``<tag>/<example>/work``, ``<tag>/examples/<example>/work``, ``<sweep>/<tag>/<example>/work``).
    ``None`` when no store with at least one version is found — an EMPTY ``versions/`` must never
    win over a tree that still holds the whole history, which is also what makes it safe to search
    every root and depth rather than guessing the layout.

    Pass ``work_root`` to search exactly one root instead.
    """
    results = Path(results_root) if results_root is not None else default_results_root()
    roots = (Path(work_root),) if work_root is not None else work_roots()

    def has_versions(d: str | Path) -> bool:
        return bool(glob.glob(str(store_dir(d) / "versions" / "v*")))

    result = _load_json(results / tag / "result.json") or {}
    archive = result.get("archive_dir")
    if archive and has_versions(archive):
        return Path(archive)
    if has_versions(results / tag):
        return results / tag
    for root in roots:
        for parts in _WORK_DEPTHS:
            pattern = os.path.join(str(root), *(p.format(tag=tag) for p in parts))
            hits = [h for h in sorted(glob.glob(pattern)) if has_versions(h)]
            if hits:
                return Path(hits[0])
    return None


def find_live_runs(job: str | int = "*", work_root: str | Path | None = None) -> list[tuple[str, Path, dict]]:
    """``(tag, work_dir, run_live)`` for every task of ``job`` that has started.

    ``run_live.json`` is written into the working directory at task START and carries the scenario,
    the reference BIC and the pass threshold, so it exists for the whole run — unlike
    ``result.json``, which appears only when the task exits. Sorted by task index.
    """
    roots = (Path(work_root),) if work_root is not None else work_roots()
    out: dict[str, tuple[str, Path, dict]] = {}
    for root in roots:
        for parts in _WORK_DEPTHS:
            pattern = os.path.join(str(root), *(p.format(tag=f"{job}_*") for p in parts),
                                   "run_live.json")
            for path in sorted(glob.glob(pattern)):
                live = _load_json(path)
                if live is None:
                    continue
                tag = _tag_from_path(path)
                if tag and tag not in out:
                    out[tag] = (tag, Path(path).parent, live)
    return [out[t] for t in sorted(out, key=lambda t: _task_sort_key(Path(t)))]


def _tag_from_path(path: str | Path) -> str | None:
    """The ``<job>_<task>`` component of a work-dir path, whatever depth it sits at."""
    for part in Path(path).parts:
        head, _, tail = part.rpartition("_")
        if head.isdigit() and tail.isdigit():
            return part
    return None


def live_rows(job: str | int = "*", work_root: str | Path | None = None) -> list[dict]:
    """Best-so-far ΔBIC per in-flight task, from SHARED STORAGE ONLY — no Phoenix, no network.

    Each row carries ``task``, ``scenario``, ``n_fits``, ``best_bic``, ``bic_reference``,
    ``bic_threshold``, ``delta_bic`` and ``passing``. Read from the run's ``provenance.jsonl``
    ``fit`` events, which record ``bic`` and ``neg_log_likelihood`` at the TOP level of the event
    (not under ``result``), falling back to the version store when a run has no provenance log.

    A pure local file scan, so it can be polled as often as wanted. The Phoenix route needs the
    collector and does not scale to a multi-GB store; ``bic`` is in scratch all along.
    """
    rows = []
    for tag, work, live in find_live_runs(job, work_root):
        best, n_fits = None, 0
        for ev in provenance_events(work, types=("fit",)):
            bic = ev.get("bic")
            if bic is None or not isinstance(bic, (int, float)) or not math.isfinite(bic):
                continue
            n_fits += 1
            best = bic if best is None else min(best, bic)
        if best is None:                                    # no provenance log: use the store
            found = best_fitted_version(work, live.get("n_data_points"))
            if found:
                best, _, n_fits = found
        ref, thr = live.get("bic_reference"), live.get("bic_threshold")
        delta = (best - ref) if (best is not None and ref is not None) else None
        rows.append({
            "task": tag, "work_dir": str(work), "scenario": live.get("scenario"),
            "n_fits": n_fits, "best_bic": best, "bic_reference": ref, "bic_threshold": thr,
            "delta_bic": delta,
            "passing": None if (delta is None or thr is None) else delta <= thr,
        })
    return rows


# ---------------------------------------------------------------------------------------------
# the Phoenix trace database
# ---------------------------------------------------------------------------------------------

#: The run identity the harness stamps on a trace's root span (``harness.tracing.record_run_result``),
#: as ``petab.<key>`` attributes.
PETAB_RUN_KEYS = ("scenario", "model", "success", "delta_bic", "criterion", "delta_criterion",
                  "bic", "k", "n", "stop_reason", "final_nll", "baseline_nll", "git_sha",
                  "recursion_limit")


def phoenix_connect(db_path: str | Path) -> sqlite3.Connection:
    """A READ-ONLY connection to the collector's SQLite DB, safe to open while it is serving.

    SAME HOST ONLY. SQLite's WAL uses shared memory to coordinate readers, and shared memory does
    not cross hosts: opening the collector's live DB from a different node than the one running it
    (a login node, say) can block in uninterruptible I/O for many minutes and still return zero
    rows. To read a live store from elsewhere, copy ``phoenix.db`` AND ``phoenix.db-wal`` to
    node-local disk and open the copy — see :func:`local_phoenix_copy`.
    """
    conn = sqlite3.connect(f"file:{Path(db_path)}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def local_phoenix_copy(db_path: str | Path, dest_dir: str | Path) -> Path:
    """Copy a live collector DB to ``dest_dir`` and return a path safe to open READ-WRITE.

    Three rules, each of which cost a wasted pull to learn:

    * Copy the ``-wal`` file too. Everything the collector wrote since its last checkpoint lives
      only there; without it the copy is silently stale.
    * Do NOT copy ``-shm``. It is a per-host mmap of WAL coordination state, and a foreign one is
      worse than none.
    * Open the copy read-WRITE. SQLite replays a WAL into the main DB on first open, and it cannot
      do that read-only — a read-only open silently serves the last checkpoint instead, which is
      the same stale answer as forgetting the ``-wal``.

    ``dest_dir`` should be node-local (``$TMPDIR`` on a compute node), which is the point: the
    copy is private, so read-write is safe and the WAL coordination never crosses a host.
    """
    import shutil

    src, dest = Path(db_path), Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / src.name
    shutil.copy2(src, target)
    wal = src.with_name(src.name + "-wal")
    if wal.exists():
        shutil.copy2(wal, dest / wal.name)
    return target


def flatten_attrs(obj: Any, prefix: str = "") -> dict[str, Any]:
    """Span attributes as ONE flat dotted dict.

    Phoenix stores OpenInference attributes sometimes nested (``{"llm": {"output_messages": [...]}}``)
    and sometimes already dotted (``{"llm.output_messages.0.message.content": ...}``), depending on
    the exporter version. Every consumer wants the dotted form, so normalise once here.
    """
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten_attrs(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(flatten_attrs(v, f"{prefix}.{i}" if prefix else str(i)))
    else:
        out[prefix] = obj
    return out


def first_attr(attrs: dict, *keys: str, default: Any = None) -> Any:
    """The first of ``keys`` present and non-empty in ``attrs``, else ``default``."""
    for k in keys:
        if k in attrs and attrs[k] not in (None, ""):
            return attrs[k]
    return default


def _span_row(row: sqlite3.Row) -> dict:
    try:
        raw = json.loads(row["attributes"]) if row["attributes"] else {}
    except ValueError:
        raw = {}
    return {
        "id": row["id"], "trace_rowid": row["trace_rowid"], "parent_id": row["parent_id"],
        "name": row["name"], "span_kind": (row["span_kind"] or "").upper(),
        "status_code": row["status_code"], "status_message": row["status_message"],
        "start_time": row["start_time"], "attributes": flatten_attrs(raw),
    }


def trace_rowids(conn: sqlite3.Connection, trace_ids: Iterable[str]) -> dict[str, int]:
    """``trace_id -> trace_rowid`` for the given OTel trace ids, via the indexed ``traces`` table.

    The rowid is what ``spans.trace_rowid`` (and its index) is keyed on, so resolving ids first
    turns a full-table scan into an index seek — the difference between minutes and milliseconds
    on a multi-GB store. ``result.json`` / ``run_live.json`` carry the trace_id, not the rowid.
    """
    ids = [str(t) for t in trace_ids if t]
    if not ids:
        return {}
    out: dict[str, int] = {}
    for start in range(0, len(ids), 500):        # keep well under SQLITE_MAX_VARIABLE_NUMBER
        chunk = ids[start:start + 500]
        placeholders = ",".join("?" * len(chunk))
        for row in conn.execute(
                f"select trace_id, id from traces where trace_id in ({placeholders})", chunk):
            out[row["trace_id"]] = row["id"]
    return out


def iter_spans(conn: sqlite3.Connection, *, kind: str | None = None,
               trace_rowid: int | None = None, trace_ids: Iterable[str] | None = None,
               name_like: str | None = None, roots_only: bool = False) -> Iterator[dict]:
    """Stream spans as dicts with FLATTENED attributes, in start order.

    Filters are pushed into SQL so a 28 GB DB is walked once and only the matching rows are
    decoded: ``kind`` (``"TOOL"``, ``"LLM"``, ...), one ``trace_rowid``, a ``name_like`` SQL pattern,
    ``roots_only`` (``parent_id IS NULL`` — one per trace, where the run identity lives).

    ``trace_ids`` takes OTel trace ids (what ``run_live.json`` and ``result.json`` record) and
    resolves them through :func:`trace_rowids` first, so a KNOWN run list is served from
    ``ix_spans_trace_rowid`` rather than scanned for. An id that no trace matches simply
    contributes nothing; if none of them match, nothing is yielded.
    """
    where, args = [], []
    if kind:
        where.append("upper(span_kind) = ?")
        args.append(kind.upper())
    if trace_rowid is not None:
        where.append("trace_rowid = ?")
        args.append(trace_rowid)
    if trace_ids is not None:
        rowids = sorted(trace_rowids(conn, trace_ids).values())
        if not rowids:
            return
        where.append(f"trace_rowid in ({','.join('?' * len(rowids))})")
        args.extend(rowids)
    if name_like:
        where.append("name LIKE ?")
        args.append(name_like)
    if roots_only:
        where.append("parent_id IS NULL")
    sql = ("select id, trace_rowid, parent_id, name, span_kind, attributes, status_code, "
           "status_message, start_time from spans")
    if where:
        sql += " where " + " and ".join(where)
    sql += " order by start_time"
    for row in conn.execute(sql, args):
        yield _span_row(row)


def petab_runs(conn: sqlite3.Connection) -> dict[int, dict]:
    """``trace_rowid -> run identity`` for every trace whose root span the harness stamped.

    The join key between the trace DB and everything else: ``result.json`` carries the trace id,
    the root span carries ``petab.scenario`` / ``petab.model`` / ``petab.delta_bic`` and friends.
    Traces without a ``petab.scenario`` (a bio-MCP server's own spans, a stray client) are skipped.
    """
    runs: dict[int, dict] = {}
    for span in iter_spans(conn, roots_only=True):
        attrs = span["attributes"]
        ident = {k: attrs.get(f"petab.{k}") for k in PETAB_RUN_KEYS}
        if ident.get("scenario") is None:
            continue
        runs[span["trace_rowid"]] = ident
    return runs


def span_error(span: dict) -> str | None:
    """The error a TOOL/LLM span recorded, or ``None`` for a clean one.

    A tool span with an OK status can still carry a REJECTION in its output (the server answered
    ``applied: false`` with an ``error``); that is a tool outcome, not a span error, and is left to
    the caller to read out of ``output.value``.
    """
    if span.get("status_code") in (None, "OK", "UNSET"):
        return None
    return span.get("status_message") or span.get("status_code")


def tool_output(span: dict) -> Any:
    """A TOOL span's returned value, JSON-decoded when it is JSON text."""
    raw = first_attr(span.get("attributes") or {}, "output.value", "tool.output")
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except ValueError:
            return raw
    return raw


# ---------------------------------------------------------------------------------------------
# a small command line, for the login node
# ---------------------------------------------------------------------------------------------

_SUMMARY_COLUMNS = [
    ("model", "model", ""), ("n", "n", "d"), ("n_scoreable", "scored", "d"),
    ("n_success", "ok", "d"), ("success_rate", "rate", ".2f"), ("median_delta", "med d", ".1f"),
    ("best_delta", "best d", ".1f"), ("mean_cost", "$/run", ".3f"), ("total_cost", "total $", ".2f"),
    ("cache_fraction", "cache", ".2f"), ("llm_calls", "calls", "d"),
]


def _cmd_summary(args) -> int:
    rows = load_sweep_results(args.jobs, args.results_dir, include_missing=True)
    if not rows:
        print("no results yet")
        return 1
    by = summarize_by(rows, args.by)
    cols = [(args.by, args.by, ""), *_SUMMARY_COLUMNS[1:]]
    print(format_table(by, cols))
    print()
    print("stop buckets:")
    for r in by:
        print(f"  {str(r[args.by])[:32]:32s} " + "  ".join(f"{k}={n}" for k, n in r["stops"].items()))
    timing = timing_summary(rows)
    print()
    print("timing (medians, s): " + "  ".join(
        f"{k[len('median_'):]}={v:.0f}" for k, v in timing.items()
        if k.startswith("median_") and v is not None)
          + f"  runs_with_queue_wait={timing['runs_with_queue_wait']}")
    return 0


#: ``rows`` CSV columns taken verbatim off ``result.json``, after the derived ones. The runaway
#: counters are here because the guard's accuracy claim rests on them: a run that scored well with
#: ``reasoning_runaways`` high was rescued by the ladder, not by the model.
_ROW_PASSTHROUGH = ("llm_seconds", "compute_seconds", "queue_seconds", "server_mode",
                    "workspace_id", "n_ops", "reasoning_runaways", "reasoning_runaway_recovered",
                    "reasoning_runaway_retry_errors", "max_output_tokens", "ablation_state")


def _cmd_rows(args) -> int:
    rows = load_sweep_results(args.jobs, args.results_dir, include_missing=True)
    writer = csv.writer(__import__("sys").stdout)
    writer.writerow(["task", "scenario", "model", "success", "delta", "criterion", "delta_bic",
                     "stop_class", "stop_reason", "cost", *_ROW_PASSTHROUGH])
    for r in rows:
        writer.writerow([r.get("task"), r.get("scenario"), r.get("model"), r.get("success"),
                         score_delta(r), r.get("criterion"),
                         r.get("delta_bic"), stop_class(r), r.get("stop_reason"),
                         run_cost(r.get("token_usage")),
                         *(r.get(f) for f in _ROW_PASSTHROUGH)])
    return 0


def _cmd_live(args) -> int:
    rows = live_rows(args.job, args.work_root)
    if not rows:
        print(f"no run_live.json for job {args.job} under "
              + ", ".join(str(r) for r in ([args.work_root] if args.work_root else work_roots()))
              + " — has the sweep started?")
        return 1
    scored = [r for r in rows if r["delta_bic"] is not None]
    print(f"{'task':<16} {'scenario':<12} {'fits':>4} {'best BIC':>11} {'ΔBIC':>9}  pass")
    for r in rows:
        d = r["delta_bic"]
        head = f"{r['task']:<16} {str(r['scenario'] or '-'):<12} {r['n_fits']:>4}"
        if d is None:
            print(f"{head} {'(no fit yet)':>11}")
            continue
        mark = "  <-- IN" if r["passing"] else ""
        print(f"{head} {r['best_bic']:>11.1f} {d:>+9.1f}{mark}")
    if scored:
        thr = next((r["bic_threshold"] for r in scored if r["bic_threshold"] is not None), 0.0)
        under = sum(1 for r in scored if r["delta_bic"] <= thr)
        deltas = sorted(r["delta_bic"] for r in scored)
        print(f"\n{under}/{len(scored)} under ΔBIC {thr:+.0f}, "
              f"median {statistics.median(deltas):+.1f}, best {deltas[0]:+.1f} "
              f"({len(rows) - len(scored)} with no scored fit yet)")
    return 0


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="python -m petab_harness.analysis",
                                description="Sweep summaries from result.json + combo.env.")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("summary", help="per-model (or per-example) outcome, cost and timing table")
    s.add_argument("jobs", nargs="+", help="Slurm array job id(s)")
    s.add_argument("--by", default="model", choices=["model", "scenario"])
    s.add_argument("--results-dir", default=None)
    s.set_defaults(fn=_cmd_summary)
    r = sub.add_parser("rows", help="one CSV row per task, with the stop bucket and cost")
    r.add_argument("jobs", nargs="+")
    r.add_argument("--results-dir", default=None)
    r.set_defaults(fn=_cmd_rows)
    live = sub.add_parser("live", help="best-so-far ΔBIC for an IN-FLIGHT sweep (no Phoenix)")
    live.add_argument("job", nargs="?", default="*", help="Slurm array job id (default: any)")
    live.add_argument("--work-root", default=None,
                      help="search only this work root (default: every root in work_roots())")
    live.set_defaults(fn=_cmd_live)
    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
