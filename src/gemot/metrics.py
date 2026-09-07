"""Across-run / across-example performance tracking.

Two sources of truth per run:

* ``work/result.json`` (written by :func:`runner._judge`) — the **outcome**: chi2, BIC,
  ΔBIC, the verdict, timings, the model, and the git sha it ran at.
* the **Phoenix trace** — the only place token **usage** lives (summed from the child LLM
  spans), plus span/tool/agent counts.

This module joins the two — preferentially by the ``phoenix_trace_id`` the runner stamps
into ``result.json`` (falling back to the run's time window, then to the latest trace in
the project) — into one flat row per run, appends it to an append-only history CSV so the
numbers can be tracked over time, and renders an across-example summary. It never fails a
run: a missing Phoenix or missing trace just leaves the usage columns blank.

The tracking is thus twofold, matching the two questions:
    * *within a trace* — the run span carries chi2/BIC/verdict/model and Phoenix rolls up
      the trace's token usage (see ``tracing.run_span`` / ``record_run_result``);
    * *across examples* — this aggregator (:func:`collect`, :func:`summarize`).
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import os
from pathlib import Path
from typing import Any

from . import paths

# Outcome columns copied straight from result.json.
_RESULT_FIELDS = (
    "scenario", "example_version", "model", "driver", "success", "stop_reason",
    "baseline_nll", "final_nll", "delta",
    "chi2", "chi2_dof", "chi2_gof_pvalue",
    "k", "n", "k_eff", "bic", "bic_reference", "delta_bic", "bic_threshold",
    "criterion", "criterion_note", "criterion_value", "criterion_reference", "delta_criterion",
    "head_version", "n_errors", "n_tool_errors", "tool_calls",
    # Edits the server REFUSED. Flat columns, not the nested `edit_rejects` dict from result.json:
    # DictWriter has no representation for a dict (it would stringify), and a name missing from
    # this tuple is dropped SILENTLY by extrasaction="ignore" below — which is exactly how this
    # metric was invisible in the history despite being written to every result.json.
    "edits_attempted", "edits_rejected", "edits_fit_reverted", "edit_reject_rate",
    "edit_reject_top_cause",
    "wall_seconds", "compute_seconds", "llm_seconds",
    "compile_seconds", "fit_seconds", "n_compiles", "n_fits",
    "git_sha", "recursion_limit", "n_starts", "temperature",
    # `git_sha` ends in `+dirty` whenever the tree had uncommitted changes, which means it does NOT
    # pin the code: the whole Aug 8-10 sweep recorded one sha across jobs whose behaviour differed.
    # This is the column you group by to tell "different code" from "different model". Flat, for
    # the same reason as the edit_rejects columns above -- result.json's `git_diff` is a dict.
    "git_diff_sha256",
    # The prompts are as much an input to a run as the code is, so the version rides beside
    # git_sha. Without it a sweep spanning a prompt edit pools both versions into one cell with
    # no column to split them on (and `DictWriter(extrasaction="ignore")` below drops silently,
    # so a missing column looks like a missing value). `prompt_group_versions` deliberately does
    # NOT get a column: a flat key is what you group a sweep BY, and the per-group hashes answer
    # the follow-up question — which group moved — which is read from result.json.
    "prompt_version", "skills_version",
    # The arm. Same argument as `prompt_version` above, for the switches that change behaviour
    # WITHOUT changing prompt text (`bio_cache`, `omnipath_local`): those move neither git_sha nor
    # prompt_version, so this is the only column that can split them. Flat and canonical
    # ("baseline" or the deviations) because a flat key is what you group a sweep by; the full
    # per-switch state is the `ablations` dict in result.json, read for the follow-up question.
    "ablation_arm",
    # ...and the default-INDEPENDENT twin, because `ablation_arm` names only the deviations and so
    # changes meaning when a DEFAULT changes: a sweep spanning 2026-08-15 has "baseline" rows with
    # free_exponents on and off. Group by `ablation_arm` for continuity across an added arm, by
    # `ablation_state` when correctness across a default flip matters.
    "ablation_state",
    # Which upstream endpoint actually served the run, flattened out of `openrouter_endpoints`.
    # `openrouter_extra_body` records what routing ASKED for; these record what it GOT, and the two
    # differ whenever `provider.only` names more than one endpoint. `served_drifted` is the column
    # to filter a benchmark on: a run split across two endpoints is not one measurement, and until
    # now nothing in the history could say whether that had happened.
    "served_provider", "served_n_distinct", "served_drifted",
    "run_started_at", "run_ended_at",
    "phoenix_project", "phoenix_trace_id",
)

# The replicate lack-of-fit verdict, flattened out of the nested `residual_structure` block for the
# same reason the edit rejects are: a sweep groups by flat columns. `lof_weak` is the one this exists
# for — "the arm passed" and "the arm passed on too few lack-of-fit degrees of freedom to mean it" are
# different runs, and without a flat column telling them apart means reading traces.
_LOF_FIELDS = ("lof_status", "lof_weak", "lof_df")

# result.json keys whose value is a NESTED dict, carried into the row so the write-time flatteners
# below can expand them. build_row copies only the flat _RESULT_FIELDS, so without this the flatteners
# saw nothing on the collect() path and every column derived from a nested block stayed empty —
# silently, which is the failure mode the _RESULT_FIELDS comment above warns about.
_NESTED_SOURCE_FIELDS = ("edit_rejects", "residual_structure", "git_diff",
                         "openrouter_endpoints")

# Columns derived from the Phoenix trace (token usage + span breakdown).
_PHOENIX_FIELDS = (
    "n_spans", "n_llm_calls", "n_tool_spans", "n_agent_spans", "n_span_errors",
    "tokens_prompt", "tokens_completion", "tokens_total", "trace_model", "phoenix_note",
)

HISTORY_FIELDS = _RESULT_FIELDS + _LOF_FIELDS + _PHOENIX_FIELDS
DEFAULT_HISTORY = paths.metrics_history()


def _parse_iso(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    try:
        return dt.datetime.fromisoformat(value)
    except Exception:
        return None




def phoenix_trace_metrics(
    base_url: str,
    project: str,
    *,
    trace_id: str | None = None,
    started_at: str | None = None,
    ended_at: str | None = None,
) -> dict[str, Any]:
    """Token usage + span counts for one run's trace, or ``{"phoenix_note": ...}`` on miss.

    The [started, ended] window narrows the query (projects accumulate every historical run).
    Attribution is deliberately conservative so it survives a CONCURRENT sweep, where the window
    returns several runs' spans: an exact ``trace_id`` match wins; otherwise usage is reported
    only when the window resolves to a SINGLE trace (or, with no id recorded and several traces,
    the most recent one). If a requested ``trace_id`` isn't in the window and other traces are, it
    returns a miss (blank usage) rather than summing the whole window and folding other runs'
    tokens into this row. A frame with no trace-id column at all falls back to summing the window.
    """
    try:
        from phoenix.client import Client

        from .tracing import phoenix_base_url
    except Exception as exc:
        return {"phoenix_note": f"client unavailable: {type(exc).__name__}"}

    kwargs: dict[str, Any] = {"project_name": project, "timeout": 60, "limit": 20000}
    start = _parse_iso(started_at)
    end = _parse_iso(ended_at)
    if start is not None:
        kwargs["start_time"] = start - dt.timedelta(seconds=10)
    if end is not None:
        kwargs["end_time"] = end + dt.timedelta(seconds=60)
    try:
        df = Client(base_url=phoenix_base_url(base_url)).spans.get_spans_dataframe(**kwargs)
    except Exception as exc:
        return {"phoenix_note": f"query failed: {type(exc).__name__}: {exc}"}
    if df is None or len(df) == 0:
        return {"phoenix_note": "no spans in window"}

    note = None
    tid_col = "context.trace_id"
    if trace_id and tid_col in df.columns and (df[tid_col] == trace_id).any():
        sub = df[df[tid_col] == trace_id]
    elif tid_col in df.columns:
        # Do NOT sum an arbitrary multi-run window: under a concurrent sweep the [start, end]
        # window (or a project queried with no id/window) returns OTHER runs' spans too, and
        # blindly summing them inflates this run's tokens/counts. Only attribute when the frame
        # resolves to a SINGLE trace; if the requested trace isn't present (not yet ingested, or
        # dropped by the `limit`) and others are, report the miss and leave usage blank — an
        # honest gap beats a total polluted by other runs.
        uniq = df[tid_col].dropna().unique().tolist()
        if len(uniq) == 1:
            sub = df[df[tid_col] == uniq[0]]
            note = ("matched single trace in window" if not trace_id
                    else f"trace {trace_id} not found; used the single trace in the window")
        elif not uniq:
            return {"phoenix_note": "no trace ids in window"}
        elif trace_id:
            return {"phoenix_note":
                    f"trace {trace_id} not found among {len(uniq)} traces in the window"}
        elif "start_time" in df.columns:
            non_null = df[df[tid_col].notna()]
            latest = non_null.sort_values("start_time").iloc[-1][tid_col]
            sub = df[df[tid_col] == latest]
            note = "no trace id; matched latest trace in window"
        else:
            return {"phoenix_note": f"no trace id and {len(uniq)} traces in the window"}
    else:
        # No trace-id column at all: the [start, end] query window is the only isolation
        # available, so fall back to summing it (the pre-existing behavior, so a single-run
        # deployment still reports usage). Flagged unverified because a concurrent window without
        # trace ids could include other runs.
        sub = df
        note = "no trace id column; summed the query window (unverified)"

    kinds = sub["attributes.openinference.span.kind"] if \
        "attributes.openinference.span.kind" in sub.columns else None

    def _kcount(kind: str) -> int:
        return int((kinds == kind).sum()) if kinds is not None else 0

    def _tok(name: str) -> int | None:
        col = f"attributes.llm.token_count.{name}"
        if col not in sub.columns:
            return None
        vals = sub[col].dropna()
        # Empty after dropna = the column exists but no span carried a token count. Report None
        # (unknown), not 0 — a genuine zero and "never captured" must stay distinguishable.
        return int(vals.sum()) if len(vals) else None

    model = None
    if "attributes.llm.model_name" in sub.columns:
        seen = sub["attributes.llm.model_name"].dropna().unique().tolist()
        model = seen[0] if seen else None

    return {
        "n_spans": int(len(sub)),
        "n_llm_calls": _kcount("LLM"),
        "n_tool_spans": _kcount("TOOL"),
        "n_agent_spans": _kcount("AGENT"),
        "n_span_errors": int((sub["status_code"] == "ERROR").sum())
        if "status_code" in sub.columns else 0,
        "tokens_prompt": _tok("prompt"),
        "tokens_completion": _tok("completion"),
        "tokens_total": _tok("total"),
        "trace_model": model,
        "phoenix_note": note,
    }


def build_row(result: dict[str, Any], *, phoenix_base_url: str | None = None) -> dict[str, Any]:
    """One flat metrics row from a result.json dict, enriched with Phoenix usage if reachable."""
    row: dict[str, Any] = {k: result.get(k) for k in _RESULT_FIELDS}
    row.update({k: result.get(k) for k in _NESTED_SOURCE_FIELDS})
    row.update({k: None for k in _PHOENIX_FIELDS})
    project = result.get("phoenix_project")
    if phoenix_base_url and project:
        row.update(phoenix_trace_metrics(
            phoenix_base_url, project,
            trace_id=result.get("phoenix_trace_id"),
            started_at=result.get("run_started_at"),
            ended_at=result.get("run_ended_at"),
        ))
    return row


def _flatten_edit_rejects(row: dict) -> dict:
    """Expand result.json's nested ``edit_rejects`` into the flat history columns.

    Kept as a transform at write time rather than changing the result.json shape: the nested form
    is the readable one for a single run, and the CSV needs columns you can group a sweep by.
    A row without the key passes through untouched (older runs, and aborts that predate a session).
    """
    rejects = row.get("edit_rejects")
    if not isinstance(rejects, dict):
        return row
    causes = rejects.get("by_cause") or {}
    top = max(causes, key=causes.get) if causes else None
    return {**row,
            "edits_attempted": rejects.get("attempted"),
            "edits_rejected": rejects.get("rejected"),
            "edits_fit_reverted": rejects.get("fit_reverted"),
            "edit_reject_rate": rejects.get("reject_rate"),
            "edit_reject_top_cause": top}


def _flatten_git_diff(row: dict) -> dict:
    """Expand result.json's nested ``git_diff`` into the single history column that identifies the
    working tree. Absent/None (a clean checkout) passes through, leaving the column empty -- which
    is itself the useful signal that the run WAS attributable to its commit."""
    diff = row.get("git_diff")
    if not isinstance(diff, dict):
        return row
    return {**row, "git_diff_sha256": diff.get("sha256")}


def _is_subsequence(old: "tuple[str, ...]", new: "tuple[str, ...]") -> bool:
    """Whether every column of ``old`` appears in ``new`` in the same relative order.

    The migration predicate for the append-only history: true exactly when columns were ADDED
    (anywhere) and none removed or reordered, so no existing value changes meaning and the file
    can be rewritten with the new header. A prefix test — the previous predicate — only covers
    additions at the END, which is why a column inserted mid-list was silently dropped instead.
    """
    it = iter(new)
    return all(col in it for col in old)


def _flatten_served_endpoints(row: dict) -> dict:
    """Expand ``openrouter_endpoints`` into the flat ``served_*`` history columns.

    ``served_provider`` names the endpoint when exactly one served the run and is left as a
    ``"a+b"`` join when several did — so the column reads as a group key either way, with
    ``served_drifted`` as the boolean to filter on. A row without the block (every run recorded
    before this existed) passes through untouched rather than gaining empty columns that would
    read as "not drifted"."""
    ep = row.get("openrouter_endpoints")
    if not isinstance(ep, dict):
        return row
    served = [s for s in (ep.get("served") or []) if isinstance(s, dict)]
    # provider+TIER, because the tier is half of what makes two endpoints different (`openai` and
    # `openai/flex` are different prices and different machines) -- a name-only key renders that
    # case as the useless "OpenAI+OpenAI" and hides the thing that actually differed.
    # Unattributed calls are LEFT OUT rather than stringified: `str(None)` writes the literal
    # "None", which is indistinguishable from a real endpoint name when grouping a sweep.
    names = sorted({f"{s['provider']}/{s['service_tier']}" if s.get("service_tier")
                    else str(s["provider"])
                    for s in served if s.get("provider") is not None})
    return {**row,
            "served_provider": "+".join(names) if names else None,
            "served_n_distinct": ep.get("n_distinct"),
            "served_drifted": ep.get("drifted")}


def _flatten_replicate_lof(row: dict) -> dict:
    """Expand the replicate lack-of-fit verdict nested under ``residual_structure`` into the flat
    ``lof_*`` history columns. Same transform-at-write-time shape as :func:`_flatten_edit_rejects`,
    and a row without the block passes through untouched — model-less runs and every run that
    predates the column. A run that HAS the block carries the verdict whenever the verdict was READ,
    including the 13 of 16 examples whose problems have no replicated design points: there it says
    ``available: false`` and ``lof_status`` records which kind of no-evidence it was. Where the read
    itself failed (``rollup_status: unavailable``) the status is ``unknown``, so the sweep CSV can
    still tell an unread verdict from a run that had no model to test."""
    rolled = row.get("residual_structure")
    if not isinstance(rolled, dict):
        return row
    lof = rolled.get("replicate_lack_of_fit")
    if not isinstance(lof, dict):
        return ({**row, "lof_status": "unknown"}
                if rolled.get("rollup_status") == "unavailable" else row)
    # `weak` and `lof_df` are OMITTED from the verdict when they say nothing (see
    # data_access._LOF_OMIT_WHEN_EMPTY_KEYS). `lof_weak` is written only where there IS a counted df
    # to qualify: coercing an absent one to False would record "the arm ran and its pass was solid"
    # for an arm that never ran, or whose structural count could not be taken — the same over-claim
    # data_access.residual_flag refuses when it reports None rather than False for an untested arm.
    lof_df = lof.get("lof_df")
    return {**row, "lof_status": lof.get("status"), "lof_df": lof_df,
            "lof_weak": None if lof_df is None else bool(lof.get("weak"))}


def append_history(rows: list[dict[str, Any]], history_path: Path = DEFAULT_HISTORY) -> Path:
    """Append rows to the append-only history CSV (writing the header if it is new)."""
    history_path = Path(history_path)
    is_new = not history_path.exists()
    # Append against the EXISTING header, not HISTORY_FIELDS. The history is append-only and
    # long-lived, so a field added (or inserted) between runs would otherwise write rows in the
    # new order under the old header — silently shifting every column after the insertion point
    # for the rest of the file. An older file keeps its own schema; new columns are dropped for
    # it (extrasaction="ignore") until the file is rotated.
    fields = HISTORY_FIELDS
    if not is_new:
        with history_path.open(newline="") as fh:
            existing = next(csv.reader(fh), None)
        if existing:
            fields = tuple(existing)
            # ...unless every old column still exists, in the same relative order — i.e. columns
            # were only ADDED, wherever. Then no old value changes meaning and the file can be
            # migrated in place by rewriting the header and padding old rows. Without this, a
            # column added to fix a cross-sweep ambiguity is silently dropped for exactly the
            # long-lived history it was added for (`ablation_state`, 2026-08-15: every appended
            # row would keep writing only the default-relative `ablation_arm` it disambiguates).
            #
            # SUBSEQUENCE, not prefix. The prefix test only fired for columns appended at the END,
            # and `ablation_state` went in at index 42 of 63 with 20 columns after it — so the
            # very case this migration was written for did not match it, and the column was
            # dropped from every appended row exactly as the paragraph above describes. A header
            # carrying a column that no longer exists is NOT migrated: that is a removal, where
            # rewriting would discard data, so the old header is kept and the new column skipped.
            if len(fields) < len(HISTORY_FIELDS) and _is_subsequence(fields, HISTORY_FIELDS):
                with history_path.open(newline="") as fh:
                    old_rows = list(csv.DictReader(fh))
                # Temp file + atomic replace, never truncate-in-place: this is an append-only
                # record spanning every sweep, and `open("w")` after reading it into memory means
                # an OOM kill, a cancellation or a full filesystem between the truncate and the
                # last writerow leaves it empty or half-written, with no copy anywhere.
                tmp = history_path.with_suffix(history_path.suffix + ".tmp")
                with tmp.open("w", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=HISTORY_FIELDS, extrasaction="ignore")
                    w.writeheader()
                    for r in old_rows:
                        w.writerow({k: r.get(k, "") for k in HISTORY_FIELDS})
                os.replace(tmp, history_path)
                fields = HISTORY_FIELDS
    with history_path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        if is_new:
            writer.writeheader()
        for row in rows:
            row = _flatten_served_endpoints(
                _flatten_git_diff(_flatten_replicate_lof(_flatten_edit_rejects(row))))
            writer.writerow({k: row.get(k) for k in fields})
    return history_path


def find_result_files(examples_dir: Path) -> list[Path]:
    """Every ``<example>/work/result.json`` under ``examples_dir`` (sorted by example)."""
    return sorted(examples_dir.glob("*/work/result.json"))


def collect(
    examples_dir: Path,
    *,
    phoenix_base_url: str | None = None,
) -> list[dict[str, Any]]:
    """Build a metrics row for every example's latest run (result.json + Phoenix usage)."""
    rows: list[dict[str, Any]] = []
    for path in find_result_files(examples_dir):
        try:
            result = json.loads(path.read_text())
        except Exception as exc:
            print(f"[metrics] skipping {path}: {type(exc).__name__}: {exc}")
            continue
        rows.append(build_row(result, phoenix_base_url=phoenix_base_url))
    return rows


def _fmt(value: Any, spec: str = "") -> str:
    if value is None:
        return "-"
    if spec:
        try:
            return format(value, spec)
        except (TypeError, ValueError):
            return str(value)
    return str(value)


def summarize(rows: list[dict[str, Any]]) -> str:
    """A compact across-example table + (when a scenario has repeats) its success rate."""
    if not rows:
        return "(no runs found)"
    header = ("scenario", "ver", "verdict", "ΔBIC", "chi2 p", "tokens", "tools", "errs",
              "wall_s", "temp", "code", "model")
    lines = [
        f"{header[0]:<16} {header[1]:>3} {header[2]:<7} {header[3]:>9} {header[4]:>8} "
        f"{header[5]:>9} {header[6]:>5} {header[7]:>4} {header[8]:>7} {header[9]:>5} "
        f"{header[10]:>10}  {header[11]}",
        "-" * 124,
    ]
    for r in rows:
        verdict = "PASS" if r.get("success") else "FAIL"
        lines.append(
            f"{_fmt(r.get('scenario')):<16} {_fmt(r.get('example_version')):>3} {verdict:<7} "
            f"{_fmt(r.get('delta_bic'), '+.3g'):>9} {_fmt(r.get('chi2_gof_pvalue'), '.2g'):>8} "
            f"{_fmt(r.get('tokens_total'), ','):>9} {_fmt(r.get('tool_calls')):>5} "
            f"{_fmt(r.get('n_errors')):>4} {_fmt(r.get('wall_seconds'), '.0f'):>7} "
            f"{_fmt(r.get('temperature'), 'g'):>5} {_fmt(r.get('git_sha')):>10}  "
            f"{_fmt(r.get('trace_model') or r.get('model'))}"
        )

    # Per-scenario success rate (only meaningful once there are repeats in the batch).
    by_scenario: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_scenario.setdefault(str(r.get("scenario")), []).append(r)
    repeated = {k: v for k, v in by_scenario.items() if len(v) > 1}
    if repeated:
        lines.append("")
        lines.append("success rate (this batch):")
        for scenario, group in repeated.items():
            passed = sum(1 for r in group if r.get("success"))
            lines.append(f"  {scenario:<16} {passed}/{len(group)}")
    n_pass = sum(1 for r in rows if r.get("success"))
    lines.append("")
    lines.append(f"overall: {n_pass}/{len(rows)} passed")
    return "\n".join(lines)
