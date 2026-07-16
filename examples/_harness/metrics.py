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
from pathlib import Path
from typing import Any

# Outcome columns copied straight from result.json.
_RESULT_FIELDS = (
    "scenario", "example_version", "model", "driver", "success", "stop_reason",
    "baseline_nll", "final_nll", "delta",
    "chi2", "chi2_dof", "chi2_gof_pvalue",
    "k", "n", "bic", "bic_reference", "delta_bic", "bic_threshold",
    "head_version", "n_errors", "tool_calls",
    "wall_seconds", "compute_seconds", "llm_seconds",
    "compile_seconds", "fit_seconds", "n_compiles", "n_fits",
    "git_sha", "recursion_limit", "n_starts", "temperature",
    "run_started_at", "run_ended_at",
    "phoenix_project", "phoenix_trace_id",
)

# Columns derived from the Phoenix trace (token usage + span breakdown).
_PHOENIX_FIELDS = (
    "n_spans", "n_llm_calls", "n_tool_spans", "n_agent_spans", "n_span_errors",
    "tokens_prompt", "tokens_completion", "tokens_total", "trace_model", "phoenix_note",
)

HISTORY_FIELDS = _RESULT_FIELDS + _PHOENIX_FIELDS
DEFAULT_HISTORY = Path(__file__).resolve().parent / "metrics_history.csv"


def _parse_iso(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    try:
        return dt.datetime.fromisoformat(value)
    except Exception:
        return None


def _phoenix_base_url(url: str) -> str:
    """Normalize to the Phoenix base for ``Client(base_url=...)``.

    ``PHOENIX_COLLECTOR_ENDPOINT`` may be either the base (``http://host:6006``) or a full
    OTLP path (``.../v1/traces``); the Phoenix client wants the base, so strip a trailing
    OTLP suffix (mirrors ``tracing._base_endpoint``).
    """
    out = url.rstrip("/")
    for suffix in ("/v1/traces", "/v1/otlp"):
        if out.endswith(suffix):
            return out[: -len(suffix)]
    return out


def phoenix_trace_metrics(
    base_url: str,
    project: str,
    *,
    trace_id: str | None = None,
    started_at: str | None = None,
    ended_at: str | None = None,
) -> dict[str, Any]:
    """Token usage + span counts for one run's trace, or ``{"phoenix_note": ...}`` on miss.

    Resolution order: exact ``trace_id`` → the run's [started, ended] window → the most
    recent trace in the project. The window both narrows the query (projects accumulate
    every historical run) and disambiguates when no trace id was recorded.
    """
    try:
        from phoenix.client import Client
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
        df = Client(base_url=_phoenix_base_url(base_url)).spans.get_spans_dataframe(**kwargs)
    except Exception as exc:
        return {"phoenix_note": f"query failed: {type(exc).__name__}: {exc}"}
    if df is None or len(df) == 0:
        return {"phoenix_note": "no spans in window"}

    note = None
    if trace_id and "context.trace_id" in df.columns and (df["context.trace_id"] == trace_id).any():
        sub = df[df["context.trace_id"] == trace_id]
    elif start is not None or end is not None:
        sub = df  # the window already isolates this run
        note = "matched by time window" if not trace_id else "trace id not found; used window"
    elif "context.trace_id" in df.columns and "start_time" in df.columns:
        latest = df.sort_values("start_time").iloc[-1]["context.trace_id"]
        sub = df[df["context.trace_id"] == latest]
        note = "matched latest trace (no id/window)"
    else:
        sub = df
        note = "no trace id column"

    kinds = sub["attributes.openinference.span.kind"] if \
        "attributes.openinference.span.kind" in sub.columns else None

    def _kcount(kind: str) -> int:
        return int((kinds == kind).sum()) if kinds is not None else 0

    def _tok(name: str) -> int | None:
        col = f"attributes.llm.token_count.{name}"
        return int(sub[col].dropna().sum()) if col in sub.columns else None

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


def append_history(rows: list[dict[str, Any]], history_path: Path = DEFAULT_HISTORY) -> Path:
    """Append rows to the append-only history CSV (writing the header if it is new)."""
    history_path = Path(history_path)
    is_new = not history_path.exists()
    with history_path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=HISTORY_FIELDS, extrasaction="ignore")
        if is_new:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in HISTORY_FIELDS})
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
