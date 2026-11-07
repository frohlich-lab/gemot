"""`gemot.metrics`: the cross-run history CSV."""

from __future__ import annotations

import csv
import sys
from types import SimpleNamespace

import pandas as pd

from gemot import metrics as M


def _rows(path):
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def test_collecting_twice_does_not_append_the_same_run_twice(tmp_path):
    """`collect_metrics.py` re-reads each example's latest result.json, so a second collection
    with no new run in between appended every run again and each counted twice (#10)."""
    out = tmp_path / "history.csv"
    a = {"scenario": "toy", "run_started_at": "2026-10-20T10:00:00.000001+00:00", "success": True}
    b = {"scenario": "toy", "run_started_at": "2026-10-20T11:00:00.000001+00:00", "success": False}

    M.append_history([a], out)
    M.append_history([a, b, b], out)

    assert [r["run_started_at"] for r in _rows(out)] == [a["run_started_at"], b["run_started_at"]]
    # The same instant in another example is another run.
    M.append_history([{**a, "scenario": "other"}], out)
    assert len(_rows(out)) == 3


def test_a_row_without_a_start_time_is_always_appended(tmp_path):
    """Nothing identifies it, so nothing says it repeats an earlier row."""
    out = tmp_path / "history.csv"
    M.append_history([{"scenario": "toy"}], out)
    M.append_history([{"scenario": "toy"}], out)
    assert len(_rows(out)) == 2


def _trace_metrics(monkeypatch, spans):
    frame = pd.DataFrame([{"context.trace_id": "run", **span} for span in spans])
    client = SimpleNamespace(spans=SimpleNamespace(get_spans_dataframe=lambda **_: frame))
    monkeypatch.setitem(sys.modules, "phoenix.client", SimpleNamespace(Client=lambda **_: client))
    return M.phoenix_trace_metrics("http://collector.example", "project", trace_id="run")


def _native_session(provider, **counts):
    return {
        "attributes.openinference.span.kind": "AGENT",
        "attributes.gemot.cli.provider": provider,
        "attributes.gemot.cli.usage_scope": "session_aggregate",
        "attributes.gemot.cli.requested_model": "requested-model",
        **{f"attributes.gemot.cli.{key}": value for key, value in counts.items()},
    }


def test_codex_uses_session_tokens_without_counting_output_items_as_api_calls(monkeypatch):
    result = _trace_metrics(
        monkeypatch,
        [
            _native_session("codex", prompt_tokens=100, completion_tokens=20, total_tokens=120),
            {
                "attributes.openinference.span.kind": "LLM",
                "attributes.gemot.cli.observation_scope": "output_item",
            },
            {
                "attributes.openinference.span.kind": "LLM",
                "attributes.gemot.cli.observation_scope": "output_item",
            },
        ],
    )
    assert result["n_spans"] == 3
    assert result["n_llm_calls"] is None
    assert result["tokens_prompt"] == 100
    assert result["tokens_completion"] == 20
    assert result["tokens_total"] == 120
    assert result["trace_model"] == "requested-model"


def test_codex_output_items_without_session_still_do_not_identify_api_calls(monkeypatch):
    result = _trace_metrics(
        monkeypatch,
        [
            {
                "attributes.openinference.span.kind": "LLM",
                "attributes.gemot.cli.observation_scope": "output_item",
            }
        ],
    )
    assert result["n_llm_calls"] is None
    assert result["tokens_total"] is None


def test_claude_aggregate_takes_precedence_over_partial_response_spans(monkeypatch):
    result = _trace_metrics(
        monkeypatch,
        [
            _native_session(
                "claude", n_llm_calls=2, prompt_tokens=100, completion_tokens=20, total_tokens=120
            ),
            {
                "attributes.openinference.span.kind": "LLM",
                "attributes.llm.model_name": "actual-model",
                "attributes.llm.token_count.prompt": 40,
                "attributes.llm.token_count.completion": 5,
                "attributes.llm.token_count.total": 45,
            },
        ],
    )
    assert result["n_llm_calls"] == 2
    assert result["tokens_total"] == 120  # Neither 45 nor aggregate + response (165).
    assert result["trace_model"] == "actual-model"


def test_native_unknown_aggregate_does_not_fall_back_to_partial_response_usage(monkeypatch):
    result = _trace_metrics(
        monkeypatch,
        [
            _native_session("claude", completion_tokens=0, prompt_tokens=None),
            {
                "attributes.openinference.span.kind": "LLM",
                "attributes.llm.token_count.prompt": 40,
                "attributes.llm.token_count.completion": 5,
                "attributes.llm.token_count.total": 45,
            },
        ],
    )
    assert result["n_llm_calls"] is None
    assert result["tokens_prompt"] is None
    assert result["tokens_completion"] == 0
    assert result["tokens_total"] is None


def test_a_span_without_a_total_counts_its_prompt_and_completion(monkeypatch):
    """A turn the thinking hand-off put together from two calls recorded prompt and completion but
    no total until #125, and summing the total column alone left the run's ``tokens_total`` short
    of the two: one such span in sweep 63372750 held 371k prompt + 132k completion tokens."""
    result = _trace_metrics(
        monkeypatch,
        [
            {
                "attributes.openinference.span.kind": "LLM",
                "attributes.llm.token_count.prompt": 40,
                "attributes.llm.token_count.completion": 5,
                "attributes.llm.token_count.total": 45,
            },
            {
                "attributes.openinference.span.kind": "LLM",
                "attributes.llm.token_count.prompt": 371_000,
                "attributes.llm.token_count.completion": 132_000,
            },
            {"attributes.openinference.span.kind": "TOOL"},
        ],
    )
    assert result["tokens_prompt"] == 371_040
    assert result["tokens_completion"] == 132_005
    assert result["tokens_total"] == 503_045  # Not 45.


def test_openrouter_span_accounting_is_unchanged(monkeypatch):
    result = _trace_metrics(
        monkeypatch,
        [
            {
                "attributes.openinference.span.kind": "LLM",
                "attributes.llm.model_name": "openrouter-model",
                "attributes.llm.token_count.prompt": 40,
                "attributes.llm.token_count.completion": 5,
                "attributes.llm.token_count.total": 45,
            },
            {
                "attributes.openinference.span.kind": "LLM",
                "attributes.llm.token_count.prompt": 20,
                "attributes.llm.token_count.completion": 7,
                "attributes.llm.token_count.total": 27,
            },
            {"attributes.openinference.span.kind": "TOOL", "status_code": "ERROR"},
        ],
    )
    assert result["n_llm_calls"] == 2
    assert result["tokens_prompt"] == 60
    assert result["tokens_completion"] == 12
    assert result["tokens_total"] == 72
    assert result["n_tool_spans"] == 1
    assert result["n_span_errors"] == 1
    assert result["trace_model"] == "openrouter-model"
