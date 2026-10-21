"""Sweep-aggregation triage: don't score infrastructure-aborted runs as modeling failures.

Covers issue #62. The unit under test is the abort triage in ``gemot.experiments``:

  * an INSTANT infrastructure abort before any work (out-of-credits 402, dropped connection,
    a scoring float() crash) with ``n_ops==0 && n_fits==0`` is EXCLUDED from the success/ΔBIC
    aggregation (its empty-model ΔBIC is scoring noise);
  * a LATE abort that DID modeling work (fits across committed versions, then credit exhaustion)
    is KEPT — its best-achieved ΔBIC is a valid partial result;
  * genuine outcomes (completed, wall-time budget, recursion limit) are always kept.

``experiments`` imports ``phoenix`` lazily and ``gemot``'s ``__init__`` is lazy too, so the
test needs neither the ``phoenix`` client nor the rest of the (heavy) harness — only stdlib.
"""

from __future__ import annotations

import pytest

from gemot import experiments


def _row(**over):
    """A completed, successful run row; override fields per case."""
    base = {
        "example": "toy",
        "model": "m",
        "temperature": 0.0,
        "success": True,
        "delta_bic": 1.0,
        "chi2_gof_pvalue": 0.5,
        "stop_reason": "completed",
        "n_ops": 5,
        "n_fits": 5,
        "prompt_version": "pv",
        "git_sha": "abc",
    }
    base.update(over)
    return base


# --------------------------------------------------------------------------- predicates
@pytest.mark.parametrize(
    "stop_reason, expected",
    [
        ("completed", False),
        (None, False),
        ("", False),
        ("wall_time_budget_exceeded", False),  # self-imposed budget = genuine stop
        ("context_budget_exceeded", False),  # the context-window stop is a budget too
        # its label before gemot#7: the counts contain the "402" out-of-credits marker
        (
            "early_stop: context_window_approached: largest prompt 1,000,402 tokens >= 90% of "
            "the 1,048,576-token window",
            False,
        ),
        ("GraphRecursionError: recursion limit", False),  # ran out of steps = modeling outcome
        ("APIStatusError: Error code: 402 - insufficient credits", True),
        ("aborted: ReadTimeout: provider stalled", True),
        ("APIConnectionError: connection reset", True),
        ("stalled: no step completed in 900s", True),
        ("TypeError: float() argument must be a string or a real number", True),
        ("aborted: httpx.ReadError: connection closed", True),  # a dotted name still matches
        # gemot#44: a marker is a whole token, never part of a longer number or word
        ("early_stop: no improvement in 5 fits (best BIC 1,402.7)", False),
        ("token_usage 20,402,551 then completed early", False),
        ("early_stop: plateau on the server at 4024358", False),
        ("infrastructure_error: disk_full", True),  # gemot#39
    ],
)
def test_is_infra_abort(stop_reason, expected):
    assert experiments._is_infra_abort(stop_reason) is expected


@pytest.mark.parametrize(
    "n_ops, n_fits, expected",
    [
        (0, 0, False),
        (1, 0, True),
        (0, 1, True),
        (26, 25, True),
        (None, None, False),
        (float("nan"), float("nan"), False),  # NaN counts as no work
        (-1, -1, False),  # garbage negatives clamp to 0 (not "work done")
        (True, False, False),  # bools are bad data, not a count of 1/0
    ],
)
def test_did_modeling_work(n_ops, n_fits, expected):
    assert experiments._did_modeling_work({"n_ops": n_ops, "n_fits": n_fits}) is expected


@pytest.mark.parametrize(
    "value, expected",
    [
        (0, 0),
        (5, 5),
        (3.0, 3),
        (2.9, 2),
        (-1, 0),
        (-99, 0),  # clamped at 0
        (True, 0),
        (False, 0),  # bool excluded from the numeric path
        (None, 0),
        (float("nan"), 0),
        ("3", 0),
    ],
)
def test_as_count(value, expected):
    got = experiments._as_count(value)
    assert got == expected and type(got) is int


def test_instant_402_abort_is_noop_infra_abort():
    # oliveira 1cc14ef: n_fits=0, empty-model ΔBIC 1.55e7 → scoring noise, excluded.
    row = _row(
        success=False,
        delta_bic=1.55e7,
        n_ops=0,
        n_fits=0,
        stop_reason="APIStatusError: Error code: 402",
    )
    assert experiments.is_noop_infra_abort(row) is True


def test_typeerror_crash_at_zero_work_is_noop_infra_abort():
    # perelson 357e575: a scoring float() TypeError before any work.
    row = _row(
        success=False,
        delta_bic=None,
        n_ops=0,
        n_fits=0,
        stop_reason="aborted: TypeError: float() argument must be a real number",
    )
    assert experiments.is_noop_infra_abort(row) is True


def test_late_402_abort_with_work_is_kept():
    # brannmark 15188b28: late credit exhaustion after real modeling — a valid partial result.
    row = _row(
        success=False,
        delta_bic=16.04,
        n_ops=26,
        n_fits=25,
        stop_reason="APIStatusError: Error code: 402",
    )
    assert experiments.is_noop_infra_abort(row) is False


@pytest.mark.parametrize(
    "stop_reason",
    [
        "context_budget_exceeded",
        "early_stop: context_window_approached: largest prompt 1,000,402 tokens >= 90% of the "
        "1,048,576-token window",
    ],
)
def test_context_window_stop_at_zero_work_is_kept(stop_reason):
    # A budget stop is an outcome even when nothing was committed, whatever numbers its label has.
    row = _row(success=False, delta_bic=None, n_ops=0, n_fits=0, stop_reason=stop_reason)
    assert experiments.is_noop_infra_abort(row) is False


def test_completed_noop_is_kept_as_failure():
    # Genuinely built nothing but finished cleanly — a real modeling failure, not infra noise.
    row = _row(success=False, delta_bic=None, n_ops=0, n_fits=0, stop_reason="completed")
    assert experiments.is_noop_infra_abort(row) is False


def test_recursion_limit_noop_is_kept():
    # Spun to the step limit without committing — a modeling failure, not an infra abort.
    row = _row(
        success=False,
        delta_bic=None,
        n_ops=0,
        n_fits=0,
        stop_reason="GraphRecursionError: recursion limit reached",
    )
    assert experiments.is_noop_infra_abort(row) is False


# --------------------------------------------------------------------------- summarize_sweep
def test_summarize_sweep_excludes_only_noop_infra_aborts():
    rows = [
        _row(
            example="brannmark",
            success=False,
            delta_bic=16.04,
            n_ops=26,
            n_fits=25,
            stop_reason="APIStatusError: Error code: 402",
        ),  # late partial → KEEP
        _row(
            example="oliveira",
            success=False,
            delta_bic=1.55e7,
            n_ops=0,
            n_fits=0,
            stop_reason="APIStatusError: Error code: 402",
        ),  # instant abort → DROP
        _row(
            example="perelson",
            success=False,
            delta_bic=None,
            n_ops=0,
            n_fits=0,
            stop_reason="aborted: TypeError: float()",
        ),  # crash at 0 work → DROP
        _row(example="repressilator", success=True, delta_bic=0.4, n_ops=8, n_fits=8),  # KEEP
    ]
    out = experiments.summarize_sweep(rows)

    # Two of four rows are no-work infra aborts.
    assert "excluded 2 no-work infrastructure abort(s)" in out
    # The kept late-abort partial's ΔBIC survives; the empty-model noise does not appear.
    assert "brannmark" in out and "repressilator" in out
    assert "oliveira" not in out and "perelson" not in out
    assert "1.55e+07" not in out
    # Overall counts only the two kept runs (1 passed).
    assert "overall: 1/2 runs passed" in out


def test_summarize_sweep_include_crashed_keeps_everything():
    rows = [
        _row(
            example="oliveira",
            success=False,
            delta_bic=1.55e7,
            n_ops=0,
            n_fits=0,
            stop_reason="APIStatusError: Error code: 402",
        ),
        _row(example="repressilator", success=True, delta_bic=0.4, n_ops=8, n_fits=8),
    ]
    out = experiments.summarize_sweep(rows, drop_noop_infra_aborts=False)
    assert "excluded" not in out
    assert "overall: 1/2 runs passed" in out
    assert "oliveira" in out
