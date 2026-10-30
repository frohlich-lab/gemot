"""The judge scores one version's nll, n and k, never a mix of two (gemot#73).

perica_egf, ``causal_constraints`` arm, Codex ``native_cli`` driver: the CLI died out of credits
while a ``make_changes`` fit was still running. The judge's scan listed v1-v5 and scored v5, the fit
then committed v6 between ``get_fit_summary`` and ``get_parameters``, and result.json carried v5's
nll (41,309.17) with v6's k (104): a BIC of 83,493.4 that no version has, on a run whose true best was
v6 (83,446.25). These tests drive the real ``_judge`` over a fake session holding exactly that
in-flight fit, landing it at a chosen call.
"""

from __future__ import annotations

import asyncio
import math
import time
from pathlib import Path

import pytest

pytest.importorskip("mcp", reason="harness-only dependency")

from gemot.runner import _ToolClock  # noqa: E402
from gemot.runner.judge import MixedVersionsError, _judge  # noqa: E402

#: The run's data-point count, and each version's (nll, estimated-row count).
_N = 4511
_V5 = (41309.17, 107)
_V6 = (41285.58, 104)


def _rows(k):
    return [{"parameterId": f"p{i}", "estimate": 1, "nominalValue": 1.0} for i in range(k)]


def _bic(nll, k):
    return 2 * nll + k * math.log(_N)


class _Scenario:
    name = "perica"
    bic_threshold = 10.0
    diagnostics = None

    def __init__(self, work: Path):
        self.example_dir = work / "ex"

    def reference(self):
        return {"nll": 41300.0, "k": 100, "n": _N}


class _Baseline:
    nll = None
    species: set = set()


class _Session:
    """A server with v5 committed and HEAD on it, and v6's fit still running. The fit lands just
    before the server answers ``lands_on``: ``"commit"`` commits v6 (HEAD moves to it), ``"edit"``
    only puts v6's model in the working state, as a change does before its fit commits."""

    def __init__(self, lands_on: str, landing: str = "commit"):
        self.stored = {5: _V5}
        self.head = 5
        self.working = _V5
        self.lands_on, self.landing = lands_on, landing
        self.calls: list[str] = []

    def _land(self):
        self.working = _V6
        if self.landing == "commit":
            self.stored[6], self.head = _V6, 6
        self.lands_on = None

    async def __call__(self, name, **kwargs):
        self.calls.append(name)
        if name == self.lands_on:
            self._land()
        if name == "list_versions":
            return [
                {"version": v, "bic": _bic(*s), "criterion_value": _bic(*s)}
                for v, s in self.stored.items()
            ]
        if name == "restore_version":
            self.head = kwargs["version"]
            self.working = self.stored[self.head]
            return {"version": self.head}
        if name == "describe_problem":
            return {"current_version": self.head}
        if name == "get_version":
            nll, k = self.stored[kwargs["version"]]
            return {
                "version": kwargs["version"],
                "neg_log_likelihood": nll,
                "simulation": {"neg_log_likelihood": nll, "n_data_points": _N},
                "parameters": _rows(k),
            }
        if name == "get_fit_summary":
            return {"neg_log_likelihood": self.working[0], "n_data_points": _N, "criterion": "bic"}
        if name == "get_parameters":
            return [{**row, "in_model": True} for row in _rows(self.working[1])]
        if name == "get_model_structure":
            return {"reactions": [], "species": []}
        return [] if name == "get_errors" else {}


def _judge_on(work: Path, session: _Session):
    return asyncio.run(
        _judge(_Scenario(work), session, _Baseline(), work, {}, _ToolClock(), time.monotonic())
    )


def test_the_judge_waits_for_the_running_fit_and_scores_its_version(tmp_path):
    """The fix: the judge's first call is the one the server answers only once the running fit has
    committed (``get_progress``, read under the lock the fit holds), so the scan lists v6 and
    scores it whole: v6's nll with v6's k, the best BIC of the run."""
    session = _Session(lands_on="get_progress")
    _, result = _judge_on(tmp_path, session)

    assert session.calls.index("get_progress") < session.calls.index("list_versions")
    assert result["scored_version"] == result["head_version"] == 6
    assert (result["final_nll"], result["k"]) == _V6
    assert result["bic"] == pytest.approx(_bic(*_V6))
    assert result["fit_wait_seconds"] is not None


@pytest.mark.parametrize(
    "landing, why",
    [
        # #73 itself: v6 commits between get_fit_summary (v5's) and get_parameters (v6's).
        ("commit", "HEAD moved from v5 to v6"),
        # A change still fitting when read, its candidate table in the working state while HEAD
        # stays on v5: the HEAD check alone would pass this.
        ("edit", "get_parameters' estimated rows are not v5's"),
    ],
)
def test_reads_across_two_versions_fail_the_judgement_instead_of_scoring_it(tmp_path, landing, why):
    """If a fit lands during the reads anyway (one the wait could not outlast, or a server whose
    progress read does not wait), the judge refuses to score rather than writing v5's nll with
    v6's k. No result.json: the runner writes the run as degraded with this message."""
    session = _Session(lands_on="get_parameters", landing=landing)
    with pytest.raises(MixedVersionsError, match=why):
        _judge_on(tmp_path, session)
    assert not (tmp_path / "result.json").exists()


def test_a_run_with_no_fit_in_flight_scores_the_version_it_restored(tmp_path):
    """The check costs a healthy run nothing: with no fit landing, the judge scores v5 and says so."""
    _, result = _judge_on(tmp_path, _Session(lands_on=None))
    assert result["scored_version"] == 5
    assert (result["final_nll"], result["k"]) == _V5
