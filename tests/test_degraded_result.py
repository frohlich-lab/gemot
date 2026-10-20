"""The degraded ``result.json`` a run writes when it aborts before a verdict (gemot#5).

Two ways it used to lose the run's work, so the sweep dropped late aborts as zero-work noise:

* the version store was read at ``work/prov/versions``, a path the runner never configures (the
  server's store is ``work/.petab_mcp``), so no degraded result carried a version count at all;
* the guard in ``runner.drive`` wrote ``base_meta`` even when the JUDGE failed after the driver
  had returned, dropping ``token_usage``, ``make_changes_calls`` and the guard counters.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from gemot import analysis as A
from gemot import experiments as E
from gemot.runner.failure import _write_degraded_result

_ABORT = "ReadTimeout: provider stalled"


def _store(work, versions: int, head: int | None) -> None:
    """A server store under ``work`` holding ``v0..v{versions-1}``, HEAD at ``head``."""
    vroot = A.store_dir(work) / "versions"
    vroot.mkdir(parents=True)
    for n in range(versions):
        (vroot / f"v{n}").mkdir()
    if head is not None:
        (A.store_dir(work) / "HEAD").write_text(str(head))


def _degraded(work) -> dict:
    _write_degraded_result(work, {"driver": "deepagents"}, SimpleNamespace(name="toy"), _ABORT)
    return json.loads((work / "result.json").read_text())


@pytest.mark.parametrize(
    "versions, head, stop_class",
    [
        # bachmann 52859568_0: a late abort after real fits must stay in the aggregation.
        (3, 2, "infra_partial"),
        # v0 is the first FIT, not a baseline: one version is one fit (the old count said 0).
        (1, 0, "infra_partial"),
        # A server that started and fit nothing: the abort really is zero-work noise.
        (0, None, "infra_noop"),
    ],
)
def test_a_degraded_result_reads_head_off_the_servers_store(tmp_path, versions, head, stop_class):
    _store(tmp_path, versions, head)
    row = _degraded(tmp_path)

    expected = -1 if head is None else head
    assert row["head_version"] == expected
    assert row["n_ops"] == expected + 1
    # Both consumers: the result.json triage, and the Phoenix-row filter that drops noise.
    assert A.stop_class(row) == stop_class
    assert E.is_noop_infra_abort(row) is (stop_class == "infra_noop")


def test_head_version_is_head_not_the_highest_version(tmp_path):
    """A restore moves HEAD back without deleting versions. ``head_version`` must mean what the
    judge's does, the version ``describe_problem`` reports, not the newest one on disk."""
    _store(tmp_path, versions=4, head=1)
    assert _degraded(tmp_path)["head_version"] == 1


def test_an_abort_before_the_server_writes_no_version_count(tmp_path):
    """No store means no evidence either way, so neither field is invented."""
    row = _degraded(tmp_path)
    assert "head_version" not in row and "n_ops" not in row


# --------------------------------------------------------------------------- runner.drive guard

#: What the stub driver reports; `_agent_meta` spreads all of it into `meta`.
_REPORT = {
    "stop_reason": "completed",
    "token_usage": {"total_tokens": 1234},
    "make_changes_calls": 7,
}


class _Session:
    """Just enough server for `_drive`'s preflight on a zero-shot problem."""

    async def call_tool(self, name, arguments=None, *args, **kwargs):
        replies = {"describe_problem": {"model_authored": False}, "get_model_structure": {}}
        return SimpleNamespace(structuredContent=replies.get(name, {}))


def _drive_failing(tmp_path, monkeypatch, *, in_judge: bool) -> dict:
    """Run `_drive` with the driver stubbed and either the driver or the judge raising."""
    from gemot import agent
    from gemot.runner import drive

    async def run_deepagents(*_a, **_kw):
        if not in_judge:
            raise TimeoutError("driver stalled")
        return dict(_REPORT)

    async def judge(*_a, **_kw):
        raise TimeoutError("judge stalled")

    monkeypatch.setattr(agent, "run_deepagents", run_deepagents, raising=False)
    monkeypatch.setattr(drive, "_judge", judge)
    base_meta = {"driver": "deepagents", "git_sha": "x", "run_started_at": "t0"}
    monkeypatch.setattr(drive, "_base_meta", lambda *_a: dict(base_meta))
    monkeypatch.setattr(drive, "_write_live_marker", lambda *_a, **_kw: None)
    args = SimpleNamespace(
        driver="deepagents",
        phoenix=False,
        api_key="k",
        model="m",
        base_url="http://invalid",
        recursion_limit=5,
        temperature=0.0,
        reasoning_effort="default",
        n_starts=1,
        max_wall_seconds=None,
    )
    scenario = SimpleNamespace(
        name="toy",
        version="1",
        system_prompt="",
        agent_task="",
        agent_entities=(),
        example_dir=tmp_path,
    )
    exit_code, result = asyncio.run(drive._drive(scenario, _Session(), tmp_path, args))
    assert exit_code == 1
    assert json.loads((tmp_path / "result.json").read_text()) == result
    return result


def test_a_judge_that_fails_keeps_what_the_driver_reported(tmp_path, monkeypatch):
    """The regression: this guard wrote `base_meta` while the server-died branch beside it wrote
    `meta`, so a judge-time abort lost the run's tokens and edit counts."""
    result = _drive_failing(tmp_path, monkeypatch, in_judge=True)
    assert result["stop_reason"] == "aborted: TimeoutError: judge stalled"
    assert result["token_usage"] == _REPORT["token_usage"]
    assert result["make_changes_calls"] == _REPORT["make_changes_calls"]


def test_a_driver_that_fails_still_writes_its_own_abort(tmp_path, monkeypatch):
    """Before the driver returns there is only `base_meta`, and the guard must still name the
    driver's exception rather than fail on an unbound `meta` and fall through to the outer one."""
    result = _drive_failing(tmp_path, monkeypatch, in_judge=False)
    assert result["stop_reason"] == "aborted: TimeoutError: driver stalled"
    assert result["git_sha"] == "x"
    assert "token_usage" not in result
