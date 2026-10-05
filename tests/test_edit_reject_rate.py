"""``result.json`` must carry how often the agent's edits were REFUSED.

A rejected edit is a SUCCESSFUL tool call that returns ``applied: false``, so it shows up in
neither ``n_tool_errors`` nor the tool clock — it was invisible in every result.json. Mining the
Phoenix DB for one sweep found 518/2691 ``set_model`` calls (19%) and 111/1475
``change_parameters`` calls (7.5%) refused, dominated by causes the skills do not mention. That is
a large, steady tax on the agent's turns that nobody could see without a 36 GB trace query.
"""

from __future__ import annotations

import json

import pytest

from gemot.runner.clock import _ToolClock


class _Block:
    def __init__(self, text: str) -> None:
        self.text = text


class _Envelope:
    """The shape ``session.call_tool`` returns (content blocks carrying a JSON body)."""

    def __init__(self, payload: dict) -> None:
        self.content = [_Block(json.dumps(payload))]


def test_an_applied_edit_is_counted_but_not_rejected():
    clock = _ToolClock()
    clock.note_result("make_changes", _Envelope({"applied": True}))
    assert clock.reject_summary() == {
        "attempted": 1,
        "rejected": 0,
        "fit_reverted": 0,
        "reject_rate": 0.0,
        "by_cause": {},
    }


@pytest.mark.parametrize(
    "error,cause",
    [
        ("Failed to parse antimony model: syntax error, unexpected ':'", "antimony_syntax"),
        ("Operation 'set_model' rejected: the resulting problem is invalid.", "invalid_problem"),
        # The reasons now travel INSIDE the message (see operations._rejection_summary). The bucket must
        # be unchanged: the substring is still there, and no earlier pattern may capture the embedded
        # issue text -- `antimony_syntax` is matched first, and a parse failure raises before
        # validation, so it can never appear here.
        (
            "Operation 'make_changes' rejected: the resulting problem is invalid: 'RAF_total' ships "
            "FIXED (estimate=0) in this problem | Extraneous parameter(s) in parameter table: {'pAkt_S6'} "
            "(+2 more in `issues`)",
            "invalid_problem",
        ),
        ("Refusing to widen 'k' a 3rd time: it re-pinned at its upper bound", "repeated_widening"),
        (
            "change_parameters: 'k' sets only one bound — a bound edit needs BOTH",
            "bad_bounds_usage",
        ),
        ("change_parameters: entry for 'k' has unknown key(s) ['bounds']", "unknown_key"),
        ("Unknown parameter id 'kx'. Known ids: [...]", "unknown_parameter"),
        ("Invalid parameterScale 'lin10'; use one of ['lin', 'log', 'log10'].", "bad_scale"),
        ("Cannot interpret estimate value 'yes' as 0/1.", "bad_estimate"),
        ("something nobody has bucketed yet", "other"),
    ],
)
def test_every_observed_rejection_reason_gets_a_bucket(error, cause):
    """The buckets are taken from the real distribution, and an unmatched reason must fall into
    ``other`` rather than being dropped — a silently uncounted rejection is the thing this
    metric exists to prevent."""
    clock = _ToolClock()
    clock.note_result("make_changes", _Envelope({"applied": False, "error": error}))
    summary = clock.reject_summary()
    assert summary["rejected"] == 1 and summary["by_cause"] == {cause: 1}


def test_non_edit_tools_are_not_counted():
    """``fit`` and friends have no ``applied`` flag; counting them would dilute the rate."""
    clock = _ToolClock()
    clock.note_result("fit", _Envelope({"applied": False, "error": "x"}))
    clock.note_result("get_measurements", _Envelope({"applied": False}))
    assert clock.reject_summary()["attempted"] == 0


def test_an_unparseable_result_never_breaks_the_run():
    """A metric must degrade, not raise, if the MCP envelope shape changes."""
    clock = _ToolClock()
    for junk in ("not an envelope", None, 42, _Envelope.__new__(_Envelope)):
        clock.note_result("make_changes", junk)
    assert clock.reject_summary()["attempted"] == 0


def test_the_rate_is_none_rather_than_zero_when_nothing_was_attempted():
    """0.0 would read as "the agent made edits and none were refused"; None says "no edits"."""
    assert _ToolClock().reject_summary()["reject_rate"] is None


def test_the_pre_rename_tool_names_still_count():
    """#130 renamed the editors to ``make_changes``; old traces and a pinned server still emit
    ``set_model``/``change_parameters``, and the metric must be comparable across that boundary."""
    clock = _ToolClock()
    for name in ("set_model", "change_parameters", "make_changes"):
        clock.note_result(name, _Envelope({"applied": False, "error": "boom"}))
    assert clock.reject_summary()["rejected"] == 3


def test_a_fit_error_on_an_APPLIED_change_is_not_a_rejection():
    """`bundle_result` flattens the fit's keys into the payload, so a change that applied — and
    was even snapshotted as a version — can carry an `error`. Counting that as a refusal inflated
    the rate and filled `by_cause` with fit failures, in the one metric whose whole job is telling
    edit refusals apart from fit failures."""
    clock = _ToolClock()
    clock.note_result(
        "make_changes",
        _Envelope(
            {"applied": True, "committed": False, "version": 7, "error": "fit did not converge"}
        ),
    )
    assert clock.reject_summary() == {
        "attempted": 1,
        "rejected": 0,
        "fit_reverted": 0,
        "reject_rate": 0.0,
        "by_cause": {},
    }


def test_a_fit_revert_is_not_an_edit_rejection():
    """Since #130 one call edits AND fits, so `applied: false` covers two events. A pure edit
    refusal comes straight from the tool's error wrapper (`{applied, error}`); once the fit stage
    has run, `bundle_result` always sets `committed`. Counting the reverts as refusals made this
    an edit-or-fit failure rate and broke comparability with the pre-#130 baseline."""
    clock = _ToolClock()
    clock.note_result(
        "make_changes",
        _Envelope(
            {
                "applied": False,
                "committed": False,
                "version": None,
                "message": "Change REVERTED: it could not be scored",
                "error": "fit diverged",
            }
        ),
    )
    s = clock.reject_summary()
    assert (s["rejected"], s["fit_reverted"], s["reject_rate"]) == (0, 1, 0.0)


def test_the_history_csv_gets_flat_edit_reject_columns(tmp_path):
    """result.json's nested dict has no CSV representation, and a name missing from
    `_RESULT_FIELDS` is dropped SILENTLY by `extrasaction="ignore"` — which is exactly how this
    metric stayed invisible in the history while being written to every result.json."""
    import csv as _csv

    from gemot import metrics

    out = tmp_path / "history.csv"
    metrics.append_history(
        [
            {
                "scenario": "demo",
                "edit_rejects": {
                    "attempted": 8,
                    "rejected": 3,
                    "fit_reverted": 1,
                    "reject_rate": 0.375,
                    "by_cause": {"antimony_syntax": 2, "other": 1},
                },
            }
        ],
        history_path=out,
    )
    row = next(iter(_csv.DictReader(out.open())))
    assert row["edits_attempted"] == "8"
    assert row["edits_rejected"] == "3"
    assert row["edits_fit_reverted"] == "1"
    assert row["edit_reject_rate"] == "0.375"
    assert row["edit_reject_top_cause"] == "antimony_syntax"


def test_every_reject_bucket_matches_a_message_the_server_can_emit():
    """Buckets are matched against English prose, so a reworded message turns a live bucket into a
    dead one with nothing failing. Two buckets were already dead on arrival: they were mined from
    traces predating #118, which turned the sigma cap into a warning. Pin them against the real
    strings in the server source instead of literals typed into this file."""
    from conftest import SERVER_SRC
    from gemot.runner.clock import _REJECT_PATTERNS

    src = "\n".join(p.read_text() for p in SERVER_SRC.rglob("*.py")).lower()
    dead = [label for label, needle in _REJECT_PATTERNS if needle not in src]
    assert not dead, f"reject buckets matching no message the server emits: {dead}"


# -- the working tree must be attributable, not just the commit ------------------------------
def test_git_diff_digest_fingerprints_an_uncommitted_tree(tmp_path):
    """`git_sha` ends in `+dirty` when the tree is modified, which means it does NOT pin the code.
    The Aug 8-10 sweep recorded `732ee74+dirty` for all 361 runs across three array jobs whose
    measured tool-failure rates differed materially, and nothing in the record could separate
    "different code" from "different model"."""
    import subprocess

    from gemot.runner.provenance import _git_diff_digest, _git_sha

    def git(*a):
        subprocess.check_call(
            ["git", *a], cwd=tmp_path, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

    git("init")
    git("config", "user.email", "t@t")
    git("config", "user.name", "t")
    (tmp_path / "a.txt").write_text("one\n")
    git("add", "a.txt")
    git("commit", "-m", "first")

    # Clean tree: no digest, and git_sha carries no +dirty -> the run IS attributable.
    assert _git_diff_digest(tmp_path) is None
    assert not (_git_sha(tmp_path) or "").endswith("+dirty")

    (tmp_path / "a.txt").write_text("one\ntwo\n")
    d = _git_diff_digest(tmp_path)
    assert (_git_sha(tmp_path) or "").endswith("+dirty")
    assert d and len(d["sha256"]) == 64
    assert d["n_files"] == 1 and d["files"] == ["a.txt"] and d["lines"] > 0

    # A DIFFERENT modification of the same commit gets a different digest -- the point of the fix.
    first = d["sha256"]
    (tmp_path / "a.txt").write_text("one\nthree\n")
    assert _git_diff_digest(tmp_path)["sha256"] != first
    # ...and the same modification reproduces the same digest.
    (tmp_path / "a.txt").write_text("one\ntwo\n")
    assert _git_diff_digest(tmp_path)["sha256"] == first


def test_the_run_records_which_server_code_served_it(tmp_path, monkeypatch):
    """`git_sha` covers the benchmark checkout only; gemot and the server are installed from their
    own repositories, so an edit to either would otherwise leave two runs indistinguishable in the
    record."""
    import importlib.metadata
    import subprocess

    from gemot.runner.provenance import _git_sha, _installed_provenance

    class _Dist:
        version = "0.1.0"

        def __init__(self, direct):
            self.direct = direct

        def read_text(self, name):
            return json.dumps(self.direct)

    def installed(direct):
        monkeypatch.setattr(importlib.metadata, "distribution", lambda name: _Dist(direct))

    installed(
        {"url": "https://example.org/s.git", "vcs_info": {"vcs": "git", "commit_id": "c0ffee"}}
    )
    assert _installed_provenance("petab-mcp") == {"version": "0.1.0", "commit": "c0ffee"}

    def git(*a):
        subprocess.check_call(["git", *a], cwd=tmp_path, stdout=subprocess.DEVNULL)

    git("init")
    (tmp_path / "server.py").write_text("one\n")
    git("add", "server.py")
    git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-m", "x")
    (tmp_path / "server.py").write_text("edited\n")
    installed({"url": tmp_path.as_uri(), "dir_info": {"editable": True}})
    code = _installed_provenance("gemot")
    assert code["git_sha"] == _git_sha(tmp_path) and code["git_sha"].endswith("+dirty")
    assert code["git_diff"]["files"] == ["server.py"]


def test_git_diff_digest_never_raises_outside_a_checkout(tmp_path):
    from gemot.runner.provenance import _git_diff_digest

    assert _git_diff_digest(tmp_path / "does-not-exist") is None
    assert _git_diff_digest(tmp_path) is None  # a real dir that is not a git tree


def test_the_history_csv_carries_the_working_tree_digest(tmp_path):
    import csv as _csv

    from gemot import metrics

    out = tmp_path / "history.csv"
    metrics.append_history(
        [
            {
                "scenario": "dirty",
                "git_sha": "abc1234+dirty",
                "git_diff": {"sha256": "f" * 64, "lines": 12, "files": ["src/x.py"], "n_files": 1},
            },
            {"scenario": "clean", "git_sha": "abc1234"},
        ],
        history_path=out,
    )
    rows = list(_csv.DictReader(out.open()))
    assert rows[0]["git_diff_sha256"] == "f" * 64
    assert rows[1]["git_diff_sha256"] == ""  # clean tree -> empty, i.e. attributable


# -- which tools were called, not just how many ------------------------------------------------
def test_the_clock_records_per_tool_call_counts():
    """`tool_calls` was a bare total, so "did the agent ever call X?" was answerable only from the
    Phoenix traces — which need a shared collector running and otherwise land in `.slurm_scratch`,
    deleted when the task completes. Job 51614722 was run to find out whether the agent called
    `get_data_regimes`; that answer is permanently unrecoverable. The name is already in hand at the
    chokepoint every call routes through, so keep it."""
    clock = _ToolClock()
    for name in ("make_changes", "make_changes", "get_data_regimes", "describe_problem"):
        clock.enter(name)
        clock.leave()
    assert clock.calls == 4
    assert clock.by_tool == {"make_changes": 2, "get_data_regimes": 1, "describe_problem": 1}


def test_an_unnamed_call_still_counts_toward_the_total():
    """The name is best-effort (it comes off the MCP envelope). A call whose name cannot be read
    must still be counted in the total rather than dropped, or the two numbers disagree."""
    clock = _ToolClock()
    clock.enter(None)
    clock.leave()
    assert clock.calls == 1 and clock.by_tool == {}
