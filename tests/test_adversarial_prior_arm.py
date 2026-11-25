"""The adversarial_prior arm: ON shows the agent an earlier run's scored model and tells it to assume
that mechanism is wrong (gemot.prior_run).

The refusals are most of what is tested, because each one guards against a run whose record says one
thing while the agent saw another: a prior named with the arm off, the arm on with no prior, a prior
of another problem, or one built under another arm setting.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gemot import ablations, metrics, prior_run, prompts
from gemot.runner import barebones as R
from gemot.scenario import Scenario

_PRIOR_MODEL = "model prior\n  A -> B; kAB*A\n  kAB = 1\nend"


def _scenario(**kw):
    return Scenario(name="s", problem_dir=Path("/x/problem"), scripted_fix=None, **kw)


def _archive(root: Path, *, model_files=("model_source.ant", "model.ant"), **result) -> Path:
    """An archived run laid out as explorer.archive writes one, scoring v3."""
    run = root / "trace123"
    vdir = run / ".petab_mcp" / "versions" / "v3"
    vdir.mkdir(parents=True)
    for name in model_files:
        (vdir / name).write_text(_PRIOR_MODEL if name != "model.ant" else "server text")
    body = {
        "scenario": "s",
        "example_version": 1,
        "ablations": {a.name: a.default_on for a in ablations.ABLATIONS},
        "scored_version": 3,
        "criterion": "bic",
        "criterion_value": 123.456,
        **result,
    }
    (run / "result.json").write_text(json.dumps(body))
    return run


@pytest.fixture
def arm(monkeypatch):
    def set_(prior: Path | None = None, **switches):
        for a in ablations.ABLATIONS:
            monkeypatch.delenv(a.env, raising=False)
        monkeypatch.delenv("PETAB_PRIOR_RUN", raising=False)
        for name, on in switches.items():
            monkeypatch.setenv(
                next(a.env for a in ablations.ABLATIONS if a.name == name), "1" if on else "0"
            )
        if prior is not None:
            monkeypatch.setenv("PETAB_PRIOR_RUN", str(prior))

    return set_


def test_the_arm_is_registered_off_and_moves_the_prompt():
    a = next(a for a in ablations.ABLATIONS if a.name == "adversarial_prior")
    assert a.env == "PETAB_ADVERSARIAL_PRIOR" and not a.default_on and a.changes_prompt
    assert prompts.ADVERSARIAL_PRIOR_NOTE_TMPL in prompts.tracked_leaves()


def test_off_the_task_and_the_record_are_untouched(arm):
    arm()
    assert _scenario().agent_task == prompts.DEFAULT_TASK
    assert R.barebones_task(_scenario()) == R.BAREBONES_TASK
    assert prior_run.provenance(_scenario()) is None


def test_on_the_lead_and_the_null_are_shown_the_agents_own_text_last(arm, tmp_path):
    """The scored version's model_source.ant (what the agent wrote), not the server's model.ant."""
    arm(_archive(tmp_path), adversarial_prior=True)
    for task, base in (
        (_scenario().agent_task, prompts.DEFAULT_TASK),
        (R.barebones_task(_scenario()), R.BAREBONES_TASK),
    ):
        assert task.startswith(base) and task.endswith(f"```\n{_PRIOR_MODEL}\n```")
        assert "WRONG" in task and "(its bic was 123.5)" in task
        assert "server text" not in task


def test_the_note_follows_the_known_constants(arm, tmp_path):
    arm(_archive(tmp_path), adversarial_prior=True)
    task = _scenario(known_constants="KNOWN CONSTANTS: k.").agent_task
    assert task.index("KNOWN CONSTANTS: k.") < task.index("A RIVAL HYPOTHESIS")


def test_a_statements_run_hands_over_its_statements(arm, tmp_path):
    """An INDRA run's model.ant is a BioNetGen expansion; the agent wrote statements.json."""
    run = _archive(tmp_path, model_files=("statements.json", "model.ant"))
    (run / ".petab_mcp" / "versions" / "v3" / "statements.json").write_text('[{"rows": 1}]')
    arm(run, adversarial_prior=True)
    prior = prior_run.resolve(_scenario())
    assert prior.model_file == "statements.json" and prior.model == '[{"rows": 1}]'


def test_without_progress_signals_the_prior_score_is_withheld(arm, tmp_path):
    """progress_signals=off withholds the numbers that rank a fit; the prior's score is one."""
    state = {a.name: a.default_on for a in ablations.ABLATIONS} | {"progress_signals": False}
    arm(_archive(tmp_path, ablations=state), adversarial_prior=True, progress_signals=False)
    task = _scenario().agent_task
    assert "below. Assume" in task and "123" not in task


def test_the_record_names_the_prior(arm, tmp_path):
    run = _archive(tmp_path)
    arm(run, adversarial_prior=True)
    rec = prior_run.provenance(_scenario())
    assert rec["run_id"] == "trace123" and rec["scored_version"] == 3
    assert rec["model_file"] == "model_source.ant" and len(rec["model_sha256"]) == 64
    assert rec["criterion"] == "bic" and rec["criterion_value"] == pytest.approx(123.456)
    assert metrics._flatten_prior_run({"prior_run": rec})["prior_run_id"] == "trace123"
    assert "prior_run_id" in metrics.HISTORY_FIELDS


def test_a_prior_named_with_the_arm_off_is_refused(arm, tmp_path):
    arm(_archive(tmp_path))
    with pytest.raises(ValueError, match="arm is off"):
        prior_run.resolve(_scenario())
    # ...and recorded, not raised, by the handler that writes a failed run's result
    assert "arm is off" in prior_run.provenance(_scenario())["error"]


def test_the_arm_without_a_prior_is_refused_not_run_as_the_baseline(arm):
    arm(adversarial_prior=True)
    with pytest.raises(ValueError, match="names no prior run"):
        prior_run.resolve(_scenario())


def test_the_arm_is_refused_on_biology_blind(arm, tmp_path):
    arm(_archive(tmp_path), adversarial_prior=True, biology_blind=True)
    with pytest.raises(ValueError, match="biology_blind"):
        prior_run.resolve(_scenario())


@pytest.mark.parametrize(
    "result, match",
    [
        ({"scenario": "other"}, "not 's'"),
        ({"example_version": 2}, "version 2"),
        ({"scored_version": -1}, "scored no version"),
        ({"scored_version": None}, "scored no version"),
    ],
)
def test_a_prior_that_answers_other_data_or_scored_nothing_is_refused(arm, tmp_path, result, match):
    arm(_archive(tmp_path, **result), adversarial_prior=True)
    with pytest.raises(ValueError, match=match):
        prior_run.resolve(_scenario())


def test_a_prior_built_under_another_arm_is_refused(arm, tmp_path):
    """A model built with the reference handoff would carry it into a run without one."""
    state = {a.name: a.default_on for a in ablations.ABLATIONS}
    arm(_archive(tmp_path, ablations=state), adversarial_prior=True, reference_handoff=False)
    with pytest.raises(ValueError, match="reference_handoff"):
        prior_run.resolve(_scenario())


def test_priors_chain_an_adversarial_run_may_be_the_prior(arm, tmp_path):
    state = {a.name: a.default_on for a in ablations.ABLATIONS} | {"adversarial_prior": True}
    arm(_archive(tmp_path, ablations=state), adversarial_prior=True)
    assert prior_run.resolve(_scenario()).scored_version == 3


def test_a_directory_that_is_not_an_archived_run_is_refused(arm, tmp_path):
    arm(tmp_path, adversarial_prior=True)
    with pytest.raises(FileNotFoundError, match="no result.json"):
        prior_run.resolve(_scenario())
