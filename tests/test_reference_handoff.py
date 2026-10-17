"""The reference_handoff ablation: OFF withholds what a problem's author wrote knowing the reference
model, the known-constants note and the biocurator's curated entity names, from every arm that
reads them (the panel's lead and biocurator, and the null).

And the indra_statements arm's task: the default task's READOUTS paragraph and the readout note
both teach the Antimony route's assignment rules (`set_assignment`, `readout_x := ...`), which the
arm's editor does not take; the arm's system prompt defines readouts through `readouts`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gemot import ablations, prompts
from gemot.runner import barebones as R
from gemot.scenario import Scenario

_NOTE = "KNOWN CONSTANTS: STAT5_total is the total STAT5."


def _scenario(**kw):
    return Scenario(
        name="s",
        problem_dir=Path("/x/problem"),
        scripted_fix=None,
        known_constants=_NOTE,
        biocurator_entities=["STAT5A", "STAT5B"],
        **kw,
    )


@pytest.fixture
def arm(monkeypatch):
    def set_(**switches):
        for a in ablations.ABLATIONS:
            monkeypatch.delenv(a.env, raising=False)
        for name, on in switches.items():
            monkeypatch.setenv(
                next(a.env for a in ablations.ABLATIONS if a.name == name), "1" if on else "0"
            )

    return set_


def test_the_arm_is_registered_on_and_moves_the_prompt():
    a = next(a for a in ablations.ABLATIONS if a.name == "reference_handoff")
    assert a.env == "PETAB_REFERENCE_HANDOFF" and a.default_on and a.changes_prompt


def test_by_default_the_handoff_reaches_the_lead_and_the_biocurator(arm):
    arm()
    s = _scenario()
    assert _NOTE in s.agent_task and s.agent_entities == ["STAT5A", "STAT5B"]


def test_off_the_lead_gets_the_bare_task_and_the_biocurator_no_names(arm):
    arm(reference_handoff=False)
    s = _scenario()
    assert s.agent_task == prompts.DEFAULT_TASK
    assert s.agent_entities == []


def test_off_the_null_gets_the_bare_task(arm):
    arm(reference_handoff=False)
    assert R.barebones_task(_scenario()) == R.BAREBONES_TASK


def test_the_indra_arm_task_teaches_no_assignment_rules(arm):
    arm(indra_statements=True)
    task = _scenario().agent_task
    assert task.startswith(prompts.INDRA_TASK) and _NOTE in task
    assert "set_assignment" not in task and ":=" not in task
    assert prompts.INDRA_TASK in prompts.tracked_leaves()


def test_the_indra_arm_without_the_handoff_is_its_task_alone(arm):
    arm(indra_statements=True, reference_handoff=False)
    assert _scenario().agent_task == prompts.INDRA_TASK
