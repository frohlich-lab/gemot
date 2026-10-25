"""The causal_constraints arm: the server holds every model to the problem's curated claims.

Four things have to hold for the arm to measure what it says (frohlich-lab/PEtab-MCP#134). The
server is started with the problem's constraints and with updates off, as one more fixed-arity
argument of the bootstrap, and the arm is refused where it cannot run as named. The lead is told
whose claims they are, and never holds ``set_constraints``. A refusal for breaking one is counted
beside the other refusal classes and per constraint, never as an infrastructure failure. And the
judged model's behaviour on each indirect constraint is read off a simulation of it, with MSB 2023's
gain norms.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
import types
from pathlib import Path

import pytest

from conftest import DATA_DIR, _write_empty_sbml
from gemot import ablations, infrastructure, prompts
from gemot import causal_constraints as CC
from gemot.runner import main as runner_main
from gemot.runner.clock import _ToolClock
from gemot.runner.server import _SERVER_BOOTSTRAP


# ------------------------------------------------------------------------------- the arm itself
def test_the_arm_is_registered_off_and_moves_the_prompt():
    arm = next(a for a in ablations.ABLATIONS if a.name == "causal_constraints")
    assert arm.env == "PETAB_CAUSAL_CONSTRAINTS"
    assert arm.default_on is False and arm.changes_prompt is True


def _bootstrap(monkeypatch, argv: list[str]) -> dict:
    """Run the bootstrap with the server and uvicorn stubbed; what ``create_server`` was given."""
    seen: dict = {}
    server = types.ModuleType("petab_mcp.server")
    server.create_server = lambda *a, **kw: (
        seen.update(kw) or types.SimpleNamespace(run=lambda **_: None)
    )
    uvicorn = types.ModuleType("uvicorn")
    uvicorn.Config = object
    monkeypatch.setitem(sys.modules, "petab_mcp.server", server)
    monkeypatch.setitem(sys.modules, "uvicorn", uvicorn)
    monkeypatch.setattr(sys, "argv", ["-c", *argv])
    exec(_SERVER_BOOTSTRAP, {})
    return seen


_ARGV = ["p.yaml", "build", "off", "127.0.0.1", "0", "5", "", "0", "build/entity_map.json"]


def test_the_server_gets_the_constraints_with_updates_off(monkeypatch):
    """The eleventh argument is the constraints file, "" off the arm. Updates stay off either way,
    so the server registers no `set_constraints` (#134, decision 6)."""
    seen = _bootstrap(monkeypatch, [*_ARGV, "indra/constraints.txt"])
    assert seen["constraints"] == "indra/constraints.txt"
    assert seen["constraint_updates"] == "off"
    assert _bootstrap(monkeypatch, [*_ARGV, ""])["constraints"] is None


def test_a_bootstrap_missing_the_constraints_argument_fails_at_startup(monkeypatch):
    """Fixed arity is the point: a launcher that forgot the argument must not bring a server up
    holding no constraints on an arm that claims it holds them."""
    with pytest.raises(ValueError, match="not enough values to unpack"):
        _bootstrap(monkeypatch, _ARGV)


def _scenario(tmp_path, constraints: str | None):
    problem = tmp_path / "example" / "problem"
    problem.mkdir(parents=True)
    if constraints is not None:
        (problem.parent / "indra").mkdir()
        (problem.parent / "indra" / "constraints.txt").write_text(constraints)
    return types.SimpleNamespace(
        name="toy", constraints_file=problem.parent / "indra" / "constraints.txt"
    )


def test_the_arm_passes_the_problems_constraints(monkeypatch, tmp_path):
    scenario = _scenario(tmp_path, "c1: A increases B, indirectly\n")
    monkeypatch.delenv("PETAB_CAUSAL_CONSTRAINTS", raising=False)
    assert runner_main._constraints(scenario) == ""
    monkeypatch.setenv("PETAB_CAUSAL_CONSTRAINTS", "1")
    monkeypatch.setenv("PETAB_INDRA_STATEMENTS", "1")
    assert runner_main._constraints(scenario) == str(scenario.constraints_file)


def test_the_arm_is_refused_rather_than_run_empty(monkeypatch, tmp_path):
    """A problem that ships no constraints would run the baseline under the arm's name."""
    monkeypatch.setenv("PETAB_CAUSAL_CONSTRAINTS", "1")
    monkeypatch.setenv("PETAB_INDRA_STATEMENTS", "1")
    with pytest.raises(FileNotFoundError, match="ships no constraints"):
        runner_main._constraints(_scenario(tmp_path, None))


def test_the_refusal_comes_before_anything_runs(monkeypatch, tmp_path):
    """Raised from `run_scenario` itself, so the working copy is untouched and the error is not
    filed as a teardown failure by the catch around the pipeline."""
    from gemot.runner.cli import run_scenario
    from gemot.scenario import Scenario

    problem = tmp_path / "example" / "problem"
    problem.mkdir(parents=True)
    monkeypatch.setenv("PETAB_CAUSAL_CONSTRAINTS", "1")
    monkeypatch.setenv("PETAB_INDRA_STATEMENTS", "1")
    with pytest.raises(FileNotFoundError, match="ships no constraints"):
        run_scenario(
            Scenario(name="toy", problem_dir=problem, scripted_fix=None), ["--driver", "scripted"]
        )
    assert not (problem.parent / "work").exists()


def test_the_arm_needs_the_statements_route(monkeypatch, tmp_path):
    """The server checks constraints on the statements route's rows, and refuses them without the
    entity map only indra_statements passes."""
    monkeypatch.setenv("PETAB_CAUSAL_CONSTRAINTS", "1")
    monkeypatch.setenv("PETAB_INDRA_STATEMENTS", "0")
    with pytest.raises(ValueError, match="needs indra_statements"):
        runner_main._constraints(_scenario(tmp_path, "c1: A increases B, indirectly\n"))


def test_the_lead_is_told_whose_claims_they_are(monkeypatch):
    from gemot.scenario import Scenario

    scenario = Scenario(name="toy", problem_dir=Path("problem"), scripted_fix=None)
    monkeypatch.setenv("PETAB_INDRA_STATEMENTS", "1")
    monkeypatch.setenv("PETAB_CAUSAL_CONSTRAINTS", "1")
    assert scenario.agent_task.startswith(prompts.INDRA_TASK + prompts.INDRA_CONSTRAINTS_NOTE)
    monkeypatch.setenv("PETAB_CAUSAL_CONSTRAINTS", "0")
    assert prompts.INDRA_CONSTRAINTS_NOTE not in scenario.agent_task
    assert prompts.INDRA_CONSTRAINTS_NOTE in prompts.tracked_leaves()


def _fresh(script: str, **switches: str) -> str:
    """Run ``script`` in a fresh interpreter with every arm at its default but ``switches``: the
    panel's tool lists are resolved as the agent package imports."""
    import os
    import subprocess

    env = {**os.environ, **{a.env: str(int(a.default_on)) for a in ablations.ABLATIONS}, **switches}
    done = subprocess.run(
        [sys.executable, "-c", script], env=env, text=True, capture_output=True, timeout=90
    )
    assert done.returncode == 0, done.stdout + done.stderr
    return done.stdout


def test_the_lead_reads_the_constraints_on_the_arm_alone():
    """`list_constraints` is how the lead reads them (describe_problem only counts them); off the
    arm the server offers no such tool, and the panel would warn of one it expected."""
    script = "from gemot.agent import tools; print('list_constraints' in tools.ORCHESTRATOR_TOOLS)"
    on = {"PETAB_INDRA_STATEMENTS": "1", "PETAB_CAUSAL_CONSTRAINTS": "1"}
    assert _fresh(script, **on).strip() == "True"
    assert _fresh(script).strip() == "False"


def test_no_agent_holds_set_constraints():
    """An arm in which the agent may change the claims it is held to is a decision of its own: it
    would write its own guard. The server registers the tool only with updates on, and no panel
    allowlist names it either."""
    script = "from gemot.agent import tools; print('set_constraints' in tools.ALLOWED_TOOLS)"
    on = {"PETAB_INDRA_STATEMENTS": "1", "PETAB_CAUSAL_CONSTRAINTS": "1"}
    assert _fresh(script, **on).strip() == "False"


# ------------------------------------------------------------------------------------ refusals
class _Envelope:
    """The shape ``session.call_tool`` returns: content blocks carrying a JSON body."""

    def __init__(self, payload: dict) -> None:
        self.content = [types.SimpleNamespace(text=json.dumps(payload))]


#: A refusal as ``petab_mcp.constraints.enforce`` words it, the why texts as fujita's had them.
_REFUSAL = (
    "make_changes: the model breaks 2 of the session's constraints: `c2` (EGFR{pY1068} "
    "phosphorylates AKT1 at S473, directly): rows `r23`, `r7` change AKT1{pS473}, and none has "
    "EGFR{pY1068} drive it: state it on the row it acts on, as its catalyst or a regulator "
    "(`..., inhibited by EGFR{pY1068}`, `activated by`); `c1` (EGF increases RPS6{pS235}, "
    "indirectly): its claims connect them only the other way. NOTHING was applied."
)


def test_a_constraint_refusal_is_a_refusal_counted_per_constraint():
    clock = _ToolClock()
    refused = {"applied": False, "error": _REFUSAL, "error_cause": CC.CONSTRAINT_VIOLATED}
    clock.note_result("make_changes", _Envelope(refused))
    clock.note_result(
        "make_changes", _Envelope({**refused, "error": _REFUSAL.replace("2 of", "1 of")})
    )
    clock.note_result("make_changes", _Envelope({"applied": True, "committed": True, "version": 1}))
    summary = clock.reject_summary()
    assert summary["rejected"] == 2 and summary["by_cause"] == {CC.CONSTRAINT_VIOLATED: 2}
    assert clock.by_constraint == {"c2": 2, "c1": 2}
    assert clock.first_commit_edit == 3


def test_a_constraint_refusal_is_never_an_infrastructure_failure():
    """The circuit breaker and the log watch read every result for the infrastructure markers; a
    refusal names the user's claims and the agent's rows, and must stay the agent's error."""
    assert infrastructure.find(_REFUSAL) is None


def test_the_refused_ids_are_the_constraints_not_the_rows_their_reasons_name():
    """The reasons name rows in backticks too (`r23`); only a constraint is followed by its text."""
    assert CC.broken_constraints(_REFUSAL) == ["c2", "c1"]


def test_the_degraded_result_keeps_the_refusals(monkeypatch, tmp_path):
    """An abort loses the judge, not what the agent was refused."""
    from gemot.runner.failure import _write_degraded_result

    monkeypatch.setenv("PETAB_CAUSAL_CONSTRAINTS", "1")
    clock = _ToolClock()
    clock.note_result(
        "make_changes",
        _Envelope({"applied": False, "error": _REFUSAL, "error_cause": CC.CONSTRAINT_VIOLATED}),
    )
    result = _write_degraded_result(
        tmp_path, {}, types.SimpleNamespace(name="toy"), "boom", clock=clock
    )
    assert result["causal_constraints"] == {
        "refusals": 1,
        "refusals_by_constraint": {"c2": 1, "c1": 1},
    }


def test_the_cost_reaches_the_history_columns():
    from gemot import metrics

    block = {"refusals": 3, "first_satisfying_version": 1, "edits_to_first_satisfying": 4}
    assert metrics.flatten_constraints({"causal_constraints": block}) == {
        "constraint_refusals": 3,
        "constraint_first_satisfying_version": 1,
        "constraint_edits_to_first_satisfying": 4,
    }
    assert metrics.flatten_constraints({"causal_constraints": None}) == {}
    assert set(metrics._CONSTRAINT_FIELDS) <= set(metrics.HISTORY_FIELDS)


# ----------------------------------------------------------------------------------- behaviour
def test_the_gain_norms_are_one_without_an_effect():
    t = [0.0, 0.1, 1.0, 10.0]
    assert CC.gains([(t, [0, 1, 2, 1])], [(t, [0, 1, 2, 1])]) == {"linf": 1.0, "l1_logt": 1.0}


def test_the_gain_norms_pool_experiments_and_weigh_log_time():
    """L∞ is the largest amount over every experiment; L1 is over log-time, so a decade early
    weighs what a decade late does."""
    t = [0.0, 1.0, 10.0, 100.0]
    base = [(t, [0, 1, 1, 1]), (t, [0, 2, 2, 2])]
    early = [(t, [0, 3, 1, 1]), (t, [0, 2, 2, 2])]
    late = [(t, [0, 1, 1, 3]), (t, [0, 2, 2, 2])]
    assert CC.gains(base, early)["linf"] == pytest.approx(1.5)
    assert CC.gains(base, early)["l1_logt"] == pytest.approx(CC.gains(base, late)["l1_logt"])


def test_an_object_absent_at_baseline_has_an_unbounded_gain():
    t = [0.0, 1.0, 10.0]
    assert CC.gains([(t, [0, 0, 0])], [(t, [0, 1, 1])])["linf"] == float("inf")


@pytest.mark.parametrize(
    "gain,expected",
    [(1.2, "as_claimed"), (0.8, "opposite"), (float("inf"), "as_claimed"), (1 + 1e-9, "none")],
)
def test_a_gain_reads_as_claimed_opposite_or_none(gain, expected):
    assert CC.verdict(gain) == expected


def test_each_constraint_is_read_as_the_server_reads_it():
    """The subject and the forms it must move, each the way it must move it, are the server's
    reading (`petab_mcp.constraints.Constraint`): a removal lowers the modified form, an inactive
    form falls where its active one rises; and only a signed indirect constraint is read."""
    from petab_mcp import constraints as K

    parsed = K.parse(
        [
            "a1: RPS6{pS235} is active",
            "a2: RPS6 is inactive",
            "c1: EGF increases RPS6{pS235}, indirectly",
            "c2: EGFR{pY1068} phosphorylates AKT1 at S473, directly",
            "c3: EGF activates RPS6, indirectly",
            "c4: AKT1 dephosphorylates RPS6 at S235, indirectly",
            "c5: EGF does not affect AKT1, indirectly",
        ]
    )
    read = {c.id: CC.claim(c) for c in parsed.items}
    assert read["c1"] == {
        "claim": "EGF increases RPS6{pS235}",
        "subject": "EGF",
        "objects": [("RPS6{pS235}", 1)],
    }
    assert isinstance(read["c2"], str) and isinstance(read["c5"], str)
    assert sorted(read["c3"]["objects"]) == [("RPS6", -1), ("RPS6{pS235}", 1)]
    assert read["c4"]["objects"] == [("RPS6{pS235}", -1)]


def _toy(tmp_path: Path) -> Path:
    """The toy problem without a model, its observable a readout and no starts shipped."""
    work = tmp_path / "work"
    work.mkdir()
    for name in ("problem.yaml", "conditions.tsv", "measurements.tsv"):
        shutil.copy(DATA_DIR / name, work / name)
    _write_empty_sbml(work / "conversion_reaction.xml")
    (work / "observables.tsv").write_text(
        "observableId\tobservableFormula\tnoiseFormula\nobs_B\treadout_B\tsigma_B\n"
    )
    (work / "parameters.tsv").write_text(
        "parameterId\tparameterScale\tlowerBound\tupperBound\tnominalValue\testimate\n"
        "sigma_B\tlin\t0.0001\t100.0\t0.1\t0\n"
    )
    return work


def test_the_judged_models_behaviour_is_read_off_its_simulation(tmp_path):
    """End to end on a stored version: the server builds a phosphorylation cascade under the
    constraints, and the record finds the subject and object species by the server's own reading,
    perturbs the subject and reads the object's gain. Raising A raises B{pS10} (a plain cascade);
    the direct constraint and the one claiming no sign are not read."""
    pytest.importorskip("amici")
    from petab_mcp.server import create_server

    work = _toy(tmp_path)
    (tmp_path / "entities.json").write_text("{}\n")
    constraints = tmp_path / "constraints.txt"
    constraints.write_text(
        "c1: A increases B{pS10}, indirectly\n"
        "c2: A phosphorylates B at S10, directly\n"
        "c3: B does not affect A, indirectly\n"
    )
    mcp = create_server(
        str(work / "problem.yaml"),
        build_root=str(tmp_path / "build"),
        auto_fit=False,
        entity_map=str(tmp_path / "entities.json"),
        constraints=str(constraints),
    )
    out = asyncio.run(
        mcp.call_tool(
            "make_changes",
            {
                "justification": "a cascade",
                "statements": [
                    "phos: A phosphorylates B at S10 | default",
                    "dephos: B is dephosphorylated at S10 | basal",
                ],
                "readouts": {"readout_B": "amount(B{pS10})"},
            },
        )
    ).structured_content
    assert out.get("committed") is True, out
    clock = _ToolClock()
    clock.note_result("make_changes", _Envelope(out))
    got = CC.record(
        work, constraints, clock, out["version"], entity_map=None, build_dir=tmp_path / "metric"
    )
    assert got["refusals"] == 0 and got["refusals_by_constraint"] == {"c1": 0, "c2": 0, "c3": 0}
    assert got["first_satisfying_version"] == out["version"]
    assert got["edits_to_first_satisfying"] == 1
    c1 = got["behaviour"]["c1"]
    assert c1["linf"]["verdict"] == "as_claimed" and c1["l1_logt"]["verdict"] == "as_claimed"
    assert c1["linf"]["gain"] > 1 and c1["perturbation"] == ["raised"]
    assert set(got["behaviour"]["c2"]) == set(got["behaviour"]["c3"]) == {"skipped"}
