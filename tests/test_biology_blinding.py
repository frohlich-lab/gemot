"""The anonymous arm must hide supplied identities without changing the numerical objective."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pandas as pd
import petab.v1 as petab
import pytest

from petab_harness.biology_blinding import (
    AnonymousSession,
    BlindNames,
    prepare,
    private_dir,
    read_names,
)

TOY = Path(__file__).parent / "data/toy_v1"


def test_metadata_formulas_overrides_and_source_are_blinded(tmp_path):
    import libsbml

    source = tmp_path / "source"
    shutil.copytree(TOY, source)
    document = libsbml.readSBMLFromFile(str(source / "conversion_reaction.xml"))
    model = document.getModel()
    model.setName("EGFR in U2OS cells")
    model.setSBOTerm(64)
    model.setNotes('<body xmlns="http://www.w3.org/1999/xhtml"><p>EGFR</p></body>')
    libsbml.writeSBMLToFile(document, str(source / "conversion_reaction.xml"))
    observables = pd.read_csv(source / "observables.tsv", sep="\t")
    observables["observableName"] = "EGFR phosphorylation in U2OS cells"
    observables["observableFormula"] = "observableParameter1_obs_B * B"
    observables["noiseFormula"] = "noiseParameter1_obs_B"
    observables["observableUnits"] = "phospho-EGFR counts"
    observables.to_csv(source / "observables.tsv", sep="\t", index=False)
    measurements = pd.read_csv(source / "measurements.tsv", sep="\t")
    measurements["observableParameters"] = 1.0
    measurements["noiseParameters"] = "sigma_B"
    measurements["datasetId"] = "EGFR_drug_treatment"
    measurements["replicateId"] = "U2OS_well_1"
    measurements["comment"] = "Source paper PMID 30042127"
    measurements.to_csv(source / "measurements.tsv", sep="\t", index=False)
    before = {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}

    work = tmp_path / "anonymous"
    names = prepare(source / "problem.yaml", work)
    anonymous = petab.Problem.from_yaml(str(work / "problem.yaml"))
    alias = names.forward["obs_B"]
    assert anonymous.observable_df.loc[alias, "observableFormula"] == (
        f"observableParameter1_{alias} * {names.forward['B']}"
    )
    assert anonymous.observable_df.loc[alias, "noiseFormula"] == f"noiseParameter1_{alias}"
    assert anonymous.measurement_df["noiseParameters"].unique().tolist() == [
        names.forward["sigma_B"]
    ]
    assert anonymous.measurement_df["measurement"].tolist() == measurements["measurement"].tolist()
    assert anonymous.measurement_df["time"].tolist() == measurements["time"].tolist()
    for p in work.iterdir():
        content = p.read_text()
        assert not any(
            s in content
            for s in ("EGFR", "U2OS", "PMID", "sigma_B", "obs_B", "conversion_reaction")
        )
    assert not (work / "model.ant").exists()
    assert not (work / "reference.json").exists()
    assert "sboTerm" not in (work / "model.xml").read_text()
    assert {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()} == before
    assert read_names(work).forward == names.forward


def test_aliases_do_not_capture_existing_identifiers_or_change_numeric_enums():
    names = BlindNames(reserved={"parameter_001", "input_001"})
    table = pd.DataFrame(
        {
            "parameterId": ["normal", "log10"],
            "parameterScale": ["log10", "lin"],
            "objectivePriorType": ["normal", "normal"],
            "nominalValue": [1.0, 2.0],
        }
    )
    result = names.table("parameters", table)
    assert result.parameterId.tolist() == ["parameter_002", "parameter_003"]
    assert result.parameterScale.tolist() == table.parameterScale.tolist()
    assert result.objectivePriorType.tolist() == table.objectivePriorType.tolist()
    assert result.nominalValue.tolist() == table.nominalValue.tolist()


def test_held_out_data_stays_private_and_uses_training_aliases(tmp_path):
    validation = tmp_path / "validation"
    validation.mkdir()
    measurements = pd.read_csv(TOY / "measurements.tsv", sep="\t").iloc[:1].copy()
    measurements["simulationConditionId"] = "unseen_EGFR_treatment"
    measurements.to_csv(validation / "measurements.tsv", sep="\t", index=False)
    conditions = pd.DataFrame({"conditionId": ["unseen_EGFR_treatment"], "k1": [0.2]})
    conditions.to_csv(validation / "conditions.tsv", sep="\t", index=False)
    pd.DataFrame({"conditionId": ["unseen_EGFR_treatment"], "unitId": ["patient_EGFR"]}).to_csv(
        validation / "units.tsv", sep="\t", index=False
    )
    baseline = tmp_path / "baseline"
    work = tmp_path / "anonymous"
    training_names = prepare(TOY / "problem.yaml", baseline)
    names = prepare(TOY / "problem.yaml", work, validation)
    for name in (
        "problem.yaml",
        "measurements.tsv",
        "observables.tsv",
        "parameters.tsv",
        "conditions.tsv",
        "model.xml",
    ):
        assert (baseline / name).read_bytes() == (work / name).read_bytes()
    assert all(names.forward[k] == v for k, v in training_names.forward.items())
    assert not private_dir(work).is_relative_to(work)
    heldout = pd.read_csv(private_dir(work) / "validation/measurements.tsv", sep="\t")
    assert heldout.observableId[0] == names.forward["obs_B"]
    assert heldout.simulationConditionId[0] == names.forward["unseen_EGFR_treatment"]
    heldconditions = pd.read_csv(private_dir(work) / "validation/conditions.tsv", sep="\t")
    assert heldconditions[names.forward["k1"]][0] == 0.2
    # The refit units ride along under the same aliases, or a blinded run would refit the whole
    # table as one system against a reference refit per unit.
    units = pd.read_csv(private_dir(work) / "validation/units.tsv", sep="\t")
    assert units.conditionId[0] == names.forward["unseen_EGFR_treatment"]
    assert units.unitId[0] == "patient_EGFR"


def test_judge_translates_model_arguments_and_noise_keys():
    names = BlindNames({"sigma_B": "parameter_001", "B": "entity_001", "k1": "parameter_002"})
    received = []

    async def call(name, **kwargs):
        received.append((name, kwargs))
        return {
            "noise_sigma_report": {"parameter_001": {"fitted": 0.1}},
            "antimony": "r: entity_001 => ; parameter_002 * entity_001;",
            "n": 6,
        }

    result = asyncio.run(
        names.judge_call(call)("make_changes", antimony="r: B => ; k1 * B;", n_starts=1)
    )
    assert received == [
        (
            "make_changes",
            {"antimony": "r: entity_001 => ; parameter_002 * entity_001;", "n_starts": 1},
        )
    ]
    assert result["noise_sigma_report"] == {"sigma_B": {"fitted": 0.1}}
    assert result["antimony"] == "r: B => ; k1 * B;"
    assert result["n"] == 6


def test_agent_payloads_hide_example_paths_and_keep_transport_error_codes():
    from mcp.shared.exceptions import McpError
    from mcp.types import CallToolResult, ErrorData, TextContent

    class Session:
        async def call_tool(self, name, *args, **kwargs):
            if name == "error":
                raise McpError(ErrorData(code=408, message="Timeout at /runs/Alkan/work"))
            return CallToolResult(
                content=[TextContent(type="text", text="/runs/Alkan/work/model.xml")],
                structuredContent={"path": "examples/Alkan/work", "nll": 12.5},
            )

    async def check():
        session = AnonymousSession(Session(), Path("/runs/Alkan"))
        result = await session.call_tool("get_model")
        assert result.content[0].text == "/task/work/model.xml"
        assert result.structuredContent == {"path": "/task/work", "nll": 12.5}
        with pytest.raises(McpError) as exc:
            await session.call_tool("error")
        assert exc.value.error.code == 408
        assert "Alkan" not in str(exc.value)

    asyncio.run(check())


def test_anonymous_sbml_preserves_training_and_held_out_likelihood(tmp_path, amici_build_root):
    from petab_mcp.engine import AmiciEngine
    from petab_mcp.fitting import fit_summary, score_at_nominal
    from petab_mcp.references import StubReferenceResolver
    from petab_mcp.session import PetabSession
    from petab_mcp.validation_data import read_held_out, score_session

    validation = tmp_path / "validation"
    validation.mkdir()
    measurements = pd.read_csv(TOY / "measurements.tsv", sep="\t").iloc[:1].copy()
    measurements["time"] = 1.5
    measurements["measurement"] = 0.45
    measurements.to_csv(validation / "measurements.tsv", sep="\t", index=False)
    work = tmp_path / "anonymous"
    prepare(TOY / "problem.yaml", work, validation)
    original = tmp_path / "original_problem"
    shutil.copytree(TOY, original)
    values = []
    heldout_values = []
    for label, yaml in (
        ("original", original / "problem.yaml"),
        ("anonymous", work / "problem.yaml"),
    ):
        session = PetabSession(
            yaml, provenance_dir=tmp_path / label / "provenance", resolver=StubReferenceResolver()
        )
        engine = AmiciEngine(session, build_root=str(amici_build_root))
        score_at_nominal(session, engine)
        values.append(fit_summary(session, engine))
        heldout = validation if label == "original" else private_dir(work) / "validation"
        heldout_values.append(
            score_session(session, engine, read_held_out(heldout / "measurements.tsv"))
        )
    assert values[0]["neg_log_likelihood"] == pytest.approx(
        values[1]["neg_log_likelihood"], abs=1e-9
    )
    assert values[0]["bic"] == pytest.approx(values[1]["bic"], abs=1e-9)
    assert heldout_values[0]["status"] == heldout_values[1]["status"] == "ok"
    assert heldout_values[0]["nll"] == pytest.approx(heldout_values[1]["nll"], abs=1e-9)


def test_blind_arm_replaces_scenario_context_and_closes_knowledge_tools():
    # The roster, tool curation and prompts are resolved at import, as for other arms.
    script = r"""
import asyncio
from pathlib import Path
import sys
import pytest
from petab_harness import agent, prompts
from petab_harness.scenario import Scenario
from petab_harness.agent.tools import _trim_descriptions
from types import SimpleNamespace
scenario = Scenario("Alkan", Path("/secret/Alkan/problem"), lambda *args: None,
                    task="EGFR biology", known_constants="U2OS cells", biocurator_entities=["EGFR"])
assert "EGFR" not in scenario.agent_task and "U2OS" not in scenario.agent_task
assert "biocurator" not in agent.SUBAGENT_NAMES
assert "get_skill" not in agent.ALLOWED_TOOLS
assert set(agent.ALLOWED_TOOLS) <= set(prompts.BIOLOGY_BLIND_TOOL_DESCRIPTIONS)
tools = [SimpleNamespace(name=name, description="EGFR input") for name in agent.ALLOWED_TOOLS]
tools = _trim_descriptions(tools)
assert all("EGFR" not in t.description for t in tools)
panel = agent._build_subagents({t.name:t for t in tools})
for spec in panel:
    assert "biochemistry" not in str(spec) and "EGFR" not in str(spec)
async def check():
    try:
        await agent.run_deepagents(None, model="test", api_key="test", base_url="http://invalid",
                                  system_prompt="EGFR", task="biology", recursion_limit=1)
    except ValueError as error:
        assert "anonymized server" in str(error)
    else:
        raise AssertionError("Unblinded server was accepted")
asyncio.run(check())
sys.path.insert(0, str(Path.cwd() / "tests"))
from test_run_loop_no_model import _run, _Probe
async def forbidden_discovery(*args, **kwargs):
    raise AssertionError("Biology tools were queried")
with pytest.MonkeyPatch.context() as patch:
    patch.setattr(agent.driver, "discover_bio_tools", forbidden_discovery)
    result, graph = _run(agent, patch, _Probe(model_authored=True), confirm_after=1,
                         driver_options={"biology_blinded": True, "bio_sessions": [object()],
                                         "bio_literature": True, "curated_entities": ["EGFR"],
                                         "biocurator_model": "must_not_be_used"})
assert result["biocurator_tools"] == [] and result["biocurator_entities"] == []
assert result["biocurator_model"] is None and result["biocurator_literature"] == []
assert [spec["name"] for spec in graph.captured["subagents"]] == list(agent.SUBAGENT_NAMES)
assert graph.captured["system_prompt"] == prompts.BIOLOGY_BLIND_SYSTEM_PROMPT
assert graph.inputs[0]["messages"][0]["content"] == prompts.BIOLOGY_BLIND_TASK
# The driver writes the input protocol into the blind task itself; a caller's task never
# reaches the agent, even one that starts like the composed task and carries prose after it.
protocols = [("stimulus_001", "piecewise(input_002*time + input_001, time <= input_003, 0)")]
composed = prompts.biology_blind_task(protocols)
with pytest.MonkeyPatch.context() as patch:
    patch.setattr(agent.driver, "discover_bio_tools", forbidden_discovery)
    result, graph = _run(agent, patch, _Probe(model_authored=True), confirm_after=1,
                         driver_options={"biology_blinded": True, "input_protocols": protocols,
                                         "task": composed + " EGFR in U2OS cells"})
assert graph.inputs[0]["messages"][0]["content"] == composed
assert "biolog" not in prompts.FINISH_DESCRIPTION
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        text=True,
        capture_output=True,
        env={**os.environ, "PETAB_BIOLOGY_BLIND": "1", "PETAB_BIO_LITERATURE": "1"},
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_cleanup_and_run_provenance_use_the_anonymous_view(tmp_path, monkeypatch):
    from petab_harness.runner.main import cleanup
    from petab_harness.scenario import Scenario

    source = tmp_path / "example/problem"
    shutil.copytree(TOY, source)
    scenario = Scenario("identified", source, lambda *args: None)
    monkeypatch.setenv("PETAB_BIOLOGY_BLIND", "1")
    work = cleanup(scenario)
    assert read_names(work) is not None
    assert "conversion_reaction" not in (work / "model.xml").read_text()
    from petab_harness.runner.cli import build_parser
    from petab_harness.runner.provenance import _base_meta

    meta = _base_meta(scenario, build_parser(scenario).parse_args(["--driver", "scripted"]))
    assert meta["ablations"]["biology_blind"] is True
    assert meta["biology_blinding"]["identifiers"] == read_names(work).forward
    assert meta["omnipath_snapshot"] is None
    monkeypatch.setenv("PETAB_BIOLOGY_BLIND", "0")
    cleanup(scenario)
    assert read_names(work) is None
    assert (work / "model.ant").read_bytes() == (source / "model.ant").read_bytes()


def _with_protocol(tmp_path, formula):
    """The toy problem with one condition column and a time-dependent input protocol over it."""
    source = tmp_path / "source"
    shutil.copytree(TOY, source)
    conditions = pd.read_csv(source / "conditions.tsv", sep="\t")
    conditions["dose_rate"] = 0.0
    conditions.to_csv(source / "conditions.tsv", sep="\t", index=False)
    (source / "inputs.tsv").write_text(
        f"inputId\tformula\tconditionInputs\nStimulus_EGF\t{formula}\tdose_rate\n"
    )
    return source


def test_the_input_protocol_reaches_the_blind_task_in_aliases(tmp_path):
    """Fujita's EGF is a pulse or a ramp only under its rule; its rate is constant in training.
    The visible arms are told the rule, so the blind arm must be too, without the names."""
    from petab_harness import prompts

    source = _with_protocol(tmp_path, "piecewise(dose_rate*time + 1, time <= 60, 0)")
    work = tmp_path / "anonymous"
    names = prepare(source / "problem.yaml", work)
    ((quantity, rule),) = names.protocols
    assert quantity == names.forward["Stimulus_EGF"] and quantity.startswith("stimulus_")
    assert rule == f"piecewise({names.forward['dose_rate']}*time + 1, time <= 60, 0)"
    assert read_names(work).protocols == names.protocols
    task = prompts.biology_blind_task(read_names(work).protocols)
    assert task.startswith(prompts.BIOLOGY_BLIND_TASK) and f"`{quantity} := {rule}`" in task
    assert not any(s in task for s in ("Stimulus_EGF", "EGF", "dose_rate"))
    # Nothing in the agent's working copy names the protocol; the note is the only channel.
    assert not (work / "inputs.tsv").exists()
    assert prompts.biology_blind_task([]) == prompts.BIOLOGY_BLIND_TASK


def test_a_protocol_that_reads_more_than_the_condition_columns_is_refused(tmp_path):
    source = _with_protocol(tmp_path, "piecewise(dose_rate*time + EGF_basal, time <= 60, 0)")
    with pytest.raises(ValueError, match="EGF_basal"):
        prepare(source / "problem.yaml", tmp_path / "anonymous")


def test_a_protocol_may_not_read_a_model_parameter_through_its_alias(tmp_path):
    """Every table symbol has an alias by the time a rule is translated, so a rule reading the
    parameter k1 would read as ``parameter_NNN`` -- no name, but not a condition column either."""
    source = _with_protocol(tmp_path, "piecewise(dose_rate*time + k1, time <= 60, 0)")
    with pytest.raises(ValueError, match="parameter_"):
        prepare(source / "problem.yaml", tmp_path / "anonymous")


@pytest.mark.parametrize(
    "protocol",
    [
        ("stimulus_001", "k1*time"),
        ("EGF", "input_001*time"),
        ("stimulus_001", "parameter_004 + input_001"),
    ],
)
def test_the_blind_task_refuses_a_rule_that_is_not_anonymous(protocol):
    from petab_harness import prompts

    with pytest.raises(ValueError, match="not an anonymized input protocol"):
        prompts.biology_blind_task([protocol])
