"""The NULL arm: the barebones pipeline's staging, delivery checks and wiring.

The barebones run has no server watching the agent work, so the rules the server ENFORCES on the
lead (the design is fixed, a shipped constant stays fixed, the model is what the YAML names) are
checked here after the fact — and a null that could quietly rewrite `observables.tsv` or read the
reference model out of `run.py` would not be a null. These tests pin those checks, the workspace
staging that withholds the answer, and the sweep/CLI plumbing that lets `--driver barebones` run
in the same array as a `deepagents` sweep.
"""

from __future__ import annotations

import asyncio
import json
import re
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("mcp", reason="harness-only dependency")

from gemot.runner import barebones as R  # noqa: E402

TOY = Path(__file__).parent / "data" / "toy_v1"

#: What a problem directory may carry beside the problem, none of which the agent may see: the
#: reference's score, its held-out score, and a file no rule names.
_NOT_THE_PROBLEM = ("reference.json", "validation_v1_baseline.json", "run.py")


def _add_the_answers(work: Path) -> None:
    (work / "reference.json").write_text('{"nll": 1.0, "k": 2, "n": 10}')
    (work / "validation_v1_baseline.json").write_text('{"nll": 3.0, "n": 4}')
    (work / "run.py").write_text("# the reference model\n")


def _workspace(tmp_path: Path) -> Path:
    """A `work/` as `cleanup()` leaves it: the problem plus the files the agent must not see."""
    work = tmp_path / "ex" / "work"
    shutil.copytree(TOY, work)
    _add_the_answers(work)
    # TOY ships a model.ant; the staging must remove it whatever it holds.
    assert (work / "model.ant").is_file()
    return work


# --------------------------------------------------------------------------- staging
def test_staging_withholds_the_answer_and_the_seed(tmp_path):
    """`reference.json` is the true model's score and `model.ant` would be loaded INSTEAD of the
    agent's SBML by the judge's server — neither may be in the workspace."""
    work = _workspace(tmp_path)
    staged = R.stage_workspace(work)
    agent = staged["agent_dir"]
    assert agent == work / R.AGENT_SUBDIR
    assert sorted(staged["withheld"]) == sorted(["model.ant", *_NOT_THE_PROBLEM])
    assert not (agent / "reference.json").exists() and not (agent / "model.ant").exists()
    assert not (work / "reference.json").exists()
    # the design tables the YAML lists are fingerprinted, the parameter table is not a design table
    assert set(staged["design_hashes"]) == {"measurements.tsv", "conditions.tsv", "observables.tsv"}
    assert staged["shipped_rows"]["sigma_B"] == (0, 0.1)
    assert staged["shipped_rows"]["k1"][0] == 1


def test_staging_tolerates_an_already_clean_workspace(tmp_path):
    work = tmp_path / "work"
    shutil.copytree(TOY, work)
    (work / "model.ant").unlink()
    assert R.stage_workspace(work)["withheld"] == []


@pytest.mark.parametrize("version", [1, 2])
def test_only_the_problem_reaches_the_agent(tmp_path, version):
    """An allowlist, not a denylist. The staging used to withhold only `reference.json` and the
    seed, so the rest of the problem directory reached the agent: `validation_v1_baseline.json`,
    the reference's held-out score and its parameters by name and fitted value, which
    PEtabAgentBench tracks for perelson, crauste, fujita, lucarelli and raimundez. The agent gets `problem.yaml`, the files it lists (the model under
    v1's `sbml_files` or v2's `model_files`) and `inputs.tsv`; nothing else stays in `work/`."""
    work = (_workspace if version == 1 else _workspace_v2)(tmp_path)
    (work / "inputs.tsv").write_text("inputId\tformula\tconditionInputs\n")
    (work / "indra").mkdir()
    (work / "indra" / "entity_map.json").write_text("{}\n")
    staged = R.stage_workspace(work)
    design = {"measurements.tsv", "conditions.tsv", "observables.tsv"}
    if version == 2:
        design.add("experiments.tsv")
    assert {f.name for f in staged["agent_dir"].iterdir()} == {
        "problem.yaml",
        "conversion_reaction.xml",
        "parameters.tsv",
        "inputs.tsv",
        *design,
        R.SANDBOX_HOME_DIR,
        R.SANDBOX_TMP_DIR,
    }
    assert {*_NOT_THE_PROBLEM, "indra"} <= set(staged["withheld"])
    assert [f.name for f in work.iterdir()] == [R.AGENT_SUBDIR]


def test_the_agent_dir_holds_only_the_problem_even_after_the_live_marker(tmp_path):
    """The live-run marker carries the reference score (`reference`, `bic_reference`), and the
    server logs and result.json land beside it; all of that is written to `work/`, which is NOT
    the mount — the agent sees `work/agent/` and nothing above it."""
    from gemot.runner.provenance import _write_live_marker

    work = _workspace(tmp_path)
    staged = R.stage_workspace(work)
    agent = staged["agent_dir"]

    class _S:
        name = "toy"
        version = 1
        bic_threshold = 10.0

        def reference(self):
            return {"nll": 1.0, "k": 2, "n": 10}

    _write_live_marker(work, {"phoenix_trace_id": "abc"}, _S(), include_reference=False)
    marker = json.loads((work / "run_live.json").read_text())
    assert marker["reference"] is None and marker["bic_reference"] is None
    visible = {f.name for f in agent.iterdir()}
    listed = {n for names in staged["files"].values() for n in names}
    assert visible == listed | {
        "problem.yaml",
        "conversion_reaction.xml",
        R.SANDBOX_HOME_DIR,
        R.SANDBOX_TMP_DIR,
    }
    assert "run_live.json" not in visible


# --------------------------------------------------------------------------- delivery checks
def test_an_untouched_delivery_is_clean(tmp_path):
    work = _workspace(tmp_path)
    staged = R.stage_workspace(work)
    work = staged["agent_dir"]
    out = R.check_delivery(work, staged)
    assert out["design_modified"] == [] and out["shipped_rows_violated"] == []
    assert out["petab_lint"]["ok"], out["petab_lint"]
    assert out["model_authored"] is True  # the toy ships a real model
    assert out["model_ant_set_aside"] is False
    # reconciled, the toy estimates what it ships estimated, and nothing else
    assert out["estimated"] == ["k1", "k2"] and out["uncharged"] == []


def test_a_rewritten_design_table_is_reported(tmp_path):
    """Byte-level: the agent was told not to touch these files at all, and a re-saved table with
    different float formatting is still a table the agent rewrote."""
    work = _workspace(tmp_path)
    staged = R.stage_workspace(work)
    work = staged["agent_dir"]
    meas = work / "measurements.tsv"
    meas.write_text(meas.read_text() + "\n")
    out = R.check_delivery(work, staged)
    assert out["design_modified"] == ["measurements.tsv"]


def test_a_design_table_dropped_from_the_yaml_is_reported(tmp_path):
    work = _workspace(tmp_path)
    staged = R.stage_workspace(work)
    work = staged["agent_dir"]
    yaml_text = (work / "problem.yaml").read_text().replace("- observables.tsv", "- obs2.tsv")
    shutil.copy(work / "observables.tsv", work / "obs2.tsv")
    (work / "problem.yaml").write_text(yaml_text)
    out = R.check_delivery(work, staged)
    assert "observables.tsv" in out["design_modified"]


def test_a_design_table_listed_beside_the_staged_ones_is_reported(tmp_path):
    """#1: an extra measurement file beside the untouched original leaves every staged hash intact,
    and the rows it adds would lower the score of a model that reproduces them."""
    work = _workspace(tmp_path)
    staged = R.stage_workspace(work)
    work = staged["agent_dir"]
    yaml_text = (work / "problem.yaml").read_text()
    yaml_text = yaml_text.replace("  - measurements.tsv", "  - measurements.tsv\n  - more.tsv")
    (work / "problem.yaml").write_text(yaml_text)
    shutil.copy(work / "measurements.tsv", work / "more.tsv")
    out = R.check_delivery(work, staged)
    assert out["design_modified"] == ["more.tsv"]


def _edit_model(work: Path, edit) -> None:
    """Apply ``edit(model)`` to the delivered toy SBML and write it back."""
    import libsbml

    path = work / "conversion_reaction.xml"
    doc = libsbml.readSBMLFromFile(str(path))
    edit(doc.getModel())
    libsbml.writeSBMLToFile(doc, str(path))


def _add(model, kind: str, sid: str, value: float | None = None):
    """A parameter (``value``) or a species (initial concentration ``value``, unset if None)."""
    if kind == "parameter":
        p = model.createParameter()
        p.setId(sid)
        p.setConstant(True)
        p.setValue(value)
        return p
    sp = model.createSpecies()
    sp.setId(sid)
    sp.setCompartment("compartment_")
    sp.setHasOnlySubstanceUnits(False)
    sp.setBoundaryCondition(False)
    sp.setConstant(False)
    if value is not None:
        sp.setInitialConcentration(value)
    return sp


def _add_decay(model, rid: str, species: list[str], rate: str):
    """``species -> ;`` at ``rate``, so every name in it is read by the model."""
    import libsbml

    rxn = model.createReaction()
    rxn.setId(rid)
    rxn.setReversible(False)
    for sid in species:
        ref = rxn.createReactant()
        ref.setSpecies(sid)
        ref.setStoichiometry(1)
        ref.setConstant(True)
    rxn.createKineticLaw().setMath(libsbml.parseL3Formula(rate))
    return rxn


def _add_rows(work: Path, *rows: str) -> None:
    with (work / "parameters.tsv").open("a") as fh:
        fh.writelines(row + "\n" for row in rows)


def test_numbers_outside_the_estimated_rows_are_charged(tmp_path):
    """#1: the panel's server gives every parameter a model declares an ESTIMATED row and every
    nonzero, non-unit initial value an estimated `<sid>_init`. Scored as shipped, the same numbers
    were free to the null written as an SBML value with no row, as a fixed row the problem does not
    ship, or as a numeric initial value. Run through the server's reconcile against the shipped
    table, each comes back estimated; a parameter nothing reads is pruned, as for the panel, so an
    estimated row for it no longer counts."""
    work = _workspace(tmp_path)
    staged = R.stage_workspace(work)
    work = staged["agent_dir"]

    def edit(m):
        for pid, value in (("kx", 0.37), ("kf", 0.2), ("kd", 0.05), ("unused", 5.0)):
            _add(m, "parameter", pid, value)
        _add(m, "species", "C", 3.7)
        _add_decay(m, "r3", ["C"], "kd * kx * kf * C")

    _edit_model(work, edit)
    _add_rows(
        work,
        "kd\tlog10\t0.0001\t100.0\t0.05\t1",
        "kf\tlin\t0.0\t1.0\t0.2\t0",
        "unused\tlog10\t0.0001\t100.0\t5.0\t1",
    )
    delivered = {f.name: f.read_bytes() for f in work.iterdir() if f.is_file()}
    out = R.check_delivery(work, staged)
    assert out["petab_lint"]["ok"], out["petab_lint"]
    # the reconcile runs on a copy: the server still scores the files as delivered
    assert {f.name: f.read_bytes() for f in work.iterdir() if f.is_file()} == delivered
    assert out["estimated"] == sorted(["k1", "k2", "kd", "kx", "kf", "C_init"])
    assert out["uncharged"] == [
        "C_init = 3.7: no parameter-table row",
        "kf = 0.2: estimate false -> true",
        "kx = 0.37: no parameter-table row",
        "unused: pruned, nothing reads it",
    ]


def test_the_panels_free_numbers_stay_free(tmp_path):
    """What the reconcile leaves fixed costs the null nothing either: a start at 0, unset, or 1 —
    written as a number or through the parameters of its initial assignment (`D = D0` with D0 at 1,
    `H = H_tot*H_fold` with both at 0 or 1), with or without a fixed row — and a value a rule
    computes (it cannot have a row at all)."""
    import libsbml

    work = _workspace(tmp_path)
    staged = R.stage_workspace(work)
    work = staged["agent_dir"]

    def edit(m):
        for sid, value in (("D", None), ("D2", None), ("E", 1.0), ("F", 0.0), ("G", None)):
            _add(m, "species", sid, value)
        _add(m, "species", "H")
        for pid, value in (("D0", 1.0), ("D20", 0.0), ("H_tot", 0.0), ("H_fold", 1.0)):
            _add(m, "parameter", pid, value)
        _add(m, "parameter", "p_rule", 0.0).setConstant(False)
        for sid, math in (("D", "D0"), ("D2", "D20"), ("H", "H_tot * H_fold")):
            ia = m.createInitialAssignment()
            ia.setSymbol(sid)
            ia.setMath(libsbml.parseL3Formula(math))
        rule = m.createAssignmentRule()
        rule.setVariable("p_rule")
        rule.setMath(libsbml.parseL3Formula("2 * k2"))
        _add_decay(m, "r3", ["D", "D2", "E", "F", "G", "H"], "k1 * p_rule * D * D2 * E * F * G * H")

    _edit_model(work, edit)
    _add_rows(work, "D20\tlin\t0.0\t1.0\t0.0\t0")
    out = R.check_delivery(work, staged)
    assert out["petab_lint"]["ok"], out["petab_lint"]
    assert out["estimated"] == ["k1", "k2"] and out["uncharged"] == []


def test_the_delivered_value_decides_the_unit_start(tmp_path):
    """The score reads the table, not the SBML: `D = D0` declared at 1 but delivered estimated at
    3.2 is a fitted start, and stays charged, where the SBML's 1 alone would read as a unit pool."""
    import libsbml

    work = _workspace(tmp_path)
    staged = R.stage_workspace(work)
    work = staged["agent_dir"]

    def edit(m):
        _add(m, "species", "D")
        _add(m, "parameter", "D0", 1.0)
        ia = m.createInitialAssignment()
        ia.setSymbol("D")
        ia.setMath(libsbml.parseL3Formula("D0"))
        _add_decay(m, "r3", ["D"], "k1 * D")

    _edit_model(work, edit)
    _add_rows(work, "D0\tlog10\t0.0001\t100.0\t3.2\t1")
    out = R.check_delivery(work, staged)
    assert out["petab_lint"]["ok"], out["petab_lint"]
    assert out["estimated"] == ["D0", "k1", "k2"] and out["uncharged"] == []


def test_a_parameter_only_the_condition_table_reads_is_charged(tmp_path):
    """The fixed tables read model symbols too, and the reconcile never prunes what they read: a
    condition that sets `A` to `A_dose` makes the model's `A_dose` a number of the fit."""
    work = _workspace(tmp_path)
    (work / "conditions.tsv").write_text("conditionId\tA\nc0\tA_dose\n")
    staged = R.stage_workspace(work)
    work = staged["agent_dir"]
    _edit_model(work, lambda m: _add(m, "parameter", "A_dose", 2.5))
    out = R.check_delivery(work, staged)
    assert out["petab_lint"]["ok"], out["petab_lint"]
    assert out["uncharged"] == ["A_dose = 2.5: no parameter-table row"]


def test_shipped_rows_are_protected(tmp_path):
    """A fixed shipped constant re-valued or freed, or a shipped row removed, is a violation; the
    agent's OWN new rows are not checked (it may add whatever it estimates)."""
    work = _workspace(tmp_path)
    staged = R.stage_workspace(work)
    work = staged["agent_dir"]
    lines = (work / "parameters.tsv").read_text().splitlines()
    header, rows = lines[0], lines[1:]
    edited = []
    for row in rows:
        cells = row.split("\t")
        if cells[0] == "sigma_B":
            cells[4] = "0.5"  # fixed row re-valued
        elif cells[0] == "A_init":
            cells[5] = "1"  # fixed row freed
        elif cells[0] == "B_init":
            continue  # shipped row removed
        edited.append("\t".join(cells))
    edited.append("kcat_new\tlog10\t1e-3\t1e3\t2.5\t1")  # the agent's own row: fine
    (work / "parameters.tsv").write_text("\n".join([header, *edited]) + "\n")
    out = R.check_delivery(work, staged)
    assert sorted(out["shipped_rows_violated"]) == sorted(
        ["sigma_B: fixed value 0.1 -> 0.5", "A_init: estimate 0 -> 1", "B_init: removed"]
    )


def test_an_agent_model_ant_is_set_aside_before_scoring(tmp_path):
    """The server prefers a model.ant beside the YAML over the SBML; the deliverable is the PEtab
    problem the YAML names, so a model.ant the agent wrote is moved out of the way, not scored."""
    work = _workspace(tmp_path)
    staged = R.stage_workspace(work)
    work = staged["agent_dir"]
    (work / "model.ant").write_text("model m\nend\n")
    out = R.check_delivery(work, staged)
    assert out["model_ant_set_aside"] is True
    assert not (work / "model.ant").exists() and (work / "model.ant.agent").is_file()


def test_an_unloadable_problem_fails_lint_not_the_runner(tmp_path):
    work = _workspace(tmp_path)
    staged = R.stage_workspace(work)
    work = staged["agent_dir"]
    (work / "conversion_reaction.xml").write_text("<sbml>not a model")
    out = R.check_delivery(work, staged)
    assert out["petab_lint"]["ok"] is False and out["petab_lint"]["messages"]
    assert out["model_authored"] is None


def test_lint_names_the_libsbml_errors(tmp_path):
    """petab's linter says only "Not OK" for an SBML libsbml rejects; the delivery record must carry
    the errors themselves — a hand-written L3V2 model missing the required `constant` /
    `hasOnlySubstanceUnits` / `boundaryCondition` / `reversible` attributes is the observed case."""
    work = _workspace(tmp_path)
    staged = R.stage_workspace(work)
    work = staged["agent_dir"]
    (work / "conversion_reaction.xml").write_text("""<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
  <model id="conversion_reaction">
    <listOfCompartments><compartment id="compartment_" size="1"/></listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="compartment_" initialConcentration="1"/>
      <species id="B" compartment="compartment_" initialConcentration="0"/>
    </listOfSpecies>
    <listOfParameters><parameter id="k1" value="0.5"/><parameter id="k2" value="0.1"/></listOfParameters>
    <listOfReactions>
      <reaction id="r1"><listOfReactants><speciesReference species="A" stoichiometry="1" constant="true"/></listOfReactants>
        <listOfProducts><speciesReference species="B" stoichiometry="1" constant="true"/></listOfProducts>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><times/><ci>k1</ci><ci>A</ci></apply></math></kineticLaw></reaction>
    </listOfReactions>
  </model>
</sbml>
""")
    out = R.check_delivery(work, staged)
    assert out["petab_lint"]["ok"] is False
    assert any(m.startswith("SBML error 20623") for m in out["petab_lint"]["messages"]), out[
        "petab_lint"
    ]
    assert out["model_authored"] is True


def test_an_untouched_model_less_problem_reads_as_no_model(tmp_path):
    """An agent that authors nothing leaves a problem whose own lint fails (the observables
    reference readout_* symbols nothing defines) — the record must say the model is missing, not
    that the delivery is malformed."""
    work = _workspace(tmp_path)
    staged = R.stage_workspace(work)
    work = staged["agent_dir"]
    (work / "conversion_reaction.xml").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">'
        '<model id="conversion_reaction"/></sbml>\n'
    )
    out = R.check_delivery(work, staged)
    assert out["model_authored"] is False


def _workspace_v2(tmp_path: Path) -> Path:
    """`_workspace` for a native PEtab v2 problem: the toy with one condition, upgraded."""
    from petab.v2.petab1to2 import petab1to2

    source = tmp_path / "v1"
    shutil.copytree(TOY, source)
    (source / "conditions.tsv").write_text("conditionId\tA\nc0\t1.0\n")
    work = tmp_path / "ex" / "work"
    petab1to2(source / "problem.yaml", work, validate=False)
    _add_the_answers(work)
    return work


def test_a_v2_delivery_is_checked_against_its_own_tables(tmp_path):
    """A v2 YAML lists its tables at the top level, adds an experiment table to the design, and
    spells `estimate` true/false; untouched, it is clean and lints under petab.v2."""
    work = _workspace_v2(tmp_path)
    staged = R.stage_workspace(work)
    assert set(staged["design_hashes"]) == {
        "measurements.tsv",
        "conditions.tsv",
        "experiments.tsv",
        "observables.tsv",
    }
    assert staged["shipped_rows"]["sigma_B"] == (0, 0.1)
    assert staged["shipped_rows"]["k1"] == (1, 0.5)
    work = staged["agent_dir"]
    out = R.check_delivery(work, staged)
    assert out["design_modified"] == [] and out["shipped_rows_violated"] == []
    assert out["petab_lint"]["ok"], out["petab_lint"]
    assert out["model_authored"] is True


def test_a_v2_delivery_that_rewrites_the_design_or_a_shipped_row_is_reported(tmp_path):
    work = _workspace_v2(tmp_path)
    staged = R.stage_workspace(work)
    work = staged["agent_dir"]
    experiments = work / "experiments.tsv"
    experiments.write_text(experiments.read_text() + "\n")
    parameters = (work / "parameters.tsv").read_text()
    header = parameters.splitlines()[0].split("\t")
    rows = []
    for line in parameters.splitlines()[1:]:
        cells = dict(zip(header, line.split("\t")))
        if cells["parameterId"] == "sigma_B":
            cells["nominalValue"] = "0.5"
        elif cells["parameterId"] == "A_init":
            cells["estimate"] = "true"
        rows.append("\t".join(cells[c] for c in header))
    (work / "parameters.tsv").write_text("\n".join(["\t".join(header), *rows]) + "\n")
    out = R.check_delivery(work, staged)
    assert out["design_modified"] == ["experiments.tsv"]
    assert sorted(out["shipped_rows_violated"]) == sorted(
        ["sigma_B: fixed value 0.1 -> 0.5", "A_init: estimate 0 -> 1"]
    )


def test_a_v2_delivery_is_charged_but_not_for_a_condition_target(tmp_path):
    """v2 names its condition targets in `targetId` rows: a species the condition table sets
    starts where the condition says, so its declared value is no number of the model's. Its
    `estimate` column reads `false`, which must still read as a fixed row."""
    work = _workspace_v2(tmp_path)
    staged = R.stage_workspace(work)
    work = staged["agent_dir"]

    def edit(m):
        import libsbml

        m.removeInitialAssignment("A")
        m.getSpecies("A").setInitialConcentration(3.7)
        _add(m, "parameter", "kx", 0.37)
        _add(m, "parameter", "kf", 0.2)
        m.getReaction("r2").getKineticLaw().setMath(libsbml.parseL3Formula("k2 * kx * kf * B"))

    _edit_model(work, edit)
    header = (work / "parameters.tsv").read_text().splitlines()[0].split("\t")
    row = {"parameterId": "kf", "parameterScale": "lin", "lowerBound": "0", "upperBound": "1"}
    row |= {"nominalValue": "0.2", "estimate": "false"}
    _add_rows(work, "\t".join(row.get(c, "") for c in header))
    out = R.check_delivery(work, staged)
    assert out["petab_lint"]["ok"], out["petab_lint"]
    assert out["uncharged"] == [
        "kf = 0.2: estimate false -> true",
        "kx = 0.37: no parameter-table row",
    ]


def test_a_v2_lint_error_is_named(tmp_path):
    work = _workspace_v2(tmp_path)
    staged = R.stage_workspace(work)
    work = staged["agent_dir"]
    lines = (work / "parameters.tsv").read_text().splitlines()
    (work / "parameters.tsv").write_text(
        "\n".join(line for line in lines if not line.startswith("sigma_B")) + "\n"
    )
    out = R.check_delivery(work, staged)
    assert out["petab_lint"]["ok"] is False
    assert any("sigma_B" in m for m in out["petab_lint"]["messages"]), out["petab_lint"]


def test_unscoreable_result_is_a_loss_not_an_abort(tmp_path):
    """`aborted:` is what the sweep aggregation drops as infrastructure noise; a delivery that
    cannot be judged is the agent's failure and must stay in the table as one."""

    class _S:
        name = "toy"

    meta = {"stop_reason": "completed", "driver": "barebones"}
    res = R._write_unscoreable_result(tmp_path, meta, _S(), "petab_invalid: bad sbml")
    assert res["success"] is False and res["delta_bic"] is None
    assert res["stop_reason"] == "completed"
    assert res["unscoreable_reason"].startswith("petab_invalid")
    assert (tmp_path / "result.json").is_file()


def test_the_judge_is_handed_the_reconciled_estimates(monkeypatch, tmp_path):
    """#1: the reconcile's estimates reach k only through this hand-off; dropped, the judge would
    price the null at its delivered rows again while every judge test still passed."""
    seen = {}

    async def call(name, **_):
        return {"model_id": "m", "model_authored": True}

    async def judge(*_, estimated=None):
        seen["estimated"] = estimated
        return 0, {}

    monkeypatch.setattr(R, "_wrap_call_timeout", lambda *_, **__: None)
    monkeypatch.setattr(R, "_wrap_tool_clock", lambda *_: SimpleNamespace(seconds=0.0))
    monkeypatch.setattr(R, "_make_call", lambda *_: call)
    monkeypatch.setattr(R, "_judge", judge)
    delivery = {"estimated": ["k1", "kx"], "uncharged": ["kx = 0.37: no parameter-table row"]}
    meta = {"delivery": delivery, "workspace_audit": {"shell_busy_seconds": 0.0}}
    args = SimpleNamespace(n_starts=1)
    scenario = SimpleNamespace(example_dir=tmp_path)
    asyncio.run(R._judge_delivered(scenario, None, tmp_path, args, meta, 0.0, 0.0))
    assert seen["estimated"] == ["k1", "kx"]


# --------------------------------------------------------------------------- timing
def test_the_scoring_reserve_comes_out_of_the_agents_wall_budget():
    """The agent got the whole `--max-wall-seconds` and the delivery was compiled and scored
    after it, in the sweep's margin below Slurm's `--time`: a slow compile was killed with no
    result.json and the run dropped, on this arm only (#8)."""
    budget = 4 * 3600 - 1200.0
    assert R._split_wall_budget(budget) == (budget - R._SCORE_RESERVE_S, R._SCORE_RESERVE_S)
    assert R._split_wall_budget(600.0) == (300.0, 300.0)  # a smoke-test budget keeps half
    assert R._split_wall_budget(None) == (None, R._SCORE_UNBUDGETED_S)  # nothing to finish inside


def test_the_null_arms_sandbox_and_scoring_time_is_compute_not_llm(tmp_path):
    """The judge's clock used to start once the server had compiled and scored the delivery, and
    never saw the agent's sandboxed fits, so `llm_seconds` (wall - compute) carried both and the
    null arm read as near-zero compute (#8). Here the run is 1000 s old, the agent's commands kept
    the sandbox busy for 500 s, and it stopped 300 s ago."""
    import argparse
    import asyncio
    import time
    from types import SimpleNamespace

    from gemot import scenario as S

    canned = {
        "describe_problem": {"current_version": 1, "model_authored": True},
        "get_progress": {"best_version": 1},
        "get_model_structure": {"reactions": [{"id": "r1"}], "species": ["A"]},
        "get_fit_summary": {"neg_log_likelihood": 100.0, "n_data_points": 50, "chi2": 47.0},
        "get_parameters": [{"estimate": 1}] * 3,
        "get_errors": [],
    }

    class _Session:
        async def call_tool(self, name, arguments=None, read_timeout_seconds=None, **_):
            return SimpleNamespace(structuredContent=canned.get(name, {}))

    async def _no_fix(call, n_starts):  # pragma: no cover - never driven here
        return None

    scenario = S.Scenario(
        name="pkg",
        problem_dir=tmp_path / "ex" / "problem",
        scripted_fix=_no_fix,
        diagnostics=None,
        reference_free=True,
        adequacy=S.ADEQUACY_CHI2,
    )
    meta = {
        "workspace_audit": {"shell_busy_seconds": 500.0},
        "delivery": {"estimated": None, "uncharged": []},
    }
    args = argparse.Namespace(n_starts=10)
    now = time.monotonic()
    judged = R._judge_delivered(scenario, _Session(), tmp_path, args, meta, now - 1000, now - 300)
    _, result = asyncio.run(judged)
    assert result["scoring_seconds"] == pytest.approx(300, abs=5)
    assert result["compute_seconds"] == pytest.approx(800, abs=5)
    assert result["llm_seconds"] == pytest.approx(200, abs=5)


# --------------------------------------------------------------------------- wiring
def _system_prompt(staged: dict) -> str:
    return R.barebones_system_prompt(staged, execute_timeout=1234.0, n_cpus=8, n_starts=100)


@pytest.mark.parametrize("workspace", [_workspace, _workspace_v2], ids=["v1", "v2"])
def test_prompt_template_formats_and_is_tracked(tmp_path, workspace):
    from gemot import prompts as P

    text = _system_prompt(R.stage_workspace(workspace(tmp_path)))
    assert "1234s" in text and "8 CPU" in text and "100 starts" in text
    assert "{" not in text and "}" not in text
    assert "barebones" in P.PROMPT_GROUPS
    assert P.BAREBONES_TASK in P.tracked_leaves()
    for slots in P.BAREBONES_PETAB_FORMATS.values():
        assert set(slots.values()) <= P.tracked_leaves()
    bio = P.BAREBONES_BIO_TOOLS_TMPL.format(n=2, names="get_interactions, omnipath_query")
    assert "2 tools" in bio and "omnipath_query" in bio and "{" not in bio
    # the null gets the contract, not the treatment: none of the lead's modelling rules
    for treatment in (
        "EXPONENTS ARE PARAMETERS",
        "ADEQUACY BEFORE PARSIMONY",
        "make_changes",
        "subagent_type",
    ):
        assert treatment not in text


def test_a_v2_problem_gets_v2_instructions(tmp_path):
    """The null's instructions said "PEtab v1" whatever it was handed: on a native v2 problem it
    was told of `estimate` = 1, of a design without `experiments.tsv`, and nothing of the estimation
    scale pyPESTO's v2 importer only takes as an argument. The version is read from the staged
    YAML, and the tables it may not modify are the ones the delivery check hashes."""
    v1 = R.stage_workspace(_workspace(tmp_path / "a"))
    v2 = R.stage_workspace(_workspace_v2(tmp_path / "b"))
    assert (v1["petab_version"], v2["petab_version"]) == (1, 2)
    text_v1, text_v2 = _system_prompt(v1), _system_prompt(v2)

    assert "PEtab v2 problem" in text_v2 and "PEtab v1" not in text_v2
    assert "`estimate` = 1" not in text_v2 and "`estimate` = `true`" in text_v2
    assert "parameter_scales" in text_v2 and "petab.v2.Problem.from_yaml" in text_v2
    assert "PEtab v1 problem" in text_v1 and "PEtab v2" not in text_v1
    assert "parameter_scales" not in text_v1
    for staged, text in ((v1, text_v1), (v2, text_v2)):
        named = set(re.findall(r"Do NOT modify (.*?): the experimental design", text)[0].split("`"))
        assert set(staged["design_hashes"]) == named - {"", ", ", " or "}


def test_an_unknown_petab_version_is_refused_before_the_agent_starts(tmp_path):
    staged = R.stage_workspace(_workspace(tmp_path))
    with pytest.raises(ValueError, match="PEtab v3"):
        _system_prompt({**staged, "petab_version": 3})


def test_the_lead_still_gets_the_readout_note_with_its_constants():
    from gemot import prompts as P
    from gemot.scenario import Scenario

    kw = dict(problem_dir=Path("/x/problem"), scripted_fix=None)
    with_note = Scenario(name="a", known_constants="KNOWN CONSTANTS: k.", **kw).agent_task
    assert with_note.endswith("KNOWN CONSTANTS: k." + P.KNOWN_CONSTANTS_READOUT_NOTE)
    assert "set_assignment" in with_note  # the lead's own tool vocabulary
    assert Scenario(name="b", **kw).agent_task == P.DEFAULT_TASK

    # the null's task never carries the lead's note
    class _S:
        known_constants = "KNOWN CONSTANTS: k."
        biocurator_entities = []

    assert P.KNOWN_CONSTANTS_READOUT_NOTE not in R.barebones_task(_S())


def test_note_check_refuses_a_tool_name_but_not_the_verb_rename():
    with pytest.raises(ValueError, match="describe_problem"):
        R.check_note_is_tool_free("Start with describe_problem, then build.")
    assert R.check_note_is_tool_free("do NOT re-estimate or rename them")


def test_task_carries_the_known_constants_and_the_curated_entities():
    """The null gets what the panel gets: the lead's known-constants note AND the entity names the
    biocurator is handed — information parity, so only the design differs."""

    class _S:
        known_constants = "KNOWN CONSTANTS: sigma ships estimated."
        biocurator_entities = ["IP3 receptor (ITPR)", "calcium ion"]

    task = R.barebones_task(_S())
    assert task.startswith(R.BAREBONES_TASK)
    assert "KNOWN CONSTANTS: sigma ships estimated." in task
    assert "IP3 receptor (ITPR), calcium ion" in task

    class _Bare:
        known_constants = ""
        biocurator_entities = []

    assert R.barebones_task(_Bare()) == R.BAREBONES_TASK


def test_bio_tool_discovery_is_the_panels_allowlist(monkeypatch):
    """Both arms hold the same lookups because both call `discover_bio_tools`: the curated
    allowlist resolves, a literature tool only under the ablation, a petab-name collision and a
    duplicate are dropped, and a server whose discovery raises costs only its own tools."""
    pytest.importorskip("langchain_mcp_adapters")
    import asyncio
    from langchain_core.tools import StructuredTool
    import langchain_mcp_adapters.tools as lmt
    from gemot.agent import biocurator as B

    def _tool(name):
        return StructuredTool.from_function(func=(lambda: ""), name=name, description=name)

    curated = B.DEFAULT_BIO_TOOLS[0]
    literature = B.LITERATURE_BIO_TOOLS[0]
    servers = {
        "a": [_tool(curated), _tool(literature), _tool("describe_problem"), _tool("unrelated")],
        "b": [_tool(curated)],  # duplicate of a's
        "dead": RuntimeError("tools/list failed"),
    }

    async def _fake_load(session):
        out = servers[session]
        if isinstance(out, Exception):
            raise out
        return out

    monkeypatch.setattr(lmt, "load_mcp_tools", _fake_load)
    got = asyncio.run(
        B.discover_bio_tools(
            ["a", "b", "dead"], literature=False, exclude_names={"describe_problem"}
        )
    )
    assert [t.name for t in got] == [curated]
    got = asyncio.run(B.discover_bio_tools(["a"], literature=True))
    assert sorted(t.name for t in got) == sorted([curated, literature])
    assert asyncio.run(B.discover_bio_tools([], literature=True)) == []
    assert asyncio.run(B.discover_bio_tools(["dead"], literature=True)) == []


def test_cli_accepts_the_barebones_driver_and_dispatches_to_its_pipeline(monkeypatch, tmp_path):
    from gemot.runner import cli
    from gemot.runner import barebones as rb

    class _S:
        name = "toy"
        problem_dir = tmp_path / "problem"
        version = 1

    args = cli.build_parser(_S()).parse_args(["--driver", "barebones"])
    assert args.driver == "barebones"

    seen = {}

    async def _fake_main(scenario, args):
        seen["driver"] = args.driver
        return 0, {"success": True}

    monkeypatch.setattr(rb, "_main_async_barebones", _fake_main)
    assert cli.run_scenario(_S(), ["--driver", "barebones", "--model", "luna"]) == 0
    assert seen["driver"] == "barebones"


def test_biology_blind_is_rejected_before_workspace_cleanup(monkeypatch):
    """Reject the unsupported arm before cleanup erases an existing run or anonymizes its files."""
    import asyncio
    from types import SimpleNamespace

    monkeypatch.setenv("PETAB_BIOLOGY_BLIND", "1")

    def _unexpected_cleanup(scenario):
        pytest.fail("unsupported ablation must be rejected before workspace cleanup")

    monkeypatch.setattr(R, "cleanup", _unexpected_cleanup)
    with pytest.raises(SystemExit, match=r"PETAB_BIOLOGY_BLIND=1.*--driver barebones"):
        asyncio.run(R._main_async_barebones(SimpleNamespace(), SimpleNamespace()))


def test_the_null_arm_waits_out_a_provider_outage_without_spending_a_resume(monkeypatch, tmp_path):
    """gemot#55 in the barebones loop, which copies the panel's watchdog: four chunk timeouts, one
    more than there are resumes, used to end the run as a stall; they are an outage now."""
    pytest.importorskip("deepagents")
    import deepagents
    from langchain_openai import StreamChunkTimeoutError

    from gemot.agent import barebones as B

    script = ["chunk"] * (B._RUN_RESUME_ATTEMPTS + 1)
    inputs = []

    class _Agent:
        def astream(self, stream_input, config=None, stream_mode=None):
            inputs.append(stream_input)
            chunk = bool(script) and script.pop(0) == "chunk"

            async def _gen():
                if chunk:
                    raise StreamChunkTimeoutError(120.0, model_name="z-ai/glm-5.3-flash")
                yield {"messages": [{"role": "assistant", "content": "done"}]}

            return _gen()

    async def _no_wait(_seconds):
        return None

    monkeypatch.setattr(deepagents, "create_deep_agent", lambda **kw: _Agent())
    monkeypatch.setattr(B.asyncio, "sleep", _no_wait)
    out = asyncio.run(
        B.run_barebones(
            tmp_path,
            model="fake/model",
            base_url="http://x",
            api_key="sk-test",
            system_prompt="sp",
            task="t",
            recursion_limit=10,
            execute_timeout_s=60.0,
        )
    )

    assert out["stop_reason"] == "completed"
    assert out["provider_outage_retries"] == B._RUN_RESUME_ATTEMPTS + 1
    assert len(inputs) == B._RUN_RESUME_ATTEMPTS + 2
