"""The observable-blinding contract: readout_* measurement mappings.

Three pieces, tested here at the unit level (tests/test_model_edits.py covers the
`set_assignment` op; the examples' own materialization is exercised by examples/setup.py):

* `benchmark._blind_observables` — the materialization-side rewrite and its leak guards;
* `validation._undefined_readout_issues` — the commit-time remedy;
* the run.py specs of the three blinded examples stay consistent with their TRUE_ANT.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

from petab_mcp.validation import _undefined_readout_issues


def _prob(formulas: dict[str, str], noise: dict[str, str] | None = None):
    """A minimal problem-shaped object carrying just an observable_df."""
    odf = pd.DataFrame({
        "observableFormula": pd.Series(formulas),
        "noiseFormula": pd.Series(noise or {k: f"noiseParameter1_{k}" for k in formulas}),
    })
    odf.index.name = "observableId"

    class _P:
        observable_df = odf

    return _P()


# ------------------------------------------------------------------ materialization-side rewrite
def test_blinding_rewrites_formula_and_writes_the_description():
    from petab_harness.benchmark import _blind_observables

    prob = _prob({"pAkt_tot": "observableParameter1_pAkt_tot * (pAkt + pAkt_S6)"})
    _blind_observables(prob, {"pAkt_tot": {
        "formula": "observableParameter1_pAkt_tot * readout_pAkt_tot",
        "measures": "immunoblot signal for phospho-Akt"}})
    row = prob.observable_df.loc["pAkt_tot"]
    assert row["observableFormula"] == "observableParameter1_pAkt_tot * readout_pAkt_tot"
    assert row["observableName"] == "immunoblot signal for phospho-Akt"
    # the model-content symbols are GONE
    assert "pAkt" not in row["observableFormula"].replace("readout_pAkt_tot", "").replace(
        "observableParameter1_pAkt_tot", "")


def test_blinding_refuses_an_unknown_observable():
    from petab_harness.benchmark import _blind_observables

    with pytest.raises(ValueError, match="no observable"):
        _blind_observables(_prob({"a": "x"}), {"b": {"formula": "readout_b", "measures": "m"}})


def test_blinding_refuses_a_formula_without_the_readout_symbol():
    """A blinded observable nothing tells the agent to define is a broken problem, silently."""
    from petab_harness.benchmark import _blind_observables

    with pytest.raises(ValueError, match="must reference 'readout_a'"):
        _blind_observables(_prob({"a": "x"}), {"a": {"formula": "0.5 * x", "measures": "m"}})


def test_blinding_refuses_a_smuggled_symbol():
    """Only symbols already in the published formula may be KEPT (scalings/offsets); anything
    new is a hint arriving through the back door."""
    from petab_harness.benchmark import _blind_observables

    with pytest.raises(ValueError, match="scaling/offset"):
        _blind_observables(_prob({"a": "scale_a * x"}),
                           {"a": {"formula": "scale_a * readout_a + hill_n", "measures": "m"}})


def test_blinding_refuses_a_noise_side_leak():
    """A noiseFormula referencing a blinded-away species would move the leak into sigma."""
    from petab_harness.benchmark import _blind_observables

    prob = _prob({"a": "x"}, noise={"a": "0.1 * x"})
    with pytest.raises(ValueError, match="leak would move into sigma"):
        _blind_observables(prob, {"a": {"formula": "readout_a", "measures": "m"}})


def test_blinding_keeps_scalings_and_the_noise_model():
    from petab_harness.benchmark import _blind_observables

    prob = _prob({"a": "scale_a * x + offset_a"})
    _blind_observables(prob, {"a": {"formula": "scale_a * readout_a + offset_a",
                                    "measures": "m"}})
    row = prob.observable_df.loc["a"]
    assert "scale_a" in row["observableFormula"] and "offset_a" in row["observableFormula"]
    assert row["noiseFormula"] == "noiseParameter1_a"  # untouched


# ------------------------------------------------------------------ commit-time remedy
def _sbml_with(ids: list[str], with_reaction: bool = True, rules: list[str] | None = None,
               rate_rules: list[str] | None = None):
    """Returns (document, model). The DOCUMENT must be kept alive by the caller: libsbml's
    SWIG proxies do not own their C++ objects, so dropping the document frees the model out
    from under the proxy and the next `getElementBySId` segfaults the interpreter."""
    import libsbml

    doc = libsbml.SBMLDocument(3, 1)
    m = doc.createModel()
    for sid in ids:
        p = m.createParameter(); p.setId(sid); p.setConstant(True); p.setValue(1.0)
    for sid in (rules or []):
        r = m.createAssignmentRule(); r.setVariable(sid); r.setFormula("1")
    for sid in (rate_rules or []):
        r = m.createRateRule(); r.setVariable(sid); r.setFormula("1")
    if with_reaction:
        # `_undefined_readout_issues` self-skips a model with no dynamics (the v0 seed), so a
        # test of the CHECK needs one reaction to get past that.
        rx = m.createReaction(); rx.setId("r1"); rx.setReversible(False); rx.setFast(False)
    return doc, m


def _lintable(formulas: dict[str, str], model_ids: list[str], names: dict | None = None,
              with_reaction: bool = True, rules: list[str] | None = None,
              rate_rules: list[str] | None = None):
    prob = _prob(formulas)
    prob.observable_df["observableName"] = pd.Series(names or {})
    doc, sm = _sbml_with(model_ids, with_reaction=with_reaction, rules=rules,
                         rate_rules=rate_rules)

    class _M:
        _doc = doc            # lifetime anchor — see _sbml_with
        sbml_model = sm

    prob.model = _M()
    prob.parameter_df = pd.DataFrame(index=pd.Index([], name="parameterId"))
    return prob


def test_an_undefined_readout_is_an_error_carrying_remedy_and_biology():
    prob = _lintable({"a": "scale_a * readout_a"}, model_ids=["scale_a"],
                     names={"a": "immunoblot signal for X"})
    issues = _undefined_readout_issues(prob)
    assert len(issues) == 1 and issues[0]["level"] == "error"
    msg = issues[0]["message"]
    assert "readout_a" in msg and "set_assignment" in msg
    assert "It measures: immunoblot signal for X." in msg


def test_a_defined_readout_passes():
    """Defined means an ASSIGNMENT RULE — that is what a measurement mapping is."""
    prob = _lintable({"a": "scale_a * readout_a"}, model_ids=["scale_a"], rules=["readout_a"])
    assert _undefined_readout_issues(prob) == []


def test_non_readout_symbols_are_not_this_checks_business():
    """A missing ordinary species is petab lint's job — one generic error, not two."""
    prob = _lintable({"a": "some_species"}, model_ids=[])
    assert _undefined_readout_issues(prob) == []


# ------------------------------------------------------------------ the three examples' specs
@pytest.fixture(scope="module")
def _examples_on_path():
    """`examples/` on sys.path for the duration of this module, then removed.

    Inserted per parametrised case with no teardown before, which left three copies behind for
    the rest of the session.
    """
    d = str(Path(__file__).resolve().parent.parent / "examples")
    sys.path.insert(0, d)
    try:
        yield
    finally:
        while d in sys.path:
            sys.path.remove(d)


# Every benchmark example is blinded now; a new one must join this list (and ship a spec).
BLINDED = ["alkan", "armistead", "bachmann", "blasi", "boehm", "brannmark", "bruno", "crauste",
           "fiedler", "fujita", "isensee", "lucarelli", "oliveira", "perelson", "rahman",
           "raimundez", "schwen", "sneyd"]


@pytest.mark.parametrize("example", BLINDED)
def test_example_spec_is_internally_consistent(example, _examples_on_path):
    """Every blinded observable's readout symbol is referenced by its blinded formula AND
    defined by TRUE_ANT — the scripted/reference build must satisfy the contract it ships.

    Imported with a plain `import`, NOT `importorskip`: this is the invariant that keeps the
    scripted/reference build able to satisfy the contract it ships, so an example that fails to
    import must FAIL here rather than turning into a green skip."""
    import importlib

    mod = importlib.import_module(f"{example}.run")
    blinding = mod.OBSERVABLE_BLINDING
    assert blinding, f"{example}: no blinding spec"
    assert mod.SCENARIO.version >= 2, (example, "blinding changed the problem; bump the version")
    import re as _re
    for obs_id, spec in blinding.items():
        # Flattened tables share one readout across gel/dataset rows via the "readout" key.
        readout = spec.get("readout", f"readout_{obs_id}")
        assert readout in spec["formula"], (example, obs_id)
        # EVERY readout the formula names, not just the declared one: schwen's ELISA references
        # readout_insulin AND readout_insulin_fragments, so checking only the key would let a
        # typo in the second ship a problem its own reference build cannot satisfy.
        for sym in _re.findall(r"readout_\w+", spec["formula"]):
            assert f"{sym} :=" in mod.TRUE_ANT, (example, obs_id, sym, "TRUE_ANT lacks it")
        assert spec["measures"].strip(), (example, obs_id, "no description")


def test_a_readout_declared_as_a_SPECIES_is_still_undefined():
    """The regression that made this whole guard a no-op. The seed declares one species per
    observable-formula identifier so v0 lints — which includes every `readout_*` — so "the id
    resolves to an SBML element" is true by construction. A model leaving each readout a free
    constant would then commit: on blasi that is 16 constants against 16 steady-state observable
    means, fitting them exactly while skipping the measurement-mapping task entirely."""
    prob = _lintable({"a": "readout_a"}, model_ids=["readout_a"], with_reaction=True)
    issues = _undefined_readout_issues(prob)
    assert len(issues) == 1, "a readout declared as a species must NOT count as defined"
    assert "ASSIGNMENT RULE" in issues[0]["message"]


def test_the_dynamics_free_seed_is_not_flagged():
    """v0 has no reactions and nothing to map a readout onto — the same self-skip
    `_unused_condition_target_issues` uses. Flagging it would make the seed itself invalid."""
    prob = _lintable({"a": "readout_a"}, model_ids=["readout_a"], with_reaction=False)
    assert _undefined_readout_issues(prob) == []


def test_a_missing_description_does_not_render_as_nan():
    """`or ""` does not catch a NaN cell — float('nan') is truthy — so the agent was told
    "It measures: nan." """
    import math

    prob = _lintable({"a": "readout_a"}, model_ids=[], with_reaction=True,
                     names={"a": math.nan})
    msg = _undefined_readout_issues(prob)[0]["message"]
    assert "nan" not in msg.lower(), msg


def test_the_seed_declares_a_readout_as_a_parameter_not_a_species():
    """The trap the first blinded sweep measured: the seed declared `readout_*` as SPECIES, agents
    copied that into their own models, wired the readout into reactions as a state, and then could
    not assign it — 4 of the sweep's 27 rejections were exactly this over-determination refusal,
    all on fujita (the example with non-identity mappings). A constant parameter lints and
    simulates identically at v0 and suggests nothing about being a state. The seed must also stay
    DYNAMICS-FREE, so a `:=` rule (which would flip `has_dynamics` and with it `model_authored`)
    is not an option."""
    from petab_mcp.reconcile import seed_model_antimony

    odf = pd.DataFrame({"observableFormula": pd.Series(
        {"a": "scale_a * readout_a", "b": "some_species"})})
    odf.index.name = "observableId"

    class _P:
        observable_df = odf
        condition_df = None
        parameter_df = pd.DataFrame(index=pd.Index(["scale_a"], name="parameterId"))

    text = seed_model_antimony(_P())
    assert "species some_species;" in text
    assert "readout_a = 0.1;" in text
    assert "species readout_a" not in text, "a seeded readout species is the over-determination trap"
    assert ":=" not in text, "a seeded rule would make the dynamics-free seed count as authored"


def test_a_pure_readout_table_demotes_every_readout():
    """No readout is ever a species, even when the formulas name nothing else.

    Two earlier attempts tied this to the AMICI zero-species segfault: keeping ALL readouts as
    species (which left boehm, blasi, crauste, perelson and rahman with the trap the demotion
    exists to remove), then keeping exactly one. `reconcile.ensure_one_species` carries that
    invariant with a placeholder instead, so the seed is free to say what is true — a measurement
    mapping is not a state."""
    from petab_mcp.reconcile import seed_model_antimony

    odf = pd.DataFrame({"observableFormula": pd.Series({"a": "readout_a", "b": "readout_b"})})
    odf.index.name = "observableId"

    class _P:
        observable_df = odf
        condition_df = None
        parameter_df = pd.DataFrame(index=pd.Index([], name="parameterId"))

    text = seed_model_antimony(_P())
    assert "species readout_" not in text, text
    assert "readout_a = 0.1;" in text and "readout_b = 0.1;" in text


def test_a_rate_rule_model_with_no_reactions_is_still_checked():
    """The self-skip must be `has_dynamics`, not a reaction count.

    ODE-style Antimony authoring produces a model that is fully dynamic with ZERO reactions, so
    a bare `getNumReactions() == 0` skipped exactly those models -- letting one commit with every
    readout left as a free constant, which is the loophole this check exists to close.
    """
    prob = _lintable({"a": "readout_a"}, model_ids=["readout_a", "x"],
                     with_reaction=False, rate_rules=["x"])
    issues = _undefined_readout_issues(prob)
    assert len(issues) == 1 and "readout_a" in str(issues[0])


def test_a_rate_rule_model_that_defines_its_readout_passes():
    prob = _lintable({"a": "readout_a"}, model_ids=["readout_a", "x"],
                     with_reaction=False, rate_rules=["x"], rules=["readout_a"])
    assert _undefined_readout_issues(prob) == []


# ------------------------------------------------------------------ collapsing through a merge
def test_collapsing_a_spec_rewrites_placeholders_only():
    """Not the prose, not the readout symbol, and not a LONGER id that merely shares a prefix."""
    from petab_harness.benchmark import _collapse_blinding

    out = _collapse_blinding(
        {"pAkt": {"formula": "observableParameter1_pAkt * readout_pAkt",
                  "measures": "phospho-Akt, quantified from the pAkt_tot lane",
                  "readout": "readout_pAkt"},
         "pAkt_tot": {"formula": "observableParameter1_pAkt_tot * readout_pAkt_tot",
                      "measures": "total Akt"}},
        {"pAkt": "pAkt_obs"})
    assert out["pAkt_obs"]["formula"] == "observableParameter1_pAkt_obs * readout_pAkt"
    assert out["pAkt_obs"]["measures"] == "phospho-Akt, quantified from the pAkt_tot lane"
    assert out["pAkt_obs"]["readout"] == "readout_pAkt"
    # the longer id is a different observable and must come through untouched
    assert out["pAkt_tot"]["formula"] == "observableParameter1_pAkt_tot * readout_pAkt_tot"


def test_collapsing_rewrites_the_noise_formula_too():
    """Both halves move: the placeholder, and -- because this spec has no explicit `readout` --
    the DERIVED readout symbol, which `_blind_observables` re-derives from the merged id."""
    from petab_harness.benchmark import _collapse_blinding

    out = _collapse_blinding(
        {"x": {"formula": "readout_x", "noise": "noiseParameter1_x * readout_x",
               "measures": "m"}},
        {"x": "x_obs"})
    assert out["x_obs"]["noise"] == "noiseParameter1_x_obs * readout_x_obs"
    assert out["x_obs"]["formula"] == "readout_x_obs"


def test_specs_that_disagree_after_collapsing_are_refused():
    from petab_harness.benchmark import _collapse_blinding

    with pytest.raises(ValueError, match="disagree"):
        _collapse_blinding(
            {"a": {"formula": "readout_a", "measures": "one"},
             "b": {"formula": "readout_a", "measures": "another"}},
            {"a": "shared", "b": "shared"})


# ------------------------------------------------------------------ the noise branch's guards
def test_a_noise_spec_may_reference_a_readout_other_than_its_own():
    """A proportional sigma rides the same readout the formula does; subtracting only the
    DERIVED `readout_<obsId>` refused that legal shape."""
    from petab_harness.benchmark import _blind_observables

    prob = _prob({"Cer_obs": "observableParameter1_Cer_obs * Cer"},
                 noise={"Cer_obs": "noiseParameter1_Cer_obs * Cer"})
    prob.parameter_df = pd.DataFrame(index=pd.Index(["observableParameter1_Cer_obs",
                                                     "noiseParameter1_Cer_obs"],
                                                    name="parameterId"))
    _blind_observables(prob, {"Cer_obs": {
        "formula": "observableParameter1_Cer_obs * readout_ceramide",
        "noise": "noiseParameter1_Cer_obs * readout_ceramide",
        "readout": "readout_ceramide", "measures": "total ceramide"}})
    assert prob.observable_df.loc["Cer_obs", "noiseFormula"] == \
        "noiseParameter1_Cer_obs * readout_ceramide"


def test_a_model_entity_left_in_the_noise_formula_is_refused():
    """The species must appear ONLY in the noiseFormula.

    Using one that the observableFormula also names proves nothing: it is already in
    `seed_entities`, so the check fires with or without the noise-side widening. A sigma-only
    species is what that widening exists for."""
    from petab_harness.benchmark import _blind_observables

    prob = _prob({"a": "observableParameter1_a * S1"},
                 noise={"a": "noiseParameter1_a * S2"})
    prob.parameter_df = pd.DataFrame(index=pd.Index(["observableParameter1_a",
                                                     "noiseParameter1_a"], name="parameterId"))
    with pytest.raises(ValueError, match=r"reference-model entities.*\['S2'\]|\['S2'\]"):
        _blind_observables(prob, {"a": {
            "formula": "observableParameter1_a * readout_a",
            "noise": "noiseParameter1_a * S2", "measures": "m"}})


def test_a_sigma_only_species_is_refused_even_without_a_noise_key():
    """The default path checked the sigma column against the observable side's REMOVALS only, so
    a species the observable formula never mentioned was not in `removed` and sailed through --
    shipping the reference-model name to the agent through list_observables' noiseFormula."""
    from petab_harness.benchmark import _blind_observables

    prob = _prob({"a": "observableParameter1_a * S1"}, noise={"a": "0.1 * S2"})
    prob.parameter_df = pd.DataFrame(index=pd.Index(["observableParameter1_a"],
                                                    name="parameterId"))
    with pytest.raises(ValueError, match="S2"):
        _blind_observables(prob, {"a": {"formula": "observableParameter1_a * readout_a",
                                        "measures": "m"}})


def test_a_noise_formula_may_reference_the_observable_itself():
    """`noiseParameter1_a * a` is PEtab's relative-error idiom -- petab substitutes the simulated
    observable for `a`. It names no model entity, so refusing it would reject a formula that
    leaks nothing."""
    from petab_harness.benchmark import _blind_observables

    prob = _prob({"a": "observableParameter1_a * S1"}, noise={"a": "noiseParameter1_a * a"})
    prob.parameter_df = pd.DataFrame(index=pd.Index(["observableParameter1_a",
                                                     "noiseParameter1_a"], name="parameterId"))
    _blind_observables(prob, {"a": {"formula": "observableParameter1_a * readout_a",
                                    "noise": "noiseParameter1_a * a", "measures": "m"}})
    assert prob.observable_df.loc["a", "noiseFormula"] == "noiseParameter1_a * a"


def test_a_derived_readout_follows_its_observable_through_a_merge():
    """A spec with no `readout` key means `readout_<observableId>`, and `_blind_observables`
    re-derives it from the MERGED id -- so the formula's symbol has to move with it or the
    "must reference readout_<mergedId>" guard fires. Eleven blinded examples omit the key."""
    from petab_harness.benchmark import _blind_observables, _collapse_blinding

    spec = _collapse_blinding({"observable_k5": {"formula": "readout_observable_k5",
                                                 "measures": "H4K5ac"}},
                              {"observable_k5": "observable_k5_obs"})
    assert spec["observable_k5_obs"]["formula"] == "readout_observable_k5_obs"
    # and it survives the guard it used to trip
    prob = _prob({"observable_k5_obs": "x_k5"})
    prob.parameter_df = pd.DataFrame(index=pd.Index([], name="parameterId"))
    _blind_observables(prob, spec)


def test_a_merge_does_not_inherit_one_members_dataset_label():
    """The kept row is an arbitrary group member, and raimundez's published observableName cells
    ARE the per-gel observable ids -- so inheriting one names the merged observable after a
    single dataset."""
    import pandas as _pd
    from petab_harness.benchmark import _merge_equivalent_observables

    odf = _pd.DataFrame({
        "observableName": ["EGFR_ID1_A", "EGFR_ID2_B"],
        "observableFormula": ["observableParameter1_EGFR_ID1_A*EGFR",
                              "observableParameter1_EGFR_ID2_B*EGFR"],
        "noiseFormula": ["noiseParameter1_EGFR_ID1_A", "noiseParameter1_EGFR_ID2_B"],
    }, index=_pd.Index(["EGFR_ID1_A", "EGFR_ID2_B"], name="observableId"))
    mdf = _pd.DataFrame({
        "observableId": ["EGFR_ID1_A", "EGFR_ID2_B"],
        # DIFFERENT conditions: the merge trials itself against `needs_flattening` and declines
        # when the members share one, since overrides differing within a condition are exactly
        # what separate observables are for.
        "simulationConditionId": ["c1", "c2"], "time": [0.0, 0.0], "measurement": [1.0, 2.0],
        "observableParameters": ["s1", "s2"], "noiseParameters": ["n1", "n2"]})

    class _P:
        observable_df = odf
        measurement_df = mdf

    prob = _P()
    mapping = _merge_equivalent_observables(prob)
    assert mapping, "these two observables differ only in their per-row overrides"
    merged_id = next(iter(set(mapping.values())))
    assert prob.observable_df.loc[merged_id, "observableName"] == merged_id


def test_a_noise_key_without_a_noise_column_is_a_clear_error():
    """The refusal must name the problem, not surface as a bare KeyError from inside pandas."""
    from petab_harness.benchmark import _blind_observables

    prob = _prob({"x": "observableParameter1_x * S"})
    del prob.observable_df["noiseFormula"]
    prob.parameter_df = pd.DataFrame(index=pd.Index(["observableParameter1_x"],
                                                    name="parameterId"))
    with pytest.raises(ValueError, match="no noiseFormula column"):
        _blind_observables(prob, {"x": {"formula": "observableParameter1_x * readout_x",
                                        "noise": "1", "measures": "m"}})
