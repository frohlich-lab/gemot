"""The observable-blinding contract: readout_* measurement mappings.

Tested here at the unit level: `benchmark._blind_observables`, the materialization-side rewrite
and its leak guards. The commit-time remedy, `validation._undefined_readout_issues`, is pinned in
PEtab-MCP's tests/test_undefined_readouts.py; the examples' own materialization is exercised by
examples/setup.py.
"""

from __future__ import annotations

import pandas as pd
import pytest


def _prob(formulas: dict[str, str], noise: dict[str, str] | None = None):
    """A minimal problem-shaped object carrying just an observable_df."""
    odf = pd.DataFrame(
        {
            "observableFormula": pd.Series(formulas),
            "noiseFormula": pd.Series(noise or {k: f"noiseParameter1_{k}" for k in formulas}),
        }
    )
    odf.index.name = "observableId"

    class _P:
        observable_df = odf

    return _P()


# ------------------------------------------------------------------ materialization-side rewrite
def test_blinding_rewrites_formula_and_writes_the_description():
    from gemot.benchmark import _blind_observables

    prob = _prob({"pAkt_tot": "observableParameter1_pAkt_tot * (pAkt + pAkt_S6)"})
    _blind_observables(
        prob,
        {
            "pAkt_tot": {
                "formula": "observableParameter1_pAkt_tot * readout_pAkt_tot",
                "measures": "immunoblot signal for phospho-Akt",
            }
        },
    )
    row = prob.observable_df.loc["pAkt_tot"]
    assert row["observableFormula"] == "observableParameter1_pAkt_tot * readout_pAkt_tot"
    assert row["observableName"] == "immunoblot signal for phospho-Akt"
    # the model-content symbols are GONE
    assert "pAkt" not in row["observableFormula"].replace("readout_pAkt_tot", "").replace(
        "observableParameter1_pAkt_tot", ""
    )


def test_blinding_refuses_an_unknown_observable():
    from gemot.benchmark import _blind_observables

    with pytest.raises(ValueError, match="no observable"):
        _blind_observables(_prob({"a": "x"}), {"b": {"formula": "readout_b", "measures": "m"}})


def test_blinding_refuses_a_formula_without_the_readout_symbol():
    """A blinded observable nothing tells the agent to define is a broken problem, silently."""
    from gemot.benchmark import _blind_observables

    with pytest.raises(ValueError, match="must reference 'readout_a'"):
        _blind_observables(_prob({"a": "x"}), {"a": {"formula": "0.5 * x", "measures": "m"}})


def test_blinding_refuses_a_smuggled_symbol():
    """Only symbols already in the published formula may be KEPT (scalings/offsets); anything
    new is a hint arriving through the back door."""
    from gemot.benchmark import _blind_observables

    with pytest.raises(ValueError, match="scaling/offset"):
        _blind_observables(
            _prob({"a": "scale_a * x"}),
            {"a": {"formula": "scale_a * readout_a + hill_n", "measures": "m"}},
        )


def test_blinding_refuses_a_noise_side_leak():
    """A noiseFormula referencing a blinded-away species would move the leak into sigma."""
    from gemot.benchmark import _blind_observables

    prob = _prob({"a": "x"}, noise={"a": "0.1 * x"})
    with pytest.raises(ValueError, match="leak would move into sigma"):
        _blind_observables(prob, {"a": {"formula": "readout_a", "measures": "m"}})


def test_blinding_keeps_scalings_and_the_noise_model():
    from gemot.benchmark import _blind_observables

    prob = _prob({"a": "scale_a * x + offset_a"})
    _blind_observables(prob, {"a": {"formula": "scale_a * readout_a + offset_a", "measures": "m"}})
    row = prob.observable_df.loc["a"]
    assert "scale_a" in row["observableFormula"] and "offset_a" in row["observableFormula"]
    assert row["noiseFormula"] == "noiseParameter1_a"  # untouched


# ------------------------------------------------------------------ collapsing through a merge
def test_collapsing_a_spec_rewrites_placeholders_only():
    """Not the prose, not the readout symbol, and not a LONGER id that merely shares a prefix."""
    from gemot.benchmark import _collapse_blinding

    out = _collapse_blinding(
        {
            "pAkt": {
                "formula": "observableParameter1_pAkt * readout_pAkt",
                "measures": "phospho-Akt, quantified from the pAkt_tot lane",
                "readout": "readout_pAkt",
            },
            "pAkt_tot": {
                "formula": "observableParameter1_pAkt_tot * readout_pAkt_tot",
                "measures": "total Akt",
            },
        },
        {"pAkt": "pAkt_obs"},
    )
    assert out["pAkt_obs"]["formula"] == "observableParameter1_pAkt_obs * readout_pAkt"
    assert out["pAkt_obs"]["measures"] == "phospho-Akt, quantified from the pAkt_tot lane"
    assert out["pAkt_obs"]["readout"] == "readout_pAkt"
    # the longer id is a different observable and must come through untouched
    assert out["pAkt_tot"]["formula"] == "observableParameter1_pAkt_tot * readout_pAkt_tot"


def test_collapsing_rewrites_the_noise_formula_too():
    """Both halves move: the placeholder, and -- because this spec has no explicit `readout` --
    the DERIVED readout symbol, which `_blind_observables` re-derives from the merged id."""
    from gemot.benchmark import _collapse_blinding

    out = _collapse_blinding(
        {"x": {"formula": "readout_x", "noise": "noiseParameter1_x * readout_x", "measures": "m"}},
        {"x": "x_obs"},
    )
    assert out["x_obs"]["noise"] == "noiseParameter1_x_obs * readout_x_obs"
    assert out["x_obs"]["formula"] == "readout_x_obs"


def test_specs_that_disagree_after_collapsing_are_refused():
    from gemot.benchmark import _collapse_blinding

    with pytest.raises(ValueError, match="disagree"):
        _collapse_blinding(
            {
                "a": {"formula": "readout_a", "measures": "one"},
                "b": {"formula": "readout_a", "measures": "another"},
            },
            {"a": "shared", "b": "shared"},
        )


# ------------------------------------------------------------------ the noise branch's guards
def test_a_noise_spec_may_reference_a_readout_other_than_its_own():
    """A proportional sigma rides the same readout the formula does; subtracting only the
    DERIVED `readout_<obsId>` refused that legal shape."""
    from gemot.benchmark import _blind_observables

    prob = _prob(
        {"Cer_obs": "observableParameter1_Cer_obs * Cer"},
        noise={"Cer_obs": "noiseParameter1_Cer_obs * Cer"},
    )
    prob.parameter_df = pd.DataFrame(
        index=pd.Index(
            ["observableParameter1_Cer_obs", "noiseParameter1_Cer_obs"], name="parameterId"
        )
    )
    _blind_observables(
        prob,
        {
            "Cer_obs": {
                "formula": "observableParameter1_Cer_obs * readout_ceramide",
                "noise": "noiseParameter1_Cer_obs * readout_ceramide",
                "readout": "readout_ceramide",
                "measures": "total ceramide",
            }
        },
    )
    assert (
        prob.observable_df.loc["Cer_obs", "noiseFormula"]
        == "noiseParameter1_Cer_obs * readout_ceramide"
    )


def test_a_model_entity_left_in_the_noise_formula_is_refused():
    """The species must appear ONLY in the noiseFormula.

    Using one that the observableFormula also names proves nothing: it is already in
    `seed_entities`, so the check fires with or without the noise-side widening. A sigma-only
    species is what that widening exists for."""
    from gemot.benchmark import _blind_observables

    prob = _prob({"a": "observableParameter1_a * S1"}, noise={"a": "noiseParameter1_a * S2"})
    prob.parameter_df = pd.DataFrame(
        index=pd.Index(["observableParameter1_a", "noiseParameter1_a"], name="parameterId")
    )
    with pytest.raises(ValueError, match=r"reference-model entities.*\['S2'\]|\['S2'\]"):
        _blind_observables(
            prob,
            {
                "a": {
                    "formula": "observableParameter1_a * readout_a",
                    "noise": "noiseParameter1_a * S2",
                    "measures": "m",
                }
            },
        )


def test_a_sigma_only_species_is_refused_even_without_a_noise_key():
    """The default path checked the sigma column against the observable side's REMOVALS only, so
    a species the observable formula never mentioned was not in `removed` and sailed through --
    shipping the reference-model name to the agent through list_observables' noiseFormula."""
    from gemot.benchmark import _blind_observables

    prob = _prob({"a": "observableParameter1_a * S1"}, noise={"a": "0.1 * S2"})
    prob.parameter_df = pd.DataFrame(index=pd.Index(["observableParameter1_a"], name="parameterId"))
    with pytest.raises(ValueError, match="S2"):
        _blind_observables(
            prob, {"a": {"formula": "observableParameter1_a * readout_a", "measures": "m"}}
        )


def test_a_noise_formula_may_reference_the_observable_itself():
    """`noiseParameter1_a * a` is PEtab's relative-error idiom -- petab substitutes the simulated
    observable for `a`. It names no model entity, so refusing it would reject a formula that
    leaks nothing."""
    from gemot.benchmark import _blind_observables

    prob = _prob({"a": "observableParameter1_a * S1"}, noise={"a": "noiseParameter1_a * a"})
    prob.parameter_df = pd.DataFrame(
        index=pd.Index(["observableParameter1_a", "noiseParameter1_a"], name="parameterId")
    )
    _blind_observables(
        prob,
        {
            "a": {
                "formula": "observableParameter1_a * readout_a",
                "noise": "noiseParameter1_a * a",
                "measures": "m",
            }
        },
    )
    assert prob.observable_df.loc["a", "noiseFormula"] == "noiseParameter1_a * a"


def test_a_derived_readout_follows_its_observable_through_a_merge():
    """A spec with no `readout` key means `readout_<observableId>`, and `_blind_observables`
    re-derives it from the MERGED id -- so the formula's symbol has to move with it or the
    "must reference readout_<mergedId>" guard fires. Eleven blinded examples omit the key."""
    from gemot.benchmark import _blind_observables, _collapse_blinding

    spec = _collapse_blinding(
        {"observable_k5": {"formula": "readout_observable_k5", "measures": "H4K5ac"}},
        {"observable_k5": "observable_k5_obs"},
    )
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
    from gemot.benchmark import _merge_equivalent_observables

    odf = _pd.DataFrame(
        {
            "observableName": ["EGFR_ID1_A", "EGFR_ID2_B"],
            "observableFormula": [
                "observableParameter1_EGFR_ID1_A*EGFR",
                "observableParameter1_EGFR_ID2_B*EGFR",
            ],
            "noiseFormula": ["noiseParameter1_EGFR_ID1_A", "noiseParameter1_EGFR_ID2_B"],
        },
        index=_pd.Index(["EGFR_ID1_A", "EGFR_ID2_B"], name="observableId"),
    )
    mdf = _pd.DataFrame(
        {
            "observableId": ["EGFR_ID1_A", "EGFR_ID2_B"],
            # DIFFERENT conditions: the merge trials itself against `needs_flattening` and declines
            # when the members share one, since overrides differing within a condition are exactly
            # what separate observables are for.
            "simulationConditionId": ["c1", "c2"],
            "time": [0.0, 0.0],
            "measurement": [1.0, 2.0],
            "observableParameters": ["s1", "s2"],
            "noiseParameters": ["n1", "n2"],
        }
    )

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
    from gemot.benchmark import _blind_observables

    prob = _prob({"x": "observableParameter1_x * S"})
    del prob.observable_df["noiseFormula"]
    prob.parameter_df = pd.DataFrame(index=pd.Index(["observableParameter1_x"], name="parameterId"))
    with pytest.raises(ValueError, match="no noiseFormula column"):
        _blind_observables(
            prob,
            {"x": {"formula": "observableParameter1_x * readout_x", "noise": "1", "measures": "m"}},
        )
