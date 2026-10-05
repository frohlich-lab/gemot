"""Collapsing a blinded observable spec through an observable merge
(``benchmark._collapse_blinding``).

The rest of the observable-blinding contract reads the server's formula parser and readout
check, so it lives with the server tier, in ``tests/server/test_observable_blinding.py``.
"""

from __future__ import annotations

import pytest


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
