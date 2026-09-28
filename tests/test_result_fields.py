"""Test the residual-structure block the judge writes into result.json
(petab_harness.runner.scoring `_residual_structure_fields`).

Pure projection of TWO payloads — the fit's own rolled-up ``residual_structure`` verdict (which only
``get_fit_summary`` reports) and the ``get_fit_diagnosis`` per-observable rows. ``runner.scoring``
shares its thresholds with the server's own adequacy gate, so it needs `petab_mcp` — but not `mcp`;
skip where that is absent."""

from __future__ import annotations

from petab_harness.runner.scoring import (  # noqa: E402
    _NOISE_SIGMA_COLLAPSE_FRACTION,
    _noise_sigma_flag,
    _residual_structure_fields,
)


def test_extracts_verdict_and_per_observable():
    """The verdict comes from the FIT (`rollup`), the rows from the diagnosis — and the block reads
    the same as when both arrived in one payload."""
    diag = {
        "per_observable": [
            {
                "observable_id": "obs_A",
                "mean_weighted_residual": 2.0,
                "structure_low_power": False,
                "sign_test": {"p": 0.002},
                "runs_test": {"p": 0.3},
            },
            {
                "observable_id": "obs_B",
                "mean_weighted_residual": 0.1,
                "structure_low_power": True,
                "sign_test": {"p": 0.8},
                "runs_test": {"p": None},
            },
        ],
    }
    out = _residual_structure_fields(diag, {"flagged_observables": ["obs_A"], "low_power": False})
    assert out["flagged_observables"] == ["obs_A"]
    assert out["low_power"] is False
    rows = {r["observable_id"]: r for r in out["per_observable"]}
    assert rows["obs_A"]["sign_test_p"] == 0.002 and rows["obs_A"]["runs_test_p"] == 0.3
    assert rows["obs_B"]["low_power"] is True and rows["obs_B"]["runs_test_p"] is None


def test_none_when_no_diagnosis():
    # model-less / no-simulation run: get_fit_diagnosis returns a status-only payload and
    # get_fit_summary has no verdict to roll up
    assert _residual_structure_fields(None) is None
    assert _residual_structure_fields({}) is None
    # the diagnosis alone no longer carries the verdict, so a rollup-only call is a valid one
    assert _residual_structure_fields(
        None, {"flagged_observables": ["obs_A"], "low_power": False}
    ) == {
        "rollup_status": "read",
        "rows_status": "unavailable",
        "flagged_observables": ["obs_A"],
        "low_power": False,
        "per_observable": [],
    }


def test_a_missing_verdict_is_unknown_never_clean_and_powered():
    """REGRESSION: the verdict and the rows come from two calls now, and a rollup that could not be
    read defaulted to `flagged_observables: []` + `low_power: False` — result.json then asserted "no
    observable flagged, and the tests were powered" for a run whose verdict was never read. A sweep
    grouped on those columns cannot tell that from a genuine clean pass, so the keys are OMITTED and
    a status says which half was read."""
    rows_only = _residual_structure_fields(
        {"status": "no simulation yet", "per_observable": [], "worst_points": []}
    )
    assert rows_only == {
        "rollup_status": "unavailable",
        "rows_status": "read",
        "per_observable": [],
    }
    assert "flagged_observables" not in rows_only and "low_power" not in rows_only
    # every shape the score read can degrade to, and none of them is a clean bill: a non-dict, an
    # {ok: false} envelope, and a dict that simply does not carry a verdict
    for shape in ("a string", {"ok": False, "error": "boom"}, {"chi2": 1.0}, {}, None):
        degraded = _residual_structure_fields({"per_observable": []}, shape)
        assert degraded["rollup_status"] == "unavailable", shape
        assert "flagged_observables" not in degraded and "low_power" not in degraded
    # ...and the ROWS half gets the same treatment, since `per_observable: []` on an unreadable
    # diagnosis is otherwise byte-identical to a genuinely empty one
    verdict_only = _residual_structure_fields(
        "not a payload at all", {"flagged_observables": [], "low_power": False}
    )
    assert verdict_only["rows_status"] == "unavailable" and verdict_only["per_observable"] == []


def test_an_unread_verdict_reaches_the_history_csv_as_unknown():
    """result.json distinguishing UNKNOWN from a clean pass is only half of it: the sweep is grouped
    on the FLAT history columns, and omitting `replicate_lack_of_fit` blanked `lof_status`/`lof_weak`/
    `lof_df` exactly as a model-less run does. A run that HAS the block but could not read its verdict
    says so."""
    from petab_harness.metrics import _flatten_replicate_lof

    clean = _flatten_replicate_lof(
        {
            "residual_structure": {
                "rollup_status": "read",
                "replicate_lack_of_fit": {"status": "passed", "weak": False, "lof_df": 7},
            }
        }
    )
    assert (clean["lof_status"], clean["lof_weak"], clean["lof_df"]) == ("passed", False, 7)

    unread = _flatten_replicate_lof(
        {"residual_structure": {"rollup_status": "unavailable", "per_observable": []}}
    )
    assert unread["lof_status"] == "unknown"
    assert "lof_weak" not in unread and "lof_df" not in unread
    # ...and a run with no block at all still passes straight through, so "no model" stays its own row
    assert _flatten_replicate_lof({"scenario": "toy"}) == {"scenario": "toy"}


# --- noise-sigma plausibility flag (the hidden success gate) ------------------------------
def _sigma_entry(fitted, cap, frac):
    return {"fitted": fitted, "upper_bound_cap": cap, "data_std": 1.0, "fraction_of_data_std": frac}


def test_noise_sigma_flag_none_when_no_estimated_sigma():
    assert _noise_sigma_flag(None) is None
    assert _noise_sigma_flag({}) is None


def test_noise_sigma_flag_plausible_sigma_not_flagged():
    # fitted below its cap and well above the collapse floor -> neither inflated nor collapsed.
    f = _noise_sigma_flag({"sigma": _sigma_entry(0.08, 0.1, 0.4)})
    assert f["inflated"] is False and f["collapsed"] is False


def test_noise_sigma_flag_detects_inflation():
    # fitted above the data-driven cap -> inflated (absorbing misfit).
    f = _noise_sigma_flag({"sigma": _sigma_entry(0.5, 0.1, 2.5)})
    assert f["inflated"] is True and f["collapsed"] is False
    assert f["max_fraction_of_data_std"] == 2.5


def test_noise_sigma_flag_detects_collapse():
    # fraction below the collapse floor -> collapsed (overfitting the noise).
    frac = _NOISE_SIGMA_COLLAPSE_FRACTION / 2
    f = _noise_sigma_flag({"sigma": _sigma_entry(frac, 0.5, frac)})
    assert f["collapsed"] is True and f["inflated"] is False
    assert f["min_fraction_of_data_std"] == frac


def test_noise_sigma_flag_detects_both_inflated_and_collapsed():
    # one inflated + one collapsed estimated sigma in the same run -> BOTH flags set (the verdict
    # must then report both, not just one).
    f = _noise_sigma_flag(
        {
            "sig_hi": _sigma_entry(0.5, 0.1, 2.5),  # inflated
            "sig_lo": _sigma_entry(0.01, 0.5, _NOISE_SIGMA_COLLAPSE_FRACTION / 2),  # collapsed
        }
    )
    assert f["inflated"] is True and f["collapsed"] is True


def test_result_json_carries_the_replicate_lack_of_fit_verdict():
    """Without it a sweep cannot tell whether the arm flagged, passed or never ran.

    Establishing that the arm had fired at all on the 2026-08-02 blasi run meant reading the
    Phoenix trace, which does not scale to a run series.
    """
    from petab_harness.runner.scoring import _residual_structure_fields

    lof = {"available": True, "status": "flagged", "n_points": 15, "max_abs_z": 54.0}
    out = _residual_structure_fields(
        {"per_observable": []},
        {"flagged_observables": ["obs_A"], "low_power": False, "replicate_lack_of_fit": lof},
    )
    assert out["replicate_lack_of_fit"] == lof
    # ... and an arm that could not run is absent rather than a fabricated pass
    out = _residual_structure_fields(
        {"per_observable": []}, {"flagged_observables": [], "low_power": True}
    )
    assert "replicate_lack_of_fit" not in out


def test_a_non_dict_residual_structure_degrades_instead_of_losing_the_block():
    """`(rolled or {}).get(...)` raised AttributeError on a truthy NON-dict verdict, because the
    isinstance guard above it is satisfied by `per_observable` alone. judge.py catches that and sets
    residual_structure to None, so result.json lost the WHOLE block — including the per-observable rows
    that were fine — which is the sweep-unreadability this projection exists to close."""
    out = _residual_structure_fields(
        {
            "per_observable": [
                {
                    "observable_id": "obs_A",
                    "mean_weighted_residual": 0.2,
                    "sign_test": {"p": 0.4},
                    "runs_test": {"p": None},
                }
            ]
        },
        "residual-structure tests unavailable",
    )
    assert [r["observable_id"] for r in out["per_observable"]] == ["obs_A"]
    # ...and the unreadable verdict is reported as such, not as a clean one
    assert out["rollup_status"] == "unavailable"
    assert "flagged_observables" not in out and "replicate_lack_of_fit" not in out


def test_the_history_csv_has_a_column_for_the_replicate_arm(tmp_path):
    """The verdict reaching result.json is not the same as a sweep being able to read it: the history
    CSV writes `DictWriter(extrasaction="ignore")` against a fixed column tuple, so a nested block is
    dropped SILENTLY — exactly how the edit-reject metric was invisible for a release. `lof_weak` is
    the column this exists for: a pass and a pass on one degree of freedom are different runs."""
    from petab_harness import metrics

    result = {
        "scenario": "blasi",
        "residual_structure": {
            "replicate_lack_of_fit": {
                "available": True,
                "status": "pass",
                "weak": True,
                "lof_df": 1,
            }
        },
    }
    path = metrics.append_history([metrics.build_row(result)], tmp_path / "history.csv")
    header, values = path.read_text().splitlines()[:2]
    row = dict(zip(header.split(","), values.split(",")))
    assert (row["lof_status"], row["lof_weak"], row["lof_df"]) == ("pass", "True", "1")

    # `weak`/`lof_df` are OMITTED from the verdict when they say nothing, so the columns read them
    # with .get: an absent `weak` is not a weak pass, and a powered pass must not look like one.
    powered = {
        "scenario": "blasi",
        "residual_structure": {
            "replicate_lack_of_fit": {"available": True, "status": "pass", "lof_df": 7}
        },
    }
    path2 = metrics.append_history([metrics.build_row(powered)], tmp_path / "h2.csv")
    row2 = dict(zip(*(line.split(",") for line in path2.read_text().splitlines()[:2])))
    assert (row2["lof_status"], row2["lof_weak"], row2["lof_df"]) == ("pass", "False", "7")

    # a run with no such block at all leaves the columns empty rather than fabricating a status
    path3 = metrics.append_history([metrics.build_row({"scenario": "toy"})], tmp_path / "h3.csv")
    row3 = dict(zip(*(line.split(",") for line in path3.read_text().splitlines()[:2])))
    assert row3["lof_status"] == "" and row3["lof_weak"] == "" and row3["lof_df"] == ""

    # ...and a run whose arm has NO counted df — an unavailable arm, or a structural count the
    # parameter table would not give — leaves `lof_weak` empty too. `bool(lof.get("weak"))` wrote a
    # positive False there, so a sweep read "the arm ran and its pass was solid" off three archived
    # blasi runs whose verdicts predate the key, and off every no-evidence run. Same rule as
    # data_access.residual_flag: untested is not a clean bill of health.
    for block in (
        {"available": False, "status": "no_replicated_points"},
        {"available": True, "status": "pass"},
    ):
        p = metrics.append_history(
            [
                metrics.build_row(
                    {"scenario": "x", "residual_structure": {"replicate_lack_of_fit": block}}
                )
            ],
            tmp_path / f"h_{block['status']}_{block['available']}.csv",
        )
        row = dict(zip(*(line.split(",") for line in p.read_text().splitlines()[:2])))
        assert (row["lof_status"], row["lof_weak"], row["lof_df"]) == (block["status"], "", "")
