"""Wiring test for the hidden sigma gate in ``runner.judge._judge``:

    success = bic_pass AND not sigma_implausible

``_noise_sigma_flag`` is unit-tested in isolation (test_result_fields); THIS test drives the WHOLE
``_judge`` with fakes to prove the flag is actually wired into the success verdict end-to-end — a
good BIC bought with an inflated (or collapsed) estimated sigma must score ``success=False`` even
though the BIC gate passes, and a rejectable BIC must fail regardless of a plausible sigma. The gate
is post-run and never shown to the agent, so a regression here would silently start counting gamed
runs as passes. Importing the harness runner pulls in ``mcp``, so skip where that dep is absent.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest
from gemot.runner import (  # noqa: E402
    _judge,
    _reference_sigma_flag,
    _ToolClock,
)

# The server's collapse floor, as this tier's fake serves it (tests/fakes/server_api.py, held to the
# server's value by tests/server/test_server_api_fakes.py). Read off the fake rather than through
# gemot.server_api because this is module scope, where the conftest's swap is not yet in place.
from fakes.server_api import NOISE_SIGMA_COLLAPSE_FRACTION


def _sigma_entry(fitted, cap, frac):
    return {"fitted": fitted, "upper_bound_cap": cap, "data_std": 1.0, "fraction_of_data_std": frac}


_COLLAPSE_FRAC = NOISE_SIGMA_COLLAPSE_FRACTION / 2  # well below the collapse floor
_PLAUSIBLE = {"sigma": _sigma_entry(0.08, 0.1, 0.4)}  # below its cap, above the collapse floor
_INFLATED = {"sigma": _sigma_entry(0.5, 0.1, 2.5)}  # past the data-driven cap (absorbs misfit)
_COLLAPSED = {"sigma": _sigma_entry(_COLLAPSE_FRAC, 0.5, _COLLAPSE_FRAC)}  # overfits the noise
# schwen's shape: comfortably under its data-std cap, but 7.83x a MEASURED pure-error replicate floor
# — schwen's own published reference. #140 vetoed exactly this; that arm is now gone (a ratio of two
# point estimates discards the floor's uncertainty, and schwen's n_i = 2 technical duplicates make
# the floor ~8x too low). The judge must NOT veto on it.
_HIGH_PURE_ERROR_RATIO = {
    "sigma": {
        **_sigma_entry(0.187, 1.19, 0.117),
        "effective_sigma": 0.187,
        # schwen's REAL numbers, self-consistently: 0.187/1.592 = 0.117
        # and the cap is 0.75 x 1.592 = 1.19. A predicate that recomputes
        # the fraction from effective_sigma/data_std must land on the
        # same entry this pins, not on _sigma_entry's placeholder 1.0.
        "data_std": 1.592,
        "data_noise_estimate": 0.0238,
        "data_noise_method": "replicate",
        "fitted_over_data_noise": 7.83,
    }
}

# k = 3 estimated params (the estimate=0 row is the fixed noise sigma), n = 50 data points. The
# reference carries the SAME (nll, k, n) as the passing candidate, so a candidate nll of 100 gives
# ΔBIC = 0 (BIC passes) and an nll of 200 gives ΔBIC = +200 (rejectable).
_REF = {"nll": 100.0, "k": 3, "n": 50}
_PARAMS = [{"estimate": 1}, {"estimate": 1}, {"estimate": 1}, {"estimate": 0}]


class _Scenario:
    name = "wiring"
    bic_threshold = 10.0
    diagnostics = None

    def __init__(self, work: Path):
        # _judge writes result.json into `work` and prints it relative to example_dir.parent.
        self.example_dir = work / "ex"

    def reference(self):
        return dict(_REF)


class _Baseline:
    nll = 150.0
    species = {"A", "B"}


@pytest.fixture
def judge_calls():
    """Every ``(tool, kwargs)`` one judge run asked for, so a test can assert what it REQUESTED and
    not only what it did with the answer.

    Function-scoped, and passed in: as a module global cleared inside the runner this was
    order-dependent and broke under pytest-xdist (or any other caller of ``_make_call``)."""
    return []


def _make_call(summary, calls=None):
    """A fake async MCP `call`: canned per tool name. get_progress and describe_problem agree on
    v1 so _restore_best_checkpoint is a no-op (already on the best checkpoint); get_fit_summary
    carries the nll + noise_sigma_report that drive both gates; everything else is inert."""
    canned = {
        "get_progress": {"best_version": 1},
        # the HEAD version comes off describe_problem — there is no current_version tool
        "describe_problem": {"current_version": 1, "model_authored": True},
        "get_model_structure": {"reactions": [{"id": "r1"}], "species": ["A", "B"]},
        "get_fit_summary": summary,
        "get_parameters": _PARAMS,
        "get_fit_diagnosis": {},
        "diff_versions": {},
        "get_errors": [],
        "get_run_timing": {},
    }

    async def call(name, **kwargs):
        if calls is not None:
            calls.append((name, kwargs))
        return canned.get(name, {})

    return call


def _run_judge(work: Path, nll: float, sigma_report, calls=None):
    summary = {
        "neg_log_likelihood": nll,
        "n_data_points": 50,
        "chi2": 45.0,
        "noise_sigma_report": sigma_report,
    }
    return asyncio.run(
        _judge(
            _Scenario(work),
            _make_call(summary, calls),
            _Baseline(),
            work,
            {},
            _ToolClock(),
            time.monotonic(),
        )
    )


def test_the_judge_asks_for_the_residual_section_only_and_for_every_observable(
    tmp_path, judge_calls
):
    """Two things pinned in one call. The projection into result.json reads `per_observable` and
    nothing else, so it asks for the residual section only — the parameter half would be several
    kB the projection discards. And it asks with `max_observables=0` (NO cap): the agent-facing
    default of 12 exists for the agent's context window, while this projection is permanent and
    is what every later analysis of the run reads, so a capped table here would silently lose
    observables from result.json forever."""
    _run_judge(tmp_path, 100.0, _PLAUSIBLE, calls=judge_calls)
    asked = [kwargs for name, kwargs in judge_calls if name == "get_fit_diagnosis"]
    assert asked == [{"sections": "residuals", "max_observables": 0}]


def test_each_run_records_only_its_own_calls(tmp_path, judge_calls):
    """The recorder is per-test state, so an exact-contents assertion says something. As a module
    global cleared inside the runner it was order-dependent: any other caller appending to it
    changed what a later assertion saw."""
    other: list[tuple[str, dict]] = []
    for name in ("a", "b"):
        (tmp_path / name).mkdir()  # _judge writes result.json into the work dir
    _run_judge(tmp_path / "a", 100.0, _PLAUSIBLE, calls=judge_calls)
    _run_judge(tmp_path / "b", 100.0, _PLAUSIBLE, calls=other)
    # each recorder holds exactly one run's calls — the same ones, since the runs are identical
    assert judge_calls and other == judge_calls


def test_good_bic_plausible_sigma_succeeds(tmp_path):
    code, result = _run_judge(tmp_path, 100.0, _PLAUSIBLE)
    assert result["delta_bic"] == pytest.approx(0.0)  # BIC gate passes
    assert result["noise_sigma_flag"]["inflated"] is False  # ... and the sigma is plausible
    assert result["success"] is True and code == 0


def test_good_bic_implausible_sigma_fails(tmp_path):
    # SAME passing BIC, but the estimated sigma is inflated -> the hidden gate vetoes the pass.
    code, result = _run_judge(tmp_path, 100.0, _INFLATED)
    assert result["delta_bic"] == pytest.approx(0.0)  # BIC still passes ...
    assert result["noise_sigma_flag"]["inflated"] is True  # ... but sigma is implausible ...
    assert result["success"] is False and code == 1  # ... so the run does NOT count.


def test_good_bic_collapsed_sigma_fails(tmp_path):
    # The OTHER implausible branch: same passing BIC, but the estimated sigma has collapsed
    # (overfitting the noise) -> the hidden gate must veto it too, not only the inflated case.
    code, result = _run_judge(tmp_path, 100.0, _COLLAPSED)
    assert result["delta_bic"] == pytest.approx(0.0)  # BIC still passes ...
    assert result["noise_sigma_flag"]["collapsed"] is True  # ... but sigma has collapsed ...
    assert result["success"] is False and code == 1  # ... so the run does NOT count.


def test_good_bic_survives_a_high_pure_error_ratio(tmp_path):
    """The hidden gate must not veto a passing BIC on the pure-error RATIO alone: that number is a
    reference-quality diagnostic, not a verdict, and on schwen it condemns the data-generating model
    itself. The cap arm is unaffected (test_good_bic_implausible_sigma_fails), and the sharp replicate
    evidence now enters through the per-design-point lack-of-fit test instead."""
    code, result = _run_judge(tmp_path, 100.0, _HIGH_PURE_ERROR_RATIO)
    assert result["delta_bic"] == pytest.approx(0.0)  # BIC passes ...
    assert result["noise_sigma_flag"]["inflated"] is False  # ... 7.83x is not a veto
    assert result["noise_sigma_flag"]["collapsed"] is False
    assert result["success"] is True and code == 0


def test_bad_bic_fails_despite_plausible_sigma(tmp_path):
    # The other arm of the AND: a plausible sigma cannot rescue a rejectable BIC.
    code, result = _run_judge(tmp_path, 200.0, _PLAUSIBLE)
    assert result["delta_bic"] > _Scenario(tmp_path).bic_threshold
    assert result["noise_sigma_flag"]["inflated"] is False
    assert result["success"] is False and code == 1


# --- Design A: judge the sigma against the TRUE model's own fitted sigma (#118) --------------
#: A sigma pooled over enough points to HAVE power: 60 points, and models spending 12 of 120 and
#: 22 of 120 parameters, so both sides keep ~50 residual df. Its RAW sampling band is [0.696, 1.445],
#: but that is narrower than the fixed factor and the union keeps the wider one — so the band these
#: fixtures actually apply is [1/3, 3], i.e. the old fixed tolerance. That is the point: at high df
#: the df-awareness never binds, and these tests pin the unchanged behaviour there. The other end,
#: where it does bind, is test_the_sigma_band_widens_as_the_group_shrinks.
_WELL_POWERED = dict(n_points=60, k=12, n=120, ref_k=22, ref_n=120)


def _ref_flag(
    fitted,
    ref_sigma,
    *,
    n_points=_WELL_POWERED["n_points"],
    k=_WELL_POWERED["k"],
    n=_WELL_POWERED["n"],
):
    return _reference_sigma_flag(
        {"s": {"fitted": fitted, "n_points": n_points}},
        {"noise_sigma": {"s": ref_sigma}, "k": _WELL_POWERED["ref_k"], "n": _WELL_POWERED["ref_n"]},
        k=k,
        n=n,
    )


def test_reference_sigma_flag_accepts_a_sigma_near_the_truth():
    """A candidate whose sigma matches the reference fit's is plausible regardless of how it
    compares to the MARGINAL data std — the comparison #118 showed no fixed band can make."""
    f = _ref_flag(0.10, 0.10)
    assert (f["inflated"], f["collapsed"], f["max_ratio"], f["min_ratio"]) == (
        False,
        False,
        1.0,
        1.0,
    )
    # The reference basis must emit the SAME keys as the data-std fallback, so a consumer of
    # result.json's noise_sigma_flag sees one shape regardless of which basis produced it.
    assert "max_fraction_of_data_std" in f and "min_fraction_of_data_std" in f
    for ratio in (0.5, 2.0):  # comfortably inside the tolerance factor
        f = _ref_flag(0.10 * ratio, 0.10)
        assert not f["inflated"] and not f["collapsed"], ratio


def test_reference_sigma_flag_catches_both_tails():
    # Twice the fixed tolerance, read off the constant: the floor is what decides at this df, and a
    # literal here would quietly stop testing the tails if the factor ever moved.
    from gemot import server_api

    F = server_api.constants().NOISE_SIGMA_REFERENCE_FACTOR

    inflated = _ref_flag(0.10 * (F * 2), 0.10)
    assert inflated["inflated"] and not inflated["collapsed"]
    collapsed = _ref_flag(0.10 / (F * 2), 0.10)
    assert collapsed["collapsed"] and not collapsed["inflated"]


def test_the_sigma_band_widens_as_the_group_shrinks():
    """The point of the F band. A ratio of 0.25 is decisive evidence when the sigma was pooled over
    60 points and unremarkable when it came from 7 — the fixed factor of 3 it replaces called both
    "collapsed", which is what discarded brannmark run 9285a352 at dBIC -21.65 (its sigmaYDosR is
    pooled over SEVEN dose-response points, one per observable)."""
    powered = _ref_flag(0.025, 0.10)  # 60 points
    assert powered["collapsed"] and powered["per_sigma"]["s"]["verdict"] == "collapsed"

    thin = _ref_flag(0.025, 0.10, n_points=7, k=17, n=43)  # brannmark's own shape
    assert not thin["collapsed"] and not thin["inflated"]
    assert thin["per_sigma"]["s"]["verdict"] == "ok"
    # ...and the band is REPORTED, so a reader can see which df produced the verdict
    assert thin["per_sigma"]["s"]["band_low"] < 0.25 < thin["per_sigma"]["s"]["band_high"]
    assert powered["per_sigma"]["s"]["band_low"] > thin["per_sigma"]["s"]["band_low"]


def test_tolerance_never_tightens_as_the_group_gets_smaller():
    """The property the band exists to have: tolerance is monotone in df. Switching to the fixed
    factor below NOISE_SIGMA_BAND_MIN_DF would invert it — at brannmark's shape a five-point sigma
    judged in [0.333, 3.0] against a seven-point one's [0.154, 8.65]: thinner evidence, harsher
    test. The floor therefore clamps the df the band is computed at, so tolerance stops widening
    rather than collapsing."""
    bands = [
        _ref_flag(0.10, 0.10, n_points=m, k=17, n=43)["per_sigma"]["s"]
        for m in (3, 5, 7, 12, 20, 60)
    ]
    lows = [b["band_low"] for b in bands]
    assert lows == sorted(lows), f"a smaller group was judged more strictly: {lows}"
    highs = [b["band_high"] for b in bands]
    assert highs == sorted(highs, reverse=True), f"band narrowed downward as df fell: {highs}"
    # ...and the clamp bounds how permissive it gets rather than letting the gate vanish
    assert bands[0]["band_low"] > 0.0 and bands[0]["band_high"] < 100
    # the REPORTED df stays the real one, unclamped, so the audit trail is not a fiction
    assert bands[0]["df"] == pytest.approx(3 * (1 - 17 / 43))


def test_a_sigma_below_the_df_floor_is_still_gated():
    """Clamping is not switching the gate off: an absurd ratio on a three-point group is still
    caught, it just has to be more absurd than on a well-powered one."""
    f = _ref_flag(0.001, 0.10, n_points=3, k=17, n=43)  # 100x too small
    assert f["collapsed"] and f["per_sigma"]["s"]["verdict"] == "collapsed"


def test_testing_several_sigmas_splits_the_alpha():
    """Flagging when ANY of g sigmas is outside its band is a multiple comparison: at a flat alpha
    an honest run fails g times as often as the alpha claims."""
    # A group small enough that the SAMPLING band is the binding one (10 points, ~6 df) — where the
    # fixed floor binds, alpha cannot move anything.
    one = _reference_sigma_flag(
        {"a": {"fitted": 0.1, "n_points": 10}},
        {"noise_sigma": {"a": 0.1}, "k": 22, "n": 43},
        k=17,
        n=43,
    )
    four = _reference_sigma_flag(
        {p: {"fitted": 0.1, "n_points": 10} for p in "abcd"},
        {"noise_sigma": {p: 0.1 for p in "abcd"}, "k": 22, "n": 43},
        k=17,
        n=43,
    )
    assert four["alpha"] == one["alpha"] / 4
    assert four["per_sigma"]["a"]["band_low"] < one["per_sigma"]["a"]["band_low"]


def test_reference_sigma_flag_is_none_without_recorded_reference_sigmas():
    """An older reference.json carries only {nll,k,n}; the judge must fall back to the data-std
    band rather than silently passing everything."""
    assert _reference_sigma_flag({"s": {"fitted": 0.1}}, {"nll": 1.0, "k": 2, "n": 3}) is None
    assert _reference_sigma_flag({"s": {"fitted": 0.1}}, {"noise_sigma": {}}) is None
    assert _reference_sigma_flag({}, {"noise_sigma": {"s": 0.1}}) is None


def test_reference_sigma_pairs_a_lone_sigma_across_an_id_mismatch():
    """The reference is fit on the PRISTINE collection problem, whose ids can differ from the
    materialized ones (perelson ships `sd_viral_load`; the collection calls it
    `sd_task0_model0_perelson1_V`). With exactly ONE sigma on each side the pairing is unambiguous,
    so match positionally rather than silently losing the comparison."""
    f = _reference_sigma_flag(
        {"sd_viral_load": {"fitted": 0.12}},
        {"noise_sigma": {"sd_task0_model0_perelson1_V": 0.1228}},
    )
    assert f is not None and not f["inflated"] and not f["collapsed"]


def test_reference_sigma_does_not_guess_when_several_sigmas_mismatch():
    """More than one sigma on either side -> stay id-matched: a WRONG pairing is worse than none."""
    assert (
        _reference_sigma_flag(
            {"a": {"fitted": 0.1}, "b": {"fitted": 0.2}}, {"noise_sigma": {"x": 0.1, "y": 99.0}}
        )
        is None
    )


def test_reference_gate_judges_effective_sigma_when_available():
    """A composite sd (scale*sigma) is only comparable across models as the PRODUCT: a candidate at
    a different amplitude trades scale against sigma and reaches an identical sd with a raw sigma
    several-fold off. Judged raw it fails for matching the truth; judged effective it passes."""
    # truth: scale 4.0 * sigma 0.25 = 1.0. Candidate: scale 0.8 * sigma 1.25 = 1.0 (same sd).
    report = {"sd_o": {"fitted": 1.25, "effective_sigma": 1.0}}
    reference = {"noise_sigma": {"sd_o": 0.25}, "noise_sigma_effective": {"sd_o": 1.0}}
    flag = _reference_sigma_flag(report, reference)
    assert flag is not None and not flag["inflated"] and not flag["collapsed"]
    assert flag["max_ratio"] == pytest.approx(1.0)
    # Without the effective map the same pair is judged raw (1.25/0.25 = 5x) and IS flagged.
    raw_only = _reference_sigma_flag(report, {"noise_sigma": {"sd_o": 0.25}})
    assert raw_only is not None and raw_only["inflated"]


def test_every_sigma_gets_a_verdict_whatever_its_degrees_of_freedom():
    """There is no unjudged sigma. An early draft returned "no verdict" below the df floor, which
    would have let an agent collapse a sigma on a three-point group untouched; the fixed factor
    applies there instead. Each entry records which band decided it, so a pass is auditable."""
    for kwargs in (
        {"n_points": 60, "k": 12, "n": 120},  # sampling band binds
        {"n_points": 7, "k": 17, "n": 43},  # brannmark's shape
        {"n_points": 3, "k": 17, "n": 43},  # below the df floor -> fixed factor
        {"n_points": None, "k": 12, "n": 120},
    ):  # no stored count -> fixed factor
        detail = _ref_flag(0.10, 0.10, **kwargs)["per_sigma"]["s"]
        assert detail["verdict"] == "ok", kwargs
        assert detail["band_low"] < 1.0 < detail["band_high"], kwargs
    # ...and where the count is missing the band says so rather than inventing a df
    assert _ref_flag(0.10, 0.10, n_points=None)["per_sigma"]["s"]["df"] is None


def test_reference_sigma_basis_is_declined_for_a_scale_coupled_sigma():
    """fiedler's noise is `s_<gel> * sigma`, with the gels' `s_*` ESTIMATED over 77.7..999.99. Only
    the product is identified, so a model at different internal units reaches the SAME effective
    noise with a proportionally different sigma. The judge's reference basis divided one model's raw
    sigma by another's and called 19 of 20 fiedler runs implausible, silently failing 17 that had
    beaten the BIC bar (best dBIC -28.8; the one scored pass was +0.50)."""
    reference = {
        "noise_sigma": {"sigma_pMek": 0.0005804511382150003},
        "noise_sigma_effective": {"sigma_pMek": 0.0005804511382150003},
        "k": 22,
        "n": 72,
    }
    # The real numbers from 51526222_124: plausible on its own data-driven metrics, 179x the
    # reference's raw sigma.
    entry = {
        "fitted": 0.10386657663301647,
        "effective_sigma": 0.10386657663301647,
        "n_points": 36,
        "data_std": 0.33907019890637025,
        "fraction_of_data_std": 0.3063276482805788,
        "status": "ok",
    }

    coupled = _reference_sigma_flag(
        {"sigma_pMek": {**entry, "scale_coupled": True}}, reference, k=22, n=72
    )
    assert coupled is None  # declined -> the judge falls through to the data-driven band

    # Without the coupling the basis still applies, and still flags this ratio -- so the fix is
    # scoped to the coupling and does not disable the gate generally.
    plain = _reference_sigma_flag(
        {"sigma_pMek": {**entry, "scale_coupled": False}}, reference, k=22, n=72
    )
    assert plain is not None and plain["inflated"] is True
    assert plain["max_ratio"] > 100
