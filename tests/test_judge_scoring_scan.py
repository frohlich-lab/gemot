"""Which version the judge scores.

``_judge`` scores ONE version, and until #154 that was the server's best-BIC checkpoint, which only
moves when a fit beats it by ``NEW_BEST_MIN_DELTA_SCORE`` (2.0): right for the streak/plateau logic,
wrong for a final decision. brannmark run 9285a352: ``v5`` (bic 344.175) improved on ``v4`` by 0.70,
under the threshold, so the checkpoint stayed at ``v4`` (bic 344.875) and the judge scored the worse
model. The scan walks every fitted version instead, best first, and stops at the first whose noise
sigma the server does not flag: the finish gate's scan, with the finish gate's predicate, so the
version the gate names to the agent is the one scored here (gemot#135). These tests drive the real
``_judge`` over a fake version store, so the scan is exercised end-to-end rather than in isolation.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

pytest.importorskip("mcp", reason="harness-only dependency")

from gemot.runner import _judge, _noise_sigma_flag, _ToolClock  # noqa: E402

#: brannmark's own numbers: the reference's fitted sigma for sigmaYDosR, and the data's spread.
_REF_SIGMA = 17.578327340872296
_DATA_STD = 34.5
#: The replicate pure error of sigmaYDosR's data, as the server reports it. A collapse is judged
#: only against it, below half of it (PEtab-MCP#374).
_PURE_ERROR = 2.0
#: A sigma the server flags collapsed: under half the replicate pure error.
_COLLAPSED_SIGMA = 0.2
#: A sigma the server accepts, 0.68 of the reference's.
_CLEAN_SIGMA = 12.0
#: brannmark v4's fitted value, 0.25 of the reference's: outside the 1/3-3x band the judge used to
#: fail runs on, inside the server's. Since gemot#135 that ratio is only recorded.
_BELOW_REFERENCE_SIGMA = 4.386

_REF = {"nll": 141.889, "k": 22, "n": 43, "noise_sigma": {"sigmaYDosR": _REF_SIGMA}}
#: 12 estimated + one fixed, so k=12 — a candidate leaner than the 22-parameter truth, as brannmark's
#: passing runs were.
_PARAMS = [{"estimate": 1}] * 12 + [{"estimate": 0}]


def _report(sigma):
    return {
        "sigmaYDosR": {
            "fitted": sigma,
            "effective_sigma": sigma,
            "upper_bound_cap": 30.0,
            "data_std": _DATA_STD,
            "fraction_of_data_std": sigma / _DATA_STD,
            "data_noise_method": "replicate",
            "data_noise_estimate": _PURE_ERROR,
        }
    }


def _flags(report):
    """The verdict the server stores in a version's meta, by the server's own predicates."""
    flag = _noise_sigma_flag(report)
    return {"noise_sigma_inflated": flag["inflated"], "noise_sigma_collapsed": flag["collapsed"]}


class _Scenario:
    name = "scan"
    bic_threshold = 10.0
    diagnostics = None

    def __init__(self, work: Path, ref: "dict | None" = None):
        self.example_dir = work / "ex"
        self._ref = dict(_REF if ref is None else ref)

    def reference(self):
        return dict(self._ref)


class _Baseline:
    nll = 200.0
    species = {"A", "B"}


class _Store:
    """A fake version store: ``list_versions`` ranks them, ``restore_version`` moves HEAD, and
    ``get_fit_summary`` answers for whichever version is currently restored — the same coupling the
    real server has, which is what makes the scan's restore-then-read loop meaningful here."""

    def __init__(
        self,
        versions: "dict[int, tuple[float, float]]",
        head: int,
        checkpoint: "int | None",
        *,
        can_list: bool = True,
        stores_reports: bool = True,
        criterion_values: "dict[int, float] | None" = None,
        criterion: "str | None" = "bic",
        k_eff: "float | None" = None,
    ):
        self.versions = versions  # version -> (nll, fitted sigma)
        # The effective parameter count the server reports on a fit — TIC's input. None = a fit
        # that produced none (or a server predating it).
        self.k_eff = k_eff
        # version -> the score under the run's OWN criterion, as the server records it beside the
        # canonical bic. None = a store predating it, which is the bic-fallback path.
        self.criterion_values = criterion_values
        # The criterion label the server reports on every fit — what the judge's gate reads to
        # decide which penalty to grade under. None = report NOTHING, the shape of a run that
        # never produced a fit to report one.
        self.criterion = criterion
        self.head = head
        self.checkpoint = checkpoint
        self.can_list = can_list
        self.stores_reports = stores_reports  # False = versions stored before their sigma flags
        self.calls: list[tuple[str, dict]] = []

    def _bic(self, version):  # 2*nll + k*ln(n), k=12 n=43 — the judge's own formula
        import math

        return 2 * self.versions[version][0] + 12 * math.log(43)

    async def __call__(self, name, **kwargs):
        self.calls.append((name, kwargs))
        if name == "list_versions":
            if not self.can_list:
                raise RuntimeError("this server has no list_versions")
            cv = self.criterion_values or {}
            return [
                {
                    "version": v,
                    "bic": self._bic(v),
                    "kind": "fit",
                    **({"criterion_value": cv[v]} if v in cv else {}),
                }
                for v in self.versions
            ]
        if name == "restore_version":
            self.head = kwargs["version"]
            return {"version": self.head}
        if name == "describe_problem":
            return {"current_version": self.head, "model_authored": True}
        if name == "get_progress":
            return {"best_version": self.checkpoint}
        if name == "get_version":
            # The meta (the server's sigma flags) and the simulation part, whether or not asked for.
            version = kwargs["version"]
            nll = self.versions[version][0]
            if not self.stores_reports:
                return {"simulation": {"neg_log_likelihood": nll}}
            report = self._report_for(version)
            return {
                **_flags(report),
                "simulation": {
                    "neg_log_likelihood": nll,
                    **({"k_eff": self.k_eff} if self.k_eff is not None else {}),
                    "noise_sigma_report": report,
                },
            }
        if name == "get_fit_summary":
            return {
                "neg_log_likelihood": self.versions[self.head][0],
                "n_data_points": 43,
                "chi2": 40.0,
                **({"criterion": self.criterion} if self.criterion else {}),
                "noise_sigma_report": self._report_for(self.head),
            }
        if name == "get_parameters":
            return _PARAMS
        if name == "get_model_structure":
            return {"reactions": [{"id": "r1"}], "species": ["A", "B"]}
        return {} if name != "get_errors" else []

    def _report_for(self, version):
        return _report(self.versions[version][1])

    @property
    def restored(self):
        return [kw["version"] for n, kw in self.calls if n == "restore_version"]


def _run(work: Path, store: _Store, ref: "dict | None" = None):
    return asyncio.run(
        _judge(_Scenario(work, ref), store, _Baseline(), work, {}, _ToolClock(), time.monotonic())
    )


def test_the_effective_parameter_count_is_read_off_the_stored_version(tmp_path):
    """`k_eff` rides into result.json off the STORED scored version — never off `get_fit_summary`,
    which the agent reads, and never re-derived (it needs the built objective) — and is None, not
    0, not k, when the store has none."""
    store = _Store({4: (140.117, _CLEAN_SIGMA)}, head=4, checkpoint=4, k_eff=9.5)
    _, result = _run(tmp_path, store)
    assert result["k_eff"] == 9.5 and result["k"] == 12

    store = _Store({4: (140.117, _CLEAN_SIGMA)}, head=4, checkpoint=4)
    _, result = _run(tmp_path, store)
    assert result["k_eff"] is None


def test_the_judge_scores_the_best_version_whose_sigma_would_pass_the_gate(tmp_path):
    """The case the scan is for. Both versions beat the reference on BIC; the better one is
    sigma-collapsed and the other is clean, and the checkpoint points at the collapsed one because
    the clean version improved on it by less than NEW_BEST_MIN_DELTA_SCORE (brannmark 9285a352 hit
    exactly that hysteresis). The run passes on the clean model instead of failing on the collapsed
    one."""
    store = _Store(
        {4: (140.467, _COLLAPSED_SIGMA), 5: (140.117, _CLEAN_SIGMA)}, head=5, checkpoint=4
    )
    code, result = _run(tmp_path, store)

    assert store.head == 5, "scored the collapsed-sigma version the checkpoint pointed at"
    assert result["success"] is True and code == 0
    assert result["delta_bic"] < -10  # the pass was there to be had all along
    assert result["noise_sigma_flag"]["collapsed"] is False


def test_the_best_bic_version_is_scored_when_its_sigma_is_already_clean(tmp_path):
    """The common case must not pay for the regression: the scan stops at the first candidate, so a
    clean best-BIC model is scored without touching any other version."""
    store = _Store({4: (145.0, _CLEAN_SIGMA), 5: (140.117, _CLEAN_SIGMA)}, head=4, checkpoint=4)
    code, result = _run(tmp_path, store)

    assert store.head == 5 and result["success"] is True
    assert store.restored == [5], "walked past the best-BIC version"


def test_the_pass_gate_grades_under_the_criterion_the_run_searched_by(tmp_path):
    """A run that searched by AICc is graded by AICc, not by canonical BIC: it would otherwise be
    optimising one penalty and assessed on another. Both candidate and reference are re-scored
    under the criterion the server REPORTS, and canonical BIC is still recorded beside it."""
    store = _Store({4: (140.117, _CLEAN_SIGMA)}, head=4, checkpoint=4, criterion="aicc")
    code, result = _run(tmp_path, store)

    assert result["criterion"] == "aicc"
    assert result["delta_criterion"] == pytest.approx(
        result["criterion_value"] - result["criterion_reference"]
    )
    # The DELTA itself moves, not just the absolute scores — the candidate is leaner than the
    # truth (k=12 vs 22 on n=43), so the two penalties disagree about how much that is worth:
    # ΔBIC -41.2 against ΔAICc -63.7. A gate still reading canonical BIC would report -41.2 here.
    assert result["delta_criterion"] == pytest.approx(-63.744, abs=0.01)
    assert result["delta_bic"] == pytest.approx(-41.156, abs=0.01)
    # ...and canonical BIC is still recorded beside it.
    assert result["bic"] is not None and result["bic_reference"] is not None


def test_a_run_that_reports_no_criterion_records_none_not_bic(tmp_path):
    """The gate FALLS BACK to canonical BIC when no criterion is readable, but the run must not
    RECORD that fallback as its arm: an aborted AICc run filed under `criterion: "bic"` pools with
    the real BIC arm in metrics_history.csv, which is the mislabelling the column exists to stop."""
    store = _Store({4: (140.117, _CLEAN_SIGMA)}, head=4, checkpoint=4, criterion=None)
    code, result = _run(tmp_path, store)

    assert result["criterion"] is None, "recorded the gate's fallback as the run's arm"
    # ...while the gate itself still graded, on canonical BIC.
    assert result["delta_criterion"] == result["delta_bic"] is not None


def test_every_criterion_is_recorded_candidate_against_reference(tmp_path):
    """`result.json` carries the whole table, not just the gate's row and canonical BIC — so a run
    can be read under a criterion it was neither run nor graded with, and a criterion sweep's arms
    can be put on each other's scale. The two flat rows the CSV/W&B allowlists take must agree
    with it."""
    from petab_mcp.criteria import PENALTIES

    store = _Store({4: (140.117, _CLEAN_SIGMA)}, head=4, checkpoint=4, criterion="aicc")
    _, result = _run(tmp_path, store)

    table = result["criteria"]
    assert set(table) == set(PENALTIES)
    assert table["bic"] == {
        "candidate": result["bic"],
        "reference": result["bic_reference"],
        "delta": result["delta_bic"],
    }
    assert table["aicc"]["delta"] == pytest.approx(result["delta_criterion"])
    assert table["caic"]["delta"] == pytest.approx(result["delta_bic"] + (12 - 22))
    # No k_eff on either side: TIC has no value, and says so with None rather than a number.
    assert result["k_eff"] is None
    assert table["tic"] == {"candidate": None, "reference": None, "delta": None}


def test_a_tic_run_against_a_reference_without_a_k_eff_is_graded_by_bic_and_says_so(tmp_path):
    """The reference.json files in the tree predate `k_eff`; a TIC arm on one of those must not
    fail every run as "could not evaluate". Canonical BIC grades it, the note names the fallback,
    and the arm is still RECORDED as tic — with its own TIC score, which is computable."""
    store = _Store({4: (140.117, _CLEAN_SIGMA)}, head=4, checkpoint=4, criterion="tic", k_eff=11.0)
    _, result = _run(tmp_path, store)

    assert result["criterion"] == "tic"
    assert result["k_eff"] == 11.0
    assert "tic has no value for the reference" in result["criterion_note"]
    assert result["delta_criterion"] == result["delta_bic"]
    assert result["criteria"]["tic"]["candidate"] == pytest.approx(2 * 140.117 + 2 * 11.0)
    assert result["criteria"]["tic"]["reference"] is None


def test_a_tic_run_against_a_regenerated_reference_is_graded_by_tic(tmp_path):
    """Once `python -m petab_agent_bench --refit` has recorded the reference fit's own k_eff, the
    gate follows the criterion: a candidate whose fit the data see as 11 parameters against a truth
    seen as 20."""
    store = _Store({4: (140.117, _CLEAN_SIGMA)}, head=4, checkpoint=4, criterion="tic", k_eff=11.0)
    _, result = _run(tmp_path, store, ref={**_REF, "k_eff": 20.0})

    assert result["criterion_note"] is None
    assert result["criterion_reference"] == pytest.approx(2 * 141.889 + 2 * 20.0)
    assert result["delta_criterion"] == pytest.approx(
        (2 * 140.117 + 2 * 11.0) - (2 * 141.889 + 2 * 20.0)
    )
    assert result["delta_criterion"] == pytest.approx(result["criteria"]["tic"]["delta"])
    assert result["success"] is True


def test_the_pass_gate_is_canonical_bic_under_the_default_criterion(tmp_path):
    """The default must be byte-identical: `criterion_value == bic` and the gate is unchanged."""
    store = _Store({4: (140.117, _CLEAN_SIGMA)}, head=4, checkpoint=4)
    code, result = _run(tmp_path, store)

    assert result["criterion"] == "bic"
    assert result["criterion_value"] == result["bic"]
    assert result["delta_criterion"] == result["delta_bic"]


def test_the_scan_ranks_candidates_by_the_runs_own_criterion_not_by_bic(tmp_path):
    """The scan complements the session's best-SCORE checkpoint, so "best first" has to be the
    run's own criterion. Here v4 is the better model under that criterion while v5 is the better
    one on canonical bic (nll 145.0 vs 140.117) — both sigma-clean, so ordering alone decides
    which gets scored. Ranking by bic would score v5, the one the run never preferred."""
    store = _Store(
        {4: (145.0, _CLEAN_SIGMA), 5: (140.117, _CLEAN_SIGMA)},
        head=5,
        checkpoint=4,
        criterion_values={4: 100.0, 5: 300.0},
    )
    code, result = _run(tmp_path, store)

    assert store.restored == [4], "ordered by bic instead of the run's criterion"
    assert store.head == 4


def test_the_scan_falls_back_to_bic_for_the_WHOLE_set_when_any_version_lacks_a_score(tmp_path):
    """A store spanning this upgrade carries `criterion_value` on its LATER versions only. Mixing
    the two keys in one sort would compare v4's canonical BIC against v5's AICc — different
    scales, arbitrary order — so the fallback is all-or-nothing. v5's criterion value would put it
    LAST if it were honoured; because v4 has none, the whole ordering reverts to bic and v5 (the
    lower-bic model) is scored."""
    store = _Store(
        {4: (145.0, _CLEAN_SIGMA), 5: (140.117, _CLEAN_SIGMA)},
        head=4,
        checkpoint=4,
        criterion_values={5: 9_999.0},
    )  # v4 predates it — v5's score must be ignored
    code, result = _run(tmp_path, store)

    assert store.restored == [5], "honoured a criterion value the rest of the store cannot match"


def test_an_all_gamed_run_is_still_reported_on_its_own_best_bic_model(tmp_path):
    """No clean version exists, so there is nothing to prefer. The run is scored on its lowest-BIC
    model and FAILS on the sigma gate — the scan must not turn a gamed run into a pass by hunting
    for a version that clears the gate on a technicality, nor score some arbitrary worse model."""
    store = _Store(
        {4: (140.467, _COLLAPSED_SIGMA), 5: (140.117, _COLLAPSED_SIGMA)}, head=4, checkpoint=4
    )
    code, result = _run(tmp_path, store)

    assert store.head == 5, "scored something other than the best-BIC model"
    assert result["success"] is False and code == 1
    assert result["noise_sigma_flag"]["collapsed"] is True
    assert result["delta_bic"] < -10, "the BIC half still passed — the sigma gate is what failed it"


def test_every_fitted_version_is_a_candidate_not_just_the_first_few(tmp_path):
    """The scan is not truncated: a run's version count is already bounded by its turn budget, and a
    cap would give up the best scoreable model exactly like the checkpoint it replaces — the same
    failure one rung down. Here the only clean model is the WORST of twelve."""
    versions = {v: (140.0 + v * 0.1, _COLLAPSED_SIGMA) for v in range(1, 13)}
    versions[12] = (141.2, _CLEAN_SIGMA)  # cleanest sigma, worst bic of the twelve
    store = _Store(versions, head=1, checkpoint=1)
    code, result = _run(tmp_path, store)

    assert store.head == 12 and result["success"] is True
    # Every version read once by the scan, then the chosen one read again for the record (k_eff).
    assert [kw["version"] for n, kw in store.calls if n == "get_version"] == list(range(1, 13)) + [
        12
    ], "stopped scanning before the last version"
    assert store.restored == [12], "the scan is a READ — only the chosen version is restored"


def test_a_server_that_cannot_list_versions_falls_back_to_the_checkpoint(tmp_path):
    """The scan needs `list_versions`; without it the judge restores `best_version` exactly as it
    did before, rather than scoring whatever HEAD happens to be."""
    store = _Store(
        {4: (140.467, _CLEAN_SIGMA), 5: (150.0, _CLEAN_SIGMA)}, head=5, checkpoint=4, can_list=False
    )
    code, result = _run(tmp_path, store)

    assert store.head == 4 and store.restored == [4]
    assert result["success"] is True


def test_the_scan_reads_stored_reports_instead_of_rolling_through_the_versions(tmp_path):
    """The scan asks each version what its fit already recorded, so looking at a candidate costs a
    read — not a rollback. Restoring each one in turn mutated the working state N times and wrote N
    entries into the provenance the run is judged on, to reach numbers stored beside every version.
    Exactly one restore happens: the version actually scored."""
    store = _Store(
        {4: (140.467, _COLLAPSED_SIGMA), 5: (140.117, _CLEAN_SIGMA)}, head=4, checkpoint=4
    )
    _run(tmp_path, store)

    assert store.restored == [5]
    # v5 is best-BIC AND clean, so the scan stops there: one read to choose it, one restore, and one
    # more read of the SCORED version for the record (its stored k_eff) — no other version touched.
    assert [kw["version"] for n, kw in store.calls if n == "get_version"] == [5, 5]


def test_a_store_without_saved_sigma_reports_scores_the_best_bic_version(tmp_path):
    """Version dirs written before the report was persisted (a resumed run, an older store) are
    unassessable rather than flagged, so the scan lands on the best-BIC model — the behaviour before
    the scan existed — instead of ranking versions on a verdict it cannot read."""
    store = _Store(
        {4: (145.0, _COLLAPSED_SIGMA), 5: (140.117, _COLLAPSED_SIGMA)},
        head=4,
        checkpoint=4,
        stores_reports=False,
    )
    _run(tmp_path, store)

    assert store.head == 5 and store.restored == [5]


def test_a_failed_restore_says_so_instead_of_naming_a_version_nothing_scored(tmp_path):
    """The scan picks a version and the judge announces it — but the announcement has to follow the
    restore, not precede it. With the scan read-only, HEAD is wherever the AGENT left it, so a failed
    restore does not leave the runner-up in place: it leaves an arbitrary model, and a log line
    naming v5 would then describe a score nothing produced."""

    class _Refusing(_Store):
        async def __call__(self, name, **kwargs):
            if name == "restore_version":
                self.calls.append((name, kwargs))
                raise RuntimeError("restore refused")
            return await super().__call__(name, **kwargs)

    store = _Refusing(
        {4: (140.467, _COLLAPSED_SIGMA), 5: (140.117, _CLEAN_SIGMA)}, head=4, checkpoint=4
    )
    code, result = _run(tmp_path, store)

    assert store.head == 4, "HEAD moved despite the restore failing"
    # scored what is actually there (v4, the collapsed one) and failed it — not a phantom v5 pass
    assert result["success"] is False
    assert result["noise_sigma_flag"]["collapsed"] is True


#: isensee's reference sigma for the microscopy pRII readout, whose noise formula names its scales
#: (`noiseParameter1_pRII_Microscopy * s_pRII_global * rho_pRII_Microscopy`; gemot#104).
_ISENSEE_RHO = 0.06996313044208774
_ISENSEE_REF = {
    "nll": 141.889,
    "k": 22,
    "n": 43,
    "noise_sigma": {"rho_pRII_Microscopy": _ISENSEE_RHO},
    "noise_sigma_effective": {"rho_pRII_Microscopy": _ISENSEE_RHO},
}


class _IsenseeStore(_Store):
    """The version store of isensee r2, cut to its two versions that matter: v15 beats the bar
    with `rho_pRII_Microscopy` at 0.25 of the reference, v3 sits at 0.94 and misses it. Both are
    inside the server's band. ``scale_coupled`` is the server's flag on both entries: False at
    PEtab-MCP 26e5b76, which read only `noiseParameters`, True since PEtab-MCP#243 counts the scales
    in the formula."""

    def __init__(self, *, scale_coupled):
        super().__init__(
            {3: (167.0, 0.94 * _ISENSEE_RHO), 15: (140.117, 0.25 * _ISENSEE_RHO)},
            head=15,
            checkpoint=15,
        )
        self.scale_coupled = scale_coupled

    def _report_for(self, version):
        rho = self.versions[version][1]
        return {
            "rho_pRII_Microscopy": {
                "fitted": rho,
                "effective_sigma": rho,
                "n_points": 30,
                "upper_bound_cap": 0.15,
                "data_std": 0.2,
                "fraction_of_data_std": rho / 0.2,
                "scale_coupled": self.scale_coupled,
            }
        }


def test_a_reference_ratio_no_longer_costs_the_run_its_better_version(tmp_path):
    """gemot#104, isensee r2: compared raw against the reference, v15 (rho at 0.25 of it) was
    refused and the judge scored v3 (+11.6, a FAIL), though v15's noise in measurement units matched
    v3's. The server's flag decides now (gemot#135), so v15 is scored and passes whether or not the
    server marks the sigma scale-coupled. The reference ratio is recorded where it can be paired:
    not for a scale-coupled sigma, which result.json names."""
    store = _IsenseeStore(scale_coupled=True)
    code, result = _run(tmp_path, store, _ISENSEE_REF)
    assert store.head == 15 and result["success"] is True and code == 0
    assert result["noise_sigma_reference"] is None
    assert result["noise_sigma_scale_coupled"] == ["rho_pRII_Microscopy"]

    store = _IsenseeStore(scale_coupled=False)
    code, result = _run(tmp_path, store, _ISENSEE_REF)
    assert store.head == 15 and result["success"] is True and code == 0  # not the round-4 failure
    assert result["noise_sigma_reference"]["collapsed"] is True
    assert result["noise_sigma_scale_coupled"] == []


def test_the_judge_scores_a_version_whose_sigma_is_far_below_the_references(tmp_path):
    """gemot#135, boehm r39: from v14 to v21 the run gained BIC by pushing a sigma to 0.12x the
    reference's, the finish gate said it judged v21, and the judge scored v13 on a reference rule the
    agent never saw. The server passes that sigma, so the judge now scores v21, as promised."""
    store = _Store({13: (141.0, _CLEAN_SIGMA), 21: (140.117, _BELOW_REFERENCE_SIGMA)}, 13, 13)
    code, result = _run(tmp_path, store)
    assert store.head == 21 and result["success"] is True and code == 0
    assert result["noise_sigma_flag"]["collapsed"] is False
    assert result["noise_sigma_reference"]["collapsed"] is True


# --- the scan the finish gate shares (gemot#135) -----------------------------------------------
# The gate and this judge choose the scored version with ONE function, `probe.scored_version`, and
# one predicate, `probe.stored_sigma_flagged`. Before, each wrote the loop out and the judge asked a
# second, hidden question (the reference's sigmas), so the gate told boehm r39 it judged v21 while
# the judge scored v13.


def _verdicts(by_version):
    async def flagged(version):
        return by_version[version]

    return flagged


def test_the_scan_stops_at_the_first_version_its_predicate_does_not_flag():
    from gemot.agent.probe import scored_version

    flags = _verdicts({4: True, 5: False, 3: False})
    assert asyncio.run(scored_version([4, 5, 3], flags)) == (5, True)


def test_an_unassessable_version_stops_the_scan_like_a_clean_one():
    """The judge's gate does not fail a version it cannot assess, so passing it over would score a
    worse version for nothing."""
    from gemot.agent.probe import scored_version

    assert asyncio.run(scored_version([4, 5], _verdicts({4: None, 5: False}))) == (4, True)


def test_when_every_version_is_flagged_the_scan_returns_the_best_scoring_one_unclean():
    from gemot.agent.probe import scored_version

    assert asyncio.run(scored_version([4, 5], _verdicts({4: True, 5: True}))) == (4, False)


class _OneStore:
    """One version store answered through both call handles: the judge's ``call(name, **kwargs)``
    and the finish gate's ``call_tool(name, args)``, whose result carries ``structured_content``."""

    def __init__(self, scores, flagged, head):
        self.scores, self.flagged, self.head = scores, flagged, head

    def _answer(self, name, kw):
        if name == "list_versions":
            return [{"version": v, "bic": b, "criterion_value": b} for v, b in self.scores.items()]
        if name == "get_version":
            collapsed = self.flagged[kw["version"]]
            return {"noise_sigma_inflated": False, "noise_sigma_collapsed": collapsed}
        if name == "describe_problem":
            return {"current_version": self.head}
        if name == "restore_version":
            self.head = kw["version"]
            return {"version": self.head}
        raise AssertionError(f"unexpected tool {name}")

    async def judge_call(self, name, **kwargs):
        return self._answer(name, kwargs)

    async def gate_call(self, name, args):
        from types import SimpleNamespace

        return SimpleNamespace(structured_content={"result": self._answer(name, args)})


def test_the_finish_gate_and_the_judge_choose_the_same_version():
    """The promise gemot#135 is about: the version the finish gate restores and names to the agent
    is the version the judge scores. boehm r39's shape, v21 best, with its sigma flagged in one case
    and not in the other."""
    from gemot.agent.probe import _restore_best
    from gemot.runner import _restore_best_checkpoint

    scores = {13: 263.5, 14: 253.7, 21: 224.8}
    for flagged, expected in (
        ({13: False, 14: False, 21: False}, 21),
        ({13: False, 14: False, 21: True}, 14),
    ):
        gate = _OneStore(scores, flagged, head=13)
        named = asyncio.run(_restore_best(gate.gate_call))
        judge = _OneStore(scores, flagged, head=13)
        scored = asyncio.run(_restore_best_checkpoint(judge.judge_call))
        assert named == (expected, 13) and scored == expected == gate.head == judge.head
