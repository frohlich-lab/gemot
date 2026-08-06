"""Which version the judge scores.

``_judge`` scores ONE version, and until #154 that was the server's best-BIC checkpoint. Two
independent mechanisms then conspired to score a model the run had already improved on:

* the checkpoint only moves when a fit beats it by ``NEW_BEST_MIN_DELTA_BIC`` (2.0), which is right
  for the streak/plateau logic and wrong for a final decision;
* the sigma gate that decides success is asked against the TRUE model's fitted sigmas, which the
  session cannot see — so the server has no way to prefer a version that would clear it.

brannmark run 9285a352 showed the first half live: ``v5`` (bic 344.175) improved on ``v4`` by 0.70,
under the threshold, so the checkpoint stayed at ``v4`` (bic 344.875) and the judge scored the worse
model. That run failed either way — every one of its versions fitted sigmaYDosR at 0.24-0.28 of the
reference's 17.58, so there was no clean model to find — but a discarded better version plus a gate
the server cannot evaluate is a pass waiting to be thrown away. These tests drive the real ``_judge``
over a fake version store, so the scan is exercised end-to-end rather than in isolation.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

pytest.importorskip("mcp", reason="harness-only dependency")

from petab_harness.runner import _judge, _ToolClock  # noqa: E402

#: brannmark's own numbers: the reference's fitted sigma for the observable that decided the run.
_REF_SIGMA = 17.578327340872296
#: brannmark v4's fitted value for it — 0.2495 of the reference, inside the collapsed half of the
#: [1/3, 3] band the judge allows (NOISE_SIGMA_REFERENCE_FACTOR = 3.0).
_COLLAPSED_SIGMA = 4.386
#: A value the gate accepts: 0.68 of the reference. Every version of that brannmark run sat between
#: 0.24 and 0.28, which is WHY the run failed — the clean alternative here is the counterfactual the
#: scan exists for, not something that run had.
_CLEAN_SIGMA = 12.0

_REF = {"nll": 141.889, "k": 22, "n": 43, "noise_sigma": {"sigmaYDosR": _REF_SIGMA}}
#: 12 estimated + one fixed, so k=12 — a candidate leaner than the 22-parameter truth, as brannmark's
#: passing runs were.
_PARAMS = [{"estimate": 1}] * 12 + [{"estimate": 0}]


def _report(sigma):
    return {"sigmaYDosR": {"fitted": sigma, "effective_sigma": sigma, "upper_bound_cap": 30.0,
                           "data_std": 34.5, "fraction_of_data_std": sigma / 34.5}}


class _Scenario:
    name = "scan"
    bic_threshold = 10.0
    diagnostics = None

    def __init__(self, work: Path):
        self.example_dir = work / "ex"

    def reference(self):
        return dict(_REF)


class _Baseline:
    nll = 200.0
    species = {"A", "B"}


class _Store:
    """A fake version store: ``list_versions`` ranks them, ``restore_version`` moves HEAD, and
    ``get_fit_summary`` answers for whichever version is currently restored — the same coupling the
    real server has, which is what makes the scan's restore-then-read loop meaningful here."""

    def __init__(self, versions: "dict[int, tuple[float, float]]", head: int,
                 checkpoint: "int | None", *, can_list: bool = True,
                 stores_reports: bool = True):
        self.versions = versions          # version -> (nll, fitted sigma)
        self.head = head
        self.checkpoint = checkpoint
        self.can_list = can_list
        self.stores_reports = stores_reports   # False = a store written before #155 persisted them
        self.calls: list[tuple[str, dict]] = []

    def _bic(self, version):              # 2*nll + k*ln(n), k=12 n=43 — the judge's own formula
        import math
        return 2 * self.versions[version][0] + 12 * math.log(43)

    async def __call__(self, name, **kwargs):
        self.calls.append((name, kwargs))
        if name == "list_versions":
            if not self.can_list:
                raise RuntimeError("this server has no list_versions")
            return [{"version": v, "bic": self._bic(v), "kind": "fit"} for v in self.versions]
        if name == "restore_version":
            self.head = kwargs["version"]
            return {"version": self.head}
        if name == "describe_problem":
            return {"current_version": self.head, "model_authored": True}
        if name == "get_progress":
            return {"best_version": self.checkpoint}
        if name == "get_version":
            if not self.stores_reports:
                return {"simulation": {"neg_log_likelihood": self.versions[kwargs["version"]][0]}}
            nll, sigma = self.versions[kwargs["version"]]
            return {"simulation": {"neg_log_likelihood": nll,
                                   "noise_sigma_report": _report(sigma)}}
        if name == "get_fit_summary":
            nll, sigma = self.versions[self.head]
            return {"neg_log_likelihood": nll, "n_data_points": 43, "chi2": 40.0,
                    "noise_sigma_report": _report(sigma)}
        if name == "get_parameters":
            return _PARAMS
        if name == "get_model_structure":
            return {"reactions": [{"id": "r1"}], "species": ["A", "B"]}
        return {} if name != "get_errors" else []

    @property
    def restored(self):
        return [kw["version"] for n, kw in self.calls if n == "restore_version"]


def _run(work: Path, store: _Store):
    return asyncio.run(_judge(_Scenario(work), store, _Baseline(), work, {},
                              _ToolClock(), time.monotonic()))


def test_the_judge_scores_the_best_version_whose_sigma_would_pass_the_gate(tmp_path):
    """The case the scan is for. Both versions beat the reference on BIC; the better one is
    sigma-collapsed and the other is clean, and the checkpoint points at the collapsed one because
    the clean version improved on it by less than NEW_BEST_MIN_DELTA_BIC (brannmark 9285a352 hit
    exactly that hysteresis). The run passes on the clean model instead of failing on the collapsed
    one."""
    store = _Store({4: (140.467, _COLLAPSED_SIGMA), 5: (140.117, _CLEAN_SIGMA)},
                   head=5, checkpoint=4)
    code, result = _run(tmp_path, store)

    assert store.head == 5, "scored the collapsed-sigma version the checkpoint pointed at"
    assert result["success"] is True and code == 0
    assert result["delta_bic"] < -10          # the pass was there to be had all along
    assert result["noise_sigma_flag"]["collapsed"] is False


def test_the_best_bic_version_is_scored_when_its_sigma_is_already_clean(tmp_path):
    """The common case must not pay for the regression: the scan stops at the first candidate, so a
    clean best-BIC model is scored without touching any other version."""
    store = _Store({4: (145.0, _CLEAN_SIGMA), 5: (140.117, _CLEAN_SIGMA)}, head=4, checkpoint=4)
    code, result = _run(tmp_path, store)

    assert store.head == 5 and result["success"] is True
    assert store.restored == [5], "walked past the best-BIC version"


def test_an_all_gamed_run_is_still_reported_on_its_own_best_bic_model(tmp_path):
    """No clean version exists, so there is nothing to prefer. The run is scored on its lowest-BIC
    model and FAILS on the sigma gate — the scan must not turn a gamed run into a pass by hunting
    for a version that clears the gate on a technicality, nor score some arbitrary worse model."""
    store = _Store({4: (140.467, _COLLAPSED_SIGMA), 5: (140.117, _COLLAPSED_SIGMA)},
                   head=4, checkpoint=4)
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
    versions[12] = (141.2, _CLEAN_SIGMA)          # cleanest sigma, worst bic of the twelve
    store = _Store(versions, head=1, checkpoint=1)
    code, result = _run(tmp_path, store)

    assert store.head == 12 and result["success"] is True
    assert [kw["version"] for n, kw in store.calls if n == "get_version"] == list(range(1, 13)), \
        "stopped scanning before the last version"
    assert store.restored == [12], "the scan is a READ — only the chosen version is restored"


def test_a_server_that_cannot_list_versions_falls_back_to_the_checkpoint(tmp_path):
    """The scan needs `list_versions`; without it the judge restores `best_version` exactly as it
    did before, rather than scoring whatever HEAD happens to be."""
    store = _Store({4: (140.467, _CLEAN_SIGMA), 5: (150.0, _CLEAN_SIGMA)},
                   head=5, checkpoint=4, can_list=False)
    code, result = _run(tmp_path, store)

    assert store.head == 4 and store.restored == [4]
    assert result["success"] is True


def test_the_scan_reads_stored_reports_instead_of_rolling_through_the_versions(tmp_path):
    """The scan asks each version what its fit already recorded, so looking at a candidate costs a
    read — not a rollback. Restoring each one in turn mutated the working state N times and wrote N
    entries into the provenance the run is judged on, to reach numbers stored beside every version.
    Exactly one restore happens: the version actually scored."""
    store = _Store({4: (140.467, _COLLAPSED_SIGMA), 5: (140.117, _CLEAN_SIGMA)},
                   head=4, checkpoint=4)
    _run(tmp_path, store)

    assert store.restored == [5]
    # v5 is best-BIC AND clean, so the scan stops there: one read, one restore, nothing else touched
    assert [kw["version"] for n, kw in store.calls if n == "get_version"] == [5]


def test_a_store_without_saved_sigma_reports_scores_the_best_bic_version(tmp_path):
    """Version dirs written before the report was persisted (a resumed run, an older store) are
    unassessable rather than flagged, so the scan lands on the best-BIC model — the behaviour before
    the scan existed — instead of ranking versions on a verdict it cannot read."""
    store = _Store({4: (145.0, _COLLAPSED_SIGMA), 5: (140.117, _COLLAPSED_SIGMA)},
                   head=4, checkpoint=4, stores_reports=False)
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

    store = _Refusing({4: (140.467, _COLLAPSED_SIGMA), 5: (140.117, _CLEAN_SIGMA)},
                      head=4, checkpoint=4)
    code, result = _run(tmp_path, store)

    assert store.head == 4, "HEAD moved despite the restore failing"
    # scored what is actually there (v4, the collapsed one) and failed it — not a phantom v5 pass
    assert result["success"] is False
    assert result["noise_sigma_flag"]["collapsed"] is True
