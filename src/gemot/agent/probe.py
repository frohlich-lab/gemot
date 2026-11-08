"""Reading the server's CURRENT state through a bare call handle.

The finish gate and the driver's run loop both need to ask the server where things stand — is
there a fitted model, what is its bic, what do the residuals look like — without going through
the agent. They share these helpers so the payload-unwrapping and fail-open rules live in one
place. :func:`best_first` is shared with the judge as well, so the version the gate judges a
finish on is the one the judge starts its scoring scan from.
"""

from __future__ import annotations

import math
from typing import Any


def _structured(res) -> Any:
    """The tool's structured payload from a CallToolResult — unwrapping FastMCP's ``{"result": x}``
    envelope — or ``None`` when there is no structured dict. The one place the finish gate and the
    driver probes share for reading a tool result, so the unwrap rule lives in exactly one spot."""
    sc = getattr(res, "structured_content", None)
    if isinstance(sc, dict):
        return sc.get("result", sc) if set(sc.keys()) == {"result"} else sc
    return None


async def _current_fit(call_tool) -> dict[str, Any]:
    """Fetch the CURRENT HEAD's fit metrics from the server (ground truth, not the agent's
    self-report). ``call_tool`` is an ``await call_tool(name, args)`` handle — the finish gate
    passes the UN-TIMED probe so this internal poll doesn't inflate the tool clock / tool_calls
    metric. Returns {} if the metrics can't be read — the finish check then sees no valid fit and
    fails CLOSED: it refuses ``finish`` rather than letting the run end with no scored model."""
    try:
        data = _structured(await call_tool("get_fit_summary", {}))
        if isinstance(data, dict):
            return data
    except Exception:  # pragma: no cover - never let the check break the run
        pass
    return {}


async def _current_diagnosis(call_tool) -> dict[str, Any]:
    """The current HEAD's fit diagnosis — BOTH sections of ``get_fit_diagnosis``, so the residual
    magnitudes and the parameter geometry (under ``identifiability``) come from ONE call at ONE
    optimum. ``{}`` if unreadable; ``call_tool`` is an ``await call_tool(name, args)`` handle (the
    finish gate passes the un-timed probe so this doesn't inflate the tool clock).

    ``sections`` is passed EXPLICITLY: the tool defaults to the cheap residual half (the agent pays
    for every payload it asks for, in re-sent history), and the parameter half is not optional here —
    it is the FIM the PARSIMONY branch refuses a rank-deficient finish on. The other thing the gate
    reads off this is the worst observable's sign/runs p-values, shown at the ARM step so a finish is
    a considered call on the diagnosis rather than a reflex. FAILS OPEN in both directions — a {}
    here leaves the gate with no grounds to refuse."""
    try:
        data = _structured(await call_tool("get_fit_diagnosis", {"sections": "all"}))
        if isinstance(data, dict):
            return data
    except Exception:  # pragma: no cover - never let the check break the run
        pass
    return {}


async def _model_committed(probe) -> bool:
    """Whether the run has authored a model (``describe_problem.model_authored``).

    The driver uses this to catch a run that ended cleanly having built nothing: the lead
    emitted a no-tool turn and the deepagents graph terminated WITHOUT going through the finish
    gate (which only guards the ``finish`` tool) — so neither the finish gate nor the
    zero-tools relabel (a `task` delegation to the biocurator counts as a tool) sees it, and it
    would otherwise score as an empty k=0 "completed" run. Probes via the un-timed
    ``progress_probe`` so the check doesn't inflate the tool clock. Fails OPEN (returns True)
    whenever it cannot determine the state, so an unavailable/failing probe never nukes a run."""
    if probe is None:
        return True
    try:
        data = _structured(await probe("describe_problem", {}))
        # Only a payload that EXPLICITLY reports model_authored decides the answer; a
        # malformed/partial payload (key absent) falls through to fail-open True, so an
        # unexpected response shape never triggers a spurious no-model resume. NOTE this is
        # `model_authored` (has a MECHANISM been built?), not the old `model_initialized`: a
        # model-less problem is now seeded with a content-free initial model, so "there is a
        # model object" is true from the start and would never flag an empty run.
        if isinstance(data, dict) and "model_authored" in data:
            return bool(data["model_authored"])
    except Exception:  # pragma: no cover - never let the check break the run
        pass
    return True


def best_first(rows) -> list[int]:
    """Every FITTED version in ``list_versions`` rows, best first UNDER THE RUN'S OWN CRITERION.

    Ordered by ``criterion_value`` (identical to ``bic`` under the default BIC), NOT by canonical
    bic: the session builds its best-score checkpoint on the criterion, so "best first" has to be
    the same notion — otherwise, under a non-BIC criterion, the order would start from a model the
    run itself never preferred. ``bic`` is the fallback when ANY version predates the recorded
    criterion, applied to the WHOLE set rather than per-version so one ordering never mixes two
    scales. A version with no finite score was never scored and cannot be scored now, and one the
    session's constraints now break (``violates``) can be neither restored nor the best, so both
    are dropped.
    """
    rows = [v for v in rows if isinstance(v, dict)] if isinstance(rows, list) else []
    rows = [v for v in rows if not v.get("violates")]
    # Rows that could be ranked AT ALL. One carrying neither score is dropped by the loop below
    # either way, so it must not get a vote on which key the rest are sorted by.
    rows = [v for v in rows if "criterion_value" in v or "bic" in v]
    # ONE scale for the WHOLE ordering: `criterion_value` only when every candidate carries it,
    # else canonical `bic` for all of them. Falling back per-version instead would sort a store
    # that spans this upgrade on two scales at once — an early version's BIC against a later
    # one's AICc — leaving the candidate order arbitrary between the two groups. A row whose
    # `criterion_value` is present but NULL still counts as carrying it: that fit is unscoreable
    # under this criterion and is meant to be dropped, not to trigger a fallback.
    key = "criterion_value" if all("criterion_value" in v for v in rows) else "bic"
    scored = []
    for v in rows:
        score, num = v.get(key), v.get("version")
        if (
            isinstance(score, (int, float))
            and not isinstance(score, bool)
            and math.isfinite(score)
            and isinstance(num, int)
        ):
            scored.append((float(score), num))
    scored.sort()
    return [num for _, num in scored]


async def scored_version(ranked: list[int], flagged) -> tuple[int, bool]:
    """The version a run is scored on, chosen from ``ranked`` (fitted versions best first,
    :func:`best_first`): the first whose noise sigma ``await flagged(version)`` does not call
    implausible, with ``True``; the best-scoring one, with ``False``, when every one is flagged.

    The judge's scoring scan (:func:`gemot.runner.judge._restore_best_checkpoint`) and the finish
    gate (:func:`_restore_best`) both choose with this, so the rule is written once. They differ only
    in ``flagged``: the judge also asks the reference model's fitted sigmas, which the gate does not
    hold (frohlich-lab/gemot#135).

    ``flagged`` answers ``True``, ``False``, or ``None`` for unassessable (no stored report, or one no
    band can read). ``None`` stops the scan as ``False`` does: the judge's sigma gate does not fail an
    unassessable version either, so passing it over would score a worse one for nothing.
    """
    for version in ranked:
        if await flagged(version) is not True:
            return version, True
    return ranked[0], False


async def _sigma_flagged(call_tool, version: int) -> "bool | None":
    """Does the server flag ``version``'s ESTIMATED noise sigma as implausible? ``None`` when
    unassessable.

    Read off the version's stored meta (``get_version`` without ``include``), which carries the
    ``noise_sigma_inflated`` / ``noise_sigma_collapsed`` the server computed at that fit: the same
    flags the gate refuses HEAD on through ``get_fit_summary``, so the two cannot disagree, and no
    server predicate has to be imported into this import-light package. Unreadable, or a meta with
    neither flag, is ``None``: like the judge's scan, an unassessable version is not passed over in
    favour of a worse one."""
    try:
        meta = _structured(await call_tool("get_version", {"version": version}))
    except Exception:  # pragma: no cover - never let the check break the run
        return None
    if not isinstance(meta, dict) or meta.get("ok") is False:
        return None
    flags = [meta.get(k) for k in ("noise_sigma_inflated", "noise_sigma_collapsed")]
    if not any(isinstance(f, bool) for f in flags):
        return None
    return any(f is True for f in flags)


async def _restore_best(call_tool) -> "tuple[int, int] | None":
    """Make the version the run is scored on HEAD, and return ``(that version, previous HEAD)``
    when that moved HEAD; ``None`` when HEAD already was it or nothing could say otherwise.

    The finish gate judges the version the run is SCORED on, which is not HEAD wherever the lead
    explored past its best model: lucarelli runs (gemot#94) were refused on a worse HEAD and spent
    fits rebuilding a clean model there ("I need HEAD on a clean model") while the version they
    would have been scored on sat in the store. A restore rather than a read of the stored version
    because every gate read (``get_fit_summary``, ``get_fit_diagnosis``) is of HEAD, and because
    the judge restores the same version before it scores.

    Which version is the judge's scan (:func:`scored_version`) without its reference: the
    best-scoring one (:func:`best_first`) whose noise sigma the server does not flag
    (:func:`_sigma_flagged`), else the plain best-scoring one when every version is flagged. Restoring the best-scoring version regardless would put a
    gamed noise model back over the repaired, slightly worse HEAD the lead had just fitted, refuse
    the finish on its sigma, and undo the repair on every attempt, while the judge would score the
    repaired version anyway. The reference sigmas the judge also checks stay hidden: they would
    tell the run something about the true model.

    Fails OPEN to HEAD: an unreadable store, an unknown HEAD or a refused restore leaves the gate
    judging HEAD, as it did before."""
    try:
        ranked = best_first(_structured(await call_tool("list_versions", {})))
        if not ranked:
            return None
        chosen, _ = await scored_version(ranked, lambda v: _sigma_flagged(call_tool, v))
        desc = _structured(await call_tool("describe_problem", {}))
        head = desc.get("current_version") if isinstance(desc, dict) else None
        if not isinstance(head, int) or head == chosen:
            return None
        restored = _structured(
            await call_tool(
                "restore_version",
                {
                    "version": chosen,
                    "justification": "finish gate: judge the version the run is scored on",
                },
            )
        )
    except Exception:  # pragma: no cover - never let the check break the run
        return None
    if not isinstance(restored, dict) or restored.get("ok") is False or restored.get("error"):
        return None
    return chosen, head
