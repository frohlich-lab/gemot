"""Reading the server's CURRENT state through a bare call handle.

The finish gate and the driver's run loop both need to ask the server where things stand — is
there a fitted model, what is its bic, what do the residuals look like — without going through
the agent. They share these four helpers so the payload-unwrapping and fail-open rules live in
one place.
"""

from __future__ import annotations

from typing import Any

def _structured(res) -> Any:
    """The tool's structured payload from a CallToolResult — unwrapping FastMCP's ``{"result": x}``
    envelope — or ``None`` when there is no structured dict. The one place the finish gate and the
    driver probes share for reading a tool result, so the unwrap rule lives in exactly one spot."""
    sc = getattr(res, "structuredContent", None)
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
    """The current HEAD's residual diagnosis (``get_fit_diagnosis``), or {} if unreadable.
    ``call_tool`` is an ``await call_tool(name, args)`` handle (the finish gate passes the
    un-timed probe so this doesn't inflate the tool clock).

    Used to put the fit's residual-structure verdict in front of the lead at the finish
    decision, so a finish is a considered call on the diagnosis rather than a reflex."""
    try:
        data = _structured(await call_tool("get_fit_diagnosis", {}))
        if isinstance(data, dict):
            return data
    except Exception:  # pragma: no cover - never let the check break the run
        pass
    return {}


async def _model_committed(probe) -> bool:
    """Whether the run has authored a model (``describe_problem.model_authored``).

    The driver uses this to catch a run that ended cleanly having built nothing: the lead
    emitted a no-tool turn and the deepagents graph terminated WITHOUT going through the finish
    gate (which only guards ``make_decision('finish')``) — so neither the finish gate nor the
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
