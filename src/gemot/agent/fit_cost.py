"""What a fit cost, and how much of the run is left, closing each fit result the lead reads.

The lead used to plan without a clock. No fit result said how long its fit took or how much of the
run remained, and no agent called ``get_run_timing``. On the 2026-11-08 sweep (frohlich-lab/
gemot_iclr_2026 63522750) one lead called a 4.3-h fit "the cheapest arbiter" and another called a
1.3-h refit "the cheap decisive test". On the three problems whose fits take hours, 14 of 15 runs
left their analysts idle while a fit ran (gemot#137).

So every fit result ends with one line, :data:`gemot.prompts.FIT_COST_TMPL`. Both of its numbers
are on the driver's budget clock (``run_deepagents``), which leaves out the time a fit spent queued
for the shared fit pool. A fit's cost is what it took from the budget the run stops on, not its
wall time.

The line is added on the client, around the tool object (like :mod:`.circuit_breaker`), because the
budget is the harness's. The server times its own fits, but it does not know the run's wall budget
or how much of it is left.

Only a result whose fit stage ran gets the line. The server sets ``committed`` once that stage has
run (``petab_mcp.operations.bundle_result``), and sets ``no_change`` on the de-dup, which skips the
fit. A refusal before the fit comes back in seconds, and its cost is not worth a line.
"""

from __future__ import annotations

import inspect
import json
import math
from collections.abc import Callable
from functools import wraps
from typing import Any

from .. import prompts
from .circuit_breaker import _content_text


def _span(seconds: float) -> str:
    """``seconds`` as the lead reads a duration: minutes under an hour, hours to a tenth above."""
    if seconds < 60:
        return "under a minute"
    if seconds < 3600:
        return f"{round(seconds / 60)} min"
    return f"{seconds / 3600:.1f} h"


def _fit_ran(content: Any, artifact: Any) -> bool:
    """Whether a ``make_changes`` result is one whose fit stage ran (see the module docstring)."""
    payload = None
    if isinstance(artifact, dict) and isinstance(artifact.get("structured_content"), dict):
        payload = artifact["structured_content"]
    else:
        try:
            payload = json.loads(_content_text(content))
        except Exception:
            return False
    return isinstance(payload, dict) and "committed" in payload and not payload.get("no_change")


def fit_cost_note(took: float, left: float) -> str:
    """The line closing a fit result: ``took`` and ``left`` are seconds of the run's budget, and
    ``left`` is ``inf`` for a run without one."""
    if math.isinf(left):
        return prompts.FIT_COST_UNBUDGETED_TMPL.format(took=_span(took))
    return prompts.FIT_COST_TMPL.format(took=_span(took), left=_span(max(left, 0.0)))


def _wrap_fit_cost(
    tools: list,
    fit_tools: frozenset[str],
    *,
    spent: Callable[[], float],
    left: Callable[[], float],
) -> list:
    """Close each fit result of the ``fit_tools`` among ``tools`` with :func:`fit_cost_note`.

    Both callables read the run's budget clock: ``spent`` its seconds so far, and ``left`` those
    left (``inf`` without a budget). Mutates the tool objects in place, like the breaker, and must
    wrap OUTSIDE it, so the line closes whatever the breaker returns. Best-effort: a result the
    line cannot be added to is returned as it came.
    """
    for tool in tools:
        name = getattr(tool, "name", None)
        orig = getattr(tool, "coroutine", None)
        if name not in fit_tools or orig is None:
            continue

        def _make(orig):
            @wraps(orig)
            async def _cost(**arguments):
                began = spent()
                result = await orig(**arguments)
                try:
                    content, artifact = result if isinstance(result, tuple) else (result, None)
                    if not _fit_ran(content, artifact):
                        return result
                    note = fit_cost_note(spent() - began, left())
                    if isinstance(content, list):
                        content = [*content, {"type": "text", "text": note}]
                    elif isinstance(content, str):
                        content = f"{content}\n\n{note}"
                    else:
                        return result
                    return (content, artifact) if isinstance(result, tuple) else content
                except Exception:  # pragma: no cover - the line must never cost the fit's result
                    return result

            try:  # keep the original signature: StructuredTool reads it for `callbacks` and `config`
                _cost.__signature__ = inspect.signature(orig)
            except (ValueError, TypeError):  # pragma: no cover - defensive
                pass
            return _cost

        tool.coroutine = _make(orig)
    return tools
