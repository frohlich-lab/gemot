"""Single source of truth for the fit-time budget the harness + server size their timeouts to.

A single model commit fits the model with a pyPESTO/Fides multistart of ``n_starts`` starts. Three
independent timeouts must each exceed one such fit, or a slow-but-healthy fit is mistaken for a
hang:

  * the server's Fides per-start cap (``fides.Options.MAXTIME``) — bounds ONE optimization start;
  * the MCP streamable-HTTP client's ``sse_read_timeout`` — drops the fit's response if the whole
    fit out-idles it (:mod:`runner`);
  * the harness stall watchdog (``agent._STEP_STALL_TIMEOUT_S``) — declares the run wedged.

Fides defaults ``MAXTIME`` to ``inf``, so today one stiff start runs unbounded, out-idles the 300s
transport, and the 600s stall watchdog then kills an otherwise-healthy run. The fix is to give
Fides a real per-start cap and size the other two off the SAME budget so they cannot drift.

pyPESTO runs the multistart across ``FIT_N_WORKERS`` worker processes when more than one core is
available (the batch runner dispatches examples one at a time, so a fit has the whole box), so the
wall-clock is the number of sequential WAVES, not the start count:

    waves(n_starts)      = ceil(n_starts / fit_workers(n_starts))
    fit_time_budget(...) = FIT_COMPILE_MARGIN_S + waves * FIT_PER_START_TIMEOUT_S

Both the per-start cap and the worker count are shared with the SERVER (``petab_mcp.fitting``,
which applies both) through the ``PETAB_MCP_FIT_PER_START_TIMEOUT`` / ``PETAB_MCP_FIT_N_WORKERS``
env vars — the runner exports its resolved values onto the server subprocess (:mod:`runner`), so
both sides compute the identical budget even when overridden. ``simulate.py``'s reference fit uses
:func:`fides_maxtime_options` + :func:`fit_engine` too, so the reference and candidate optimizers
agree and ΔBIC parity holds.

Kept dependency-light (stdlib + lazy ``fides`` / ``pypesto`` imports) so ``simulate.py`` can keep
regenerating data from a bare amici+petab environment without pulling in the server package.
"""

from __future__ import annotations

import math
import os


def _env_float(name: str, default: float) -> float:
    """A finite, strictly-positive float from ``name``, or ``default`` if the env var is unset,
    non-numeric, or not a finite positive number — so a bad override can never make a timeout NaN/
    Inf/<=0 or crash import."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return default
    return val if math.isfinite(val) and val > 0 else default


def _env_int(name: str, default: int) -> int:
    """A positive int (>=1) from ``name``, or ``default`` for an unset/non-numeric/<1 override."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        val = int(raw)
    except (TypeError, ValueError):
        return default
    return val if val >= 1 else default


def _default_workers() -> int:
    """Leave one core free for the OS / the server's own event loop."""
    return max(1, (os.cpu_count() or 1) - 1)


# Per-start wall-clock cap for a single Fides optimization start, in seconds. Generous by default:
# it should only ever truncate a genuinely stuck start, never a start that is still converging
# (Fides returns its best-so-far for a capped start, so too tight a cap silently degrades fit
# quality / ΔBIC). Override with PETAB_MCP_FIT_PER_START_TIMEOUT.
FIT_PER_START_TIMEOUT_S = _env_float("PETAB_MCP_FIT_PER_START_TIMEOUT", 60.0)
# Slack added on top of the pure fitting time to cover the one-time AMICI/pyPESTO compile on the
# first build of a structure plus simulation/serialization — none of which Fides MAXTIME bounds.
# Override with PETAB_MCP_FIT_COMPILE_MARGIN.
FIT_COMPILE_MARGIN_S = _env_float("PETAB_MCP_FIT_COMPILE_MARGIN", 600.0)
# Max pyPESTO worker processes for the multistart. Default uses the box minus one core; set to 1 to
# force the old sequential behaviour, or lower it when running several example runs concurrently so
# the machine is not oversubscribed. Override with PETAB_MCP_FIT_N_WORKERS.
FIT_N_WORKERS = _env_int("PETAB_MCP_FIT_N_WORKERS", _default_workers())


def fit_workers(n_starts: int) -> int:
    """Effective parallel workers for an ``n_starts`` multistart — never more than there are starts,
    always at least one."""
    return max(1, min(FIT_N_WORKERS, max(1, int(n_starts))))


def fit_time_budget(n_starts: int) -> float:
    """Worst-case wall-clock (seconds) for one full ``n_starts`` multistart fit, incl. the one-time
    compile: the number of sequential worker WAVES times the per-start cap, plus compile slack.
    Callers size their I/O / stall timeouts to this so a within-budget fit never gets dropped or
    flagged as a hang."""
    n = max(1, int(n_starts))
    waves = math.ceil(n / fit_workers(n))
    return FIT_COMPILE_MARGIN_S + waves * FIT_PER_START_TIMEOUT_S


def fides_maxtime_options() -> dict:
    """``{fides.Options.MAXTIME: FIT_PER_START_TIMEOUT_S}`` if this ``fides`` exposes ``MAXTIME``,
    else ``{}`` (older/absent fides just runs uncapped). Pass into ``FidesOptimizer(options=...)``
    to bound each optimization start's wall time."""
    try:
        import fides
        return {fides.Options.MAXTIME: FIT_PER_START_TIMEOUT_S}
    except Exception:  # noqa: BLE001 - no fides / no MAXTIME on this version: run uncapped
        return {}


def fit_engine(n_starts: int):
    """A pyPESTO engine that runs the multistart across ``fit_workers(n_starts)`` worker processes,
    or ``None`` (pyPESTO's sequential ``SingleCoreEngine``) for a single effective worker or when a
    parallel engine can't be constructed. Pass to ``pypesto.optimize.minimize(engine=...)``."""
    workers = fit_workers(n_starts)
    if workers < 2:
        return None
    try:
        from pypesto.engine import MultiProcessEngine
        return MultiProcessEngine(n_procs=workers)
    except Exception:  # noqa: BLE001 - parallel engine unavailable: fall back to sequential
        return None
