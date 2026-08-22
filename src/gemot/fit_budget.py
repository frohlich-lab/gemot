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
    """Default worker count: the CPUs a batch scheduler granted (``SLURM_CPUS_PER_TASK``) when
    set, else the box minus one core (leaving one free for the OS / the server's event loop).

    ``os.cpu_count()`` reports the whole node and ignores a cgroup/cpuset limit, so under Slurm —
    which pins the process to N cores but leaves ``os.cpu_count()`` reporting the node's full
    count — the box-minus-one default would oversubscribe the allocation *and* desync the wave
    math here from the server's real parallelism. Kept identical to
    ``petab_mcp.fitting._default_fit_workers`` so this budget matches the server's actual worker
    count on and off a scheduler.
    """
    slurm = _env_int("SLURM_CPUS_PER_TASK", 0)
    if slurm >= 1:
        return slurm
    return max(1, (os.cpu_count() or 1) - 1)


# Per-start wall-clock cap for a single Fides optimization start, in seconds. Generous by default:
# it should only ever truncate a genuinely stuck start, never a start that is still converging
# (Fides returns its best-so-far for a capped start, so too tight a cap silently degrades fit
# quality / ΔBIC). Override with PETAB_MCP_FIT_PER_START_TIMEOUT.
FIT_PER_START_TIMEOUT_S = _env_float("PETAB_MCP_FIT_PER_START_TIMEOUT", 180.0)
# Slack added on top of the pure fitting time to cover the one-time AMICI/pyPESTO compile on the
# first build of a structure plus simulation/serialization — none of which Fides MAXTIME bounds.
# Override with PETAB_MCP_FIT_COMPILE_MARGIN.
FIT_COMPILE_MARGIN_S = _env_float("PETAB_MCP_FIT_COMPILE_MARGIN", 600.0)
# Max pyPESTO worker processes for the multistart. Default uses the box minus one core; set to 1 to
# force the old sequential behaviour, or lower it when running several example runs concurrently so
# the machine is not oversubscribed. Override with PETAB_MCP_FIT_N_WORKERS.
FIT_N_WORKERS = _env_int("PETAB_MCP_FIT_N_WORKERS", _default_workers())
# Multiplier applied on top of the raw compile+waves budget below, so the stall watchdog / sse
# transport timeout have real slack over a fit's NOMINAL duration — not size to it exactly. A
# schwen sweep run once stalled at precisely fit_time_budget's value (a healthy 50-start fit that
# used its full per-start allowance on every wave, plus normal MCP/network/serialization
# overhead on top), got misclassified as hung, and cascaded into an unrecovered teardown error.
# Override with PETAB_MCP_FIT_BUDGET_MARGIN.
FIT_BUDGET_MARGIN = _env_float("PETAB_MCP_FIT_BUDGET_MARGIN", 1.4)
# AMICI absolute integration tolerance for EVERY fit in the project. AMICI's stock default is
# 1e-16, which on a stiff model is tight enough that the FORWARD-SENSITIVITY solve cannot make
# progress: the objective VALUE still evaluates fine, but the gradient comes back non-finite, so
# Fides reads +INF at iteration 0 and the start dies. Nothing in a fit's output distinguishes that
# from a genuinely bad start — it is a solver that cannot differentiate there, reported as a start
# that failed.
#
# Measured on lucarelli, whose published optimum sits in exactly such a region. At 1e-16 a
# 5000-start multistart never once reached the committed bar (best 1772.80, 182 BIC adrift) and
# 22% of its starts died NOT_FINITE; at 1e-10, 99 of 500 starts land ON the bar, none die, and the
# objective at the optimum moves by 2e-5. Raising max_steps does nothing: it is the tolerance, not
# the step budget. Override with PETAB_MCP_FIT_AMICI_ATOL.
FIT_AMICI_ATOL = _env_float("PETAB_MCP_FIT_AMICI_ATOL", 1e-10)


def fit_workers(n_starts: int, n_workers: int | None = None) -> int:
    """Effective parallel workers for an ``n_starts`` multistart — never more than there are starts,
    always at least one.

    ``n_workers`` overrides the ``FIT_N_WORKERS`` cap for this call. Pass it when a specific
    caller needs a specific parallelism — e.g. a reference refit that must run SEQUENTIALLY.
    That used to be done by setting ``PETAB_MCP_FIT_N_WORKERS`` at runtime, which silently did
    nothing: ``FIT_N_WORKERS`` is bound at import, and importing an example's ``run.py`` already
    pulls this module in. An argument cannot lose that race."""
    cap = FIT_N_WORKERS if n_workers is None else max(1, int(n_workers))
    return max(1, min(cap, max(1, int(n_starts))))


def fit_time_budget(n_starts: int) -> float:
    """Wall-clock (seconds) the stall watchdog / transport timeout size themselves to for one
    full ``n_starts`` multistart fit: the number of sequential worker WAVES times the per-start
    cap, plus compile slack, times ``FIT_BUDGET_MARGIN`` so a fit that legitimately uses its
    entire nominal budget (every wave hitting its per-start cap) still has headroom over normal
    MCP/network/serialization overhead before being mistaken for a hang. Callers size their I/O /
    stall timeouts to this so a within-budget fit never gets dropped or flagged as a hang."""
    n = max(1, int(n_starts))
    waves = math.ceil(n / fit_workers(n))
    return (FIT_COMPILE_MARGIN_S + waves * FIT_PER_START_TIMEOUT_S) * FIT_BUDGET_MARGIN


def apply_amici_solver_defaults(problem, atol: float | None = None) -> None:
    """Set :data:`FIT_AMICI_ATOL` on ``problem``'s AMICI solver, in place.

    Call this on every pyPESTO problem either side fits, immediately after
    ``PetabImporter.create_problem()`` — reference fits AND candidate fits, or the two optimizers
    stop agreeing and ΔBIC stops being a comparison. It lives here, next to
    :func:`fides_maxtime_options` and :func:`fit_engine`, for that reason: this module is the one
    place the harness and the server both already read their fit settings from.

    ``atol`` overrides the module default for THIS call. Best-effort — a problem whose objective
    exposes no AMICI solver (a non-AMICI objective, or a pyPESTO that renamed the attribute) is
    left exactly as it was rather than raising, since a missing tolerance degrades fit quality but
    a raise here would take down every fit path at once."""
    value = FIT_AMICI_ATOL if atol is None else float(atol)
    solver = getattr(getattr(problem, "objective", None), "amici_solver", None)
    setter = (getattr(solver, "set_absolute_tolerance", None)
              or getattr(solver, "setAbsoluteTolerance", None))
    if setter is not None:
        setter(value)


def fides_maxtime_options(per_start_timeout: float | None = None) -> dict:
    """``{fides.Options.MAXTIME: FIT_PER_START_TIMEOUT_S}`` if this ``fides`` exposes ``MAXTIME``,
    else ``{}`` (older/absent fides just runs uncapped). Pass into ``FidesOptimizer(options=...)``
    to bound each optimization start's wall time.

    ``per_start_timeout`` overrides the module default for THIS call. It exists for the same reason
    :func:`fit_workers` takes ``n_workers``: ``FIT_PER_START_TIMEOUT_S`` is bound at import, and
    importing an example's ``run.py`` already imports this module, so a per-example value could not
    be delivered through ``PETAB_MCP_FIT_PER_START_TIMEOUT`` after the fact. The cap that leaves
    ~80% of a model's starts terminating on their own convergence criterion (rather than being
    truncated at MAXTIME, which makes Fides return best-so-far and silently loosens the bar) spans
    two and a half orders of magnitude across the suite — 60 s on ``perelson`` against 28800 s on
    ``raimundez`` — so one global default cannot serve both without either wasting the cluster or
    committing an under-converged reference. See each ``run.py``'s ``REFIT_PER_START_TIMEOUT``."""
    timeout = FIT_PER_START_TIMEOUT_S if per_start_timeout is None else float(per_start_timeout)
    try:
        import fides
        return {fides.Options.MAXTIME: timeout}
    except Exception:  # noqa: BLE001 - no fides / no MAXTIME on this version: run uncapped
        return {}


def fit_engine(n_starts: int, n_workers: int | None = None):
    """A pyPESTO engine that runs the multistart across ``fit_workers(n_starts, n_workers)`` worker
    processes, or ``None`` (pyPESTO's sequential ``SingleCoreEngine``) for a single effective worker
    or when a parallel engine can't be constructed. Pass to ``pypesto.optimize.minimize(engine=...)``.

    ``n_workers=1`` is how a caller forces the sequential engine — see :func:`fit_workers`."""
    workers = fit_workers(n_starts, n_workers)
    if workers < 2:
        return None
    try:
        from pypesto.engine import MultiProcessEngine
        return MultiProcessEngine(n_procs=workers)
    except Exception:  # noqa: BLE001 - parallel engine unavailable: fall back to sequential
        return None
