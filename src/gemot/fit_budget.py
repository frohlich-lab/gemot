"""Single source of truth for the fit-time budget the harness + server size their timeouts to.

A single model commit fits the model with a pyPESTO/Fides multistart of ``n_starts`` starts. Three
independent timeouts must each exceed one such fit, or a slow-but-healthy fit is mistaken for a
hang:

  * the server's Fides per-start cap (``fides.Options.MAXTIME``) — bounds ONE optimization start;
  * the MCP client's per-call ``read_timeout_seconds`` — drops the fit's response if the whole fit
    out-idles it (``runner.clock._wrap_call_timeout``; the session-wide ``sse_read_timeout`` above
    it is deliberately non-binding, since it cannot see a per-call ``n_starts``);
  * the harness stall watchdog (``agent._STEP_STALL_TIMEOUT_S``) — declares the run wedged.

The last two are sized off the ``n_starts`` the call/step ACTUALLY asked for, not the run default
(:func:`fit_time_budget_for`, which both share): ``make_changes`` takes an ``n_starts`` that
REPLACES the default, and ``build_true_model`` floors it higher still. A timeout sized to the run
default — or to any fixed multiple of it — either kills a healthy fit or leaves a wedged step
undetected for longer than it should.

Fides defaults ``MAXTIME`` to ``inf``, so one stiff start would run unbounded, out-idle the 300s
transport default, and the stall watchdog would then kill an otherwise-healthy run. The fix
is to give Fides a real per-start cap and derive the other two from this ONE function so they
cannot drift — same formula, each evaluated at its own ``n_starts``.

pyPESTO runs the multistart across the server's ``FIT_N_WORKERS`` worker processes when more than
one core is available (the batch runner dispatches examples one at a time, so a fit has the whole
box), so the wall-clock is the number of sequential WAVES, not the start count:

    waves(n_starts)      = ceil(n_starts / fit_workers(n_starts))
    fit_time_budget(...) = FIT_COMPILE_MARGIN_S + waves * <the server's FIT_PER_START_TIMEOUT_S>

The per-start cap, the worker count and the AMICI tolerance are the SERVER's
(``petab_mcp.fitting``, which applies them to every candidate fit), read through
:mod:`gemot.server_api` each time a budget is computed: ONE definition, so the harness's timeouts
are sized to the cap and parallelism the server really uses. The runner still exports the resolved
cap and worker count onto the server subprocess (:mod:`gemot.runner.main`), so a subprocess that
would resolve its defaults differently cannot drift from them. ``simulate.py``'s reference fit goes
through :func:`fides_maxtime_options`, :func:`fit_engine` and :func:`apply_amici_solver_defaults`
too, so the reference and candidate optimizers agree and ΔBIC parity holds.

The harness's OWN are the knobs built on top: the compile margin, the budget margin, the ceilings
and the watchdog margin below. Importing this module stays light (stdlib only; ``fides``,
``pypesto`` and the server load on first use).
"""

from __future__ import annotations

import math
import os

from . import server_api


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


# Slack added on top of the pure fitting time to cover the one-time AMICI/pyPESTO compile on the
# first build of a structure plus simulation/serialization — none of which Fides MAXTIME bounds.
# Override with PETAB_MCP_FIT_COMPILE_MARGIN.
FIT_COMPILE_MARGIN_S = _env_float("PETAB_MCP_FIT_COMPILE_MARGIN", 600.0)
# Multiplier applied on top of the raw compile+waves budget below, so the stall watchdog / sse
# transport timeout have real slack over a fit's NOMINAL duration — not size to it exactly. A
# schwen sweep run once stalled at precisely fit_time_budget's value (a healthy 50-start fit that
# used its full per-start allowance on every wave, plus normal MCP/network/serialization
# overhead on top), got misclassified as hung, and cascaded into an unrecovered teardown error.
# Override with PETAB_MCP_FIT_BUDGET_MARGIN.
FIT_BUDGET_MARGIN = _env_float("PETAB_MCP_FIT_BUDGET_MARGIN", 1.4)


def fit_workers(n_starts: int, n_workers: int | None = None) -> int:
    """Effective parallel workers for an ``n_starts`` multistart — never more than there are starts,
    always at least one.

    ``n_workers`` overrides the server's ``FIT_N_WORKERS`` cap for this call. Pass it when a
    specific caller needs a specific parallelism — e.g. a reference refit that must run
    SEQUENTIALLY. That used to be done by setting ``PETAB_MCP_FIT_N_WORKERS`` at runtime, which
    silently did nothing: ``FIT_N_WORKERS`` is bound when its module is first imported, which an
    example's ``run.py`` has already done. An argument cannot lose that race."""
    cap = server_api.fitting().FIT_N_WORKERS if n_workers is None else max(1, int(n_workers))
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
    per_start = server_api.fitting().FIT_PER_START_TIMEOUT_S
    return (FIT_COMPILE_MARGIN_S + waves * per_start) * FIT_BUDGET_MARGIN


#: Hard upper bound on any single derived fit budget. `n_starts` is chosen by the MODEL and the
#: server accepts any int >= 1 (`_coerce_n_starts`), so a hallucinated `n_starts=1000000` would
#: otherwise compute a ~115-DAY timeout and switch off both the stall watchdog and the transport
#: cut for that call. This clamps the derived TIMEOUT, which is not the same as capping n_starts:
#: a server-side cap on the count itself was tried and reverted (any cap near the run default
#: refuses scenario.build_true_model's bruno-400 / fujita-300 reference builds). The fit still
#: runs at whatever it asked for; it just cannot buy itself an unbounded window.
FIT_BUDGET_CEILING_S = _env_float("PETAB_MCP_FIT_BUDGET_CEILING", 24 * 3600.0)

#: Upper bound on the START COUNT any derived budget is computed from. Distinct from the seconds
#: ceiling above and NOT redundant with it: the seconds clamp is applied to the RESULT, so
#: without this the formula still evaluates at the raw count first -- and `n_starts=10**400`
#: (an int the model can emit, and `_pending_fit_starts` will happily read) overflows
#: `waves * FIT_PER_START_TIMEOUT_S` to an OverflowError that ends the run. Ten million is orders
#: of magnitude above any real multistart, so it never touches a legitimate count.
_MAX_DERIVABLE_N_STARTS = 10_000_000

#: Margin the STALL WATCHDOG sits above the per-call transport cut for the same fit.
#:
#: The three timeouts have to be strictly ordered, not merely "all big enough": httpx ceiling >
#: watchdog > per-call cut. Ties are not safe. When the watchdog and the transport cut are set to
#: the same number the watchdog ALWAYS wins -- its `asyncio.wait_for` is armed a graph-dispatch
#: earlier than the call's `anyio.fail_after` -- so an overrunning fit is cancelled mid-flight as
#: "stalled" and the clean `McpError(408)` path (which resumes from the checkpoint WITHOUT the
#: model being told to re-fit) is unreachable. This margin makes the transport cut provably the
#: first to fire, so a slow fit degrades the gentle way.
FIT_WATCHDOG_MARGIN = _env_float("PETAB_MCP_FIT_WATCHDOG_MARGIN", 1.15)


def fit_time_budget_for(asked, default: int) -> float:
    """:func:`fit_time_budget` for a call/step that ASKED for ``asked`` starts, or ``default``.

    ONE implementation, deliberately: the stall watchdog (``agent.driver``) and the per-call
    transport cut (``runner.clock``) must agree about how long a given fit is allowed to take, or
    they stop firing at comparable times and whichever is accidentally tighter becomes the real
    limit. They read ``n_starts`` from different places — a pending ``tool_calls`` entry and a
    live ``call_tool`` argument — but the rule turning that number into seconds belongs here,
    next to the formula it feeds.

    ``asked`` is untrusted: it comes from the model. Coerced with a bare ``int()`` to match the
    server's own ``_coerce_n_starts``, so a model sending ``"300"`` as a string gets a window for
    the 300-start fit it will actually run; anything ``int()`` refuses — including ``None``,
    ``bool`` (which ``int()`` would silently read as 0/1) and ``float('inf')`` (which raises
    OverflowError) — falls back to ``default``.

    Bounds, in the order they are applied:

    * the COUNT is capped at :data:`_MAX_DERIVABLE_N_STARTS` BEFORE the formula runs, so an absurd
      value cannot overflow the arithmetic instead of being clamped;
    * the result is capped at :data:`FIT_BUDGET_CEILING_S`;
    * and then floored at ``fit_time_budget(default)``. The floor is applied LAST and therefore
      wins: a call asking for FEWER starts than the default makes a shorter fit, but the default
      budget is the floor already trusted and shrinking a timeout buys nothing. This also means a
      run whose own default budget exceeds the ceiling is not silently cut below it — the ceiling
      bounds what the MODEL can ask for, never what the operator configured.
    """
    default_n = max(1, min(int(default), _MAX_DERIVABLE_N_STARTS))
    floor = fit_time_budget(default_n)
    n = default_n
    if asked is not None and not isinstance(asked, bool):
        try:
            n = max(n, min(int(asked), _MAX_DERIVABLE_N_STARTS))
        except (TypeError, ValueError, OverflowError):
            pass
    return max(floor, min(fit_time_budget(n), FIT_BUDGET_CEILING_S))


def apply_amici_solver_defaults(problem, atol: float | None = None) -> None:
    """Set the server's ``FIT_AMICI_ATOL`` on ``problem``'s AMICI solver, in place.

    Call this on every pyPESTO problem either side fits, immediately after
    ``PetabImporter.create_problem()`` — reference fits AND candidate fits, or the two optimizers
    stop agreeing and ΔBIC stops being a comparison. It lives here, next to
    :func:`fides_maxtime_options` and :func:`fit_engine`, for that reason: this module is where the
    harness reads the server's fit settings, and AMICI's stock 1e-16 makes a stiff model's
    sensitivities non-finite (the server's ``fitting.py`` gives the measurement).

    ``atol`` overrides the server's default for THIS call. Best-effort — a problem whose objective
    exposes no AMICI solver (a non-AMICI objective, or a pyPESTO that renamed the attribute) is
    left exactly as it was rather than raising, since a missing tolerance degrades fit quality but
    a raise here would take down every fit path at once."""
    value = server_api.fitting().FIT_AMICI_ATOL if atol is None else float(atol)
    solver = getattr(getattr(problem, "objective", None), "amici_solver", None)
    setter = getattr(solver, "set_absolute_tolerance", None) or getattr(
        solver, "setAbsoluteTolerance", None
    )
    if setter is not None:
        setter(value)


def fides_maxtime_options(per_start_timeout: float | None = None) -> dict:
    """``{fides.Options.MAXTIME: <the server's FIT_PER_START_TIMEOUT_S>}`` if this ``fides`` exposes
    ``MAXTIME``,
    else ``{}`` (older/absent fides just runs uncapped). Pass into ``FidesOptimizer(options=...)``
    to bound each optimization start's wall time.

    ``per_start_timeout`` overrides the server's default for THIS call. It exists for the same
    reason :func:`fit_workers` takes ``n_workers``: ``FIT_PER_START_TIMEOUT_S`` is bound when its
    module is first imported, so a per-example value could not be delivered through
    ``PETAB_MCP_FIT_PER_START_TIMEOUT`` after the fact. The cap that leaves
    ~80% of a model's starts terminating on their own convergence criterion (rather than being
    truncated at MAXTIME, which makes Fides return best-so-far and silently loosens the bar) spans
    two and a half orders of magnitude across the suite — 60 s on ``perelson`` against 28800 s on
    ``raimundez`` — so one global default cannot serve both without either wasting the cluster or
    committing an under-converged reference. See each ``run.py``'s ``REFIT_PER_START_TIMEOUT``."""
    timeout = (
        server_api.fitting().FIT_PER_START_TIMEOUT_S
        if per_start_timeout is None
        else float(per_start_timeout)
    )
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
