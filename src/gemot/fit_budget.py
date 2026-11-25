"""Single source of truth for the fit-time budget the harness + server size their timeouts to.

A single model commit fits the model with a pyPESTO/Fides multistart of ``n_starts`` starts. Two
timeouts must each exceed one such fit, or a slow-but-healthy fit is mistaken for a hang:

  * the server's Fides per-start cap (``fides.Options.MAXTIME``) — bounds ONE optimization start;
  * the harness stall watchdog's floor (``agent.driver._STEP_STALL_TIMEOUT_S``) — declares the
    run wedged.

The watchdog is sized off the ``n_starts`` the step ACTUALLY asked for, not the run default
(:func:`fit_time_budget_for`): ``make_changes`` takes an ``n_starts`` that REPLACES the default,
and ``build_true_model`` floors it higher still. A timeout sized to the run default — or to any
fixed multiple of it — either kills a healthy fit or leaves a wedged step undetected for longer
than it should.

A tool call is NOT sized here. On a shared fit pool a fit's wall time includes its wait behind
other runs' fits, which no formula bounds, so a call lives on the server's heartbeats instead
(``runner.clock._wrap_call_timeout``, gemot#106), and they push the watchdog out too.

Fides defaults ``MAXTIME`` to ``inf``, so one stiff start would run unbounded and the stall
watchdog would then kill an otherwise-healthy run. The fix is to give Fides a real per-start cap
and derive the watchdog from the same formula.

pyPESTO runs the multistart across ``FIT_N_WORKERS`` worker processes when more than one core is
available (the batch runner dispatches examples one at a time, so a fit has the whole box), so the
wall-clock is the number of sequential WAVES, not the start count:

    waves(n_starts)      = ceil(n_starts / fit_workers(n_starts))
    fit_time_budget(...) = FIT_COMPILE_MARGIN_S + waves * FIT_PER_START_TIMEOUT_S

On a sweep's fit pool (``PETAB_MCP_FIT_POOL``) the starts do not run on the job's own cores: the
server sends them to the pool's workers and spreads a fit over the idle ones (PEtab-MCP#254), so a
fit takes about one wave. Waves counted over the job's ``FIT_N_WORKERS`` then overstate it by up to
the pool's size over the job's CPUs: alkan's 100 starts on 8 CPUs came to 13 waves of 2,400 s, and
its pooled fits took 2,400-3,000 s (gemot#117). The timeouts keep that generous figure, since each
live heartbeat moves them out anyway. Only the driver's last-step reserve counts one wave on a pool
(``pool=True``): it is the one place where overstating a fit costs the run, which it stops early.

Both the per-start cap and the worker count are shared with the SERVER (``petab_mcp.fitting``,
which applies both) through the ``PETAB_MCP_FIT_PER_START_TIMEOUT`` / ``PETAB_MCP_FIT_N_WORKERS``
env vars — the runner exports its resolved values onto the server subprocess (:mod:`runner`), so
both sides compute the identical budget even when overridden. The values are restated here
rather than imported because this module must stay import-light — the agent driver reads it, and
importing any ``petab_mcp`` module pulls in petab and pandas (tests/test_import_lightness.py) — so
tests/test_import_lightness.py also pins the two copies together. The reference fit, which needs
the fit settings themselves rather than the budget, lives in PEtabAgentBench and takes them from
the server directly.

The server reads more fit settings from its environment than these two. :func:`server_fit_settings`
resolves every one of them as the server does, so a run can record what its fits ran under
(``fit_settings`` in result.json); the same test pins its names and defaults to the server's.

Kept dependency-light (stdlib only).
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping


def _env_float(
    name: str, default: float | None, env: Mapping[str, str] | None = None
) -> float | None:
    """A finite, strictly-positive float from ``name``, or ``default`` if the env var is unset,
    non-numeric, or not a finite positive number — so a bad override can never make a timeout NaN/
    Inf/<=0 or crash import. Read from ``env``, else this process's environment."""
    raw = (os.environ if env is None else env).get(name)
    if raw is None:
        return default
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return default
    return val if math.isfinite(val) and val > 0 else default


def _env_int(name: str, default: int, env: Mapping[str, str] | None = None) -> int:
    """A positive int (>=1) from ``name``, or ``default`` for an unset/non-numeric/<1 override.
    Read from ``env``, else this process's environment."""
    raw = (os.environ if env is None else env).get(name)
    if raw is None:
        return default
    try:
        val = int(raw)
    except (TypeError, ValueError):
        return default
    return val if val >= 1 else default


def _default_workers(env: Mapping[str, str] | None = None) -> int:
    """Default worker count: the CPUs a batch scheduler granted (``SLURM_CPUS_PER_TASK``) when
    set, else the box minus one core (leaving one free for the OS / the server's event loop).

    ``os.cpu_count()`` reports the whole node and ignores a cgroup/cpuset limit, so under Slurm —
    which pins the process to N cores but leaves ``os.cpu_count()`` reporting the node's full
    count — the box-minus-one default would oversubscribe the allocation *and* desync the wave
    math here from the server's real parallelism. Kept identical to
    ``petab_mcp.fitting._default_fit_workers`` so this budget matches the server's actual worker
    count on and off a scheduler.
    """
    slurm = _env_int("SLURM_CPUS_PER_TASK", 0, env)
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


def server_fit_settings(env: Mapping[str, str]) -> dict[str, float | int | None]:
    """Every fit setting a petab-mcp server launched with ``env`` reads from it, keyed by its
    environment variable, as the server resolves it: its value, not the string it was handed, with
    the server's default where ``env`` sets none or an unusable one. ``None`` for an AMICI
    tolerance leaves AMICI's own default, that of the AMICI the run records in ``dependency_code``.

    The two above size the budget; the rest change what a fit searches without sizing any timeout:
    AMICI's tolerances what it integrates, the task settings how its starts are batched, and AMICI's
    thread count how many of an objective's experiments it simulates at once. Read from
    ``env`` rather than from ``petab_mcp.fitting``, which resolves once from its own process's
    environment, while the run launches its server with another (``runner.server``).
    """
    return {
        "PETAB_MCP_FIT_PER_START_TIMEOUT": _env_float(
            "PETAB_MCP_FIT_PER_START_TIMEOUT", 180.0, env
        ),
        "PETAB_MCP_FIT_N_WORKERS": _env_int("PETAB_MCP_FIT_N_WORKERS", _default_workers(env), env),
        "PETAB_MCP_FIT_STARTS_PER_TASK": _env_int("PETAB_MCP_FIT_STARTS_PER_TASK", 0, env),
        "PETAB_MCP_FIT_TASK_SECONDS": _env_float("PETAB_MCP_FIT_TASK_SECONDS", 270.0, env),
        "PETAB_MCP_FIT_AMICI_ATOL": _env_float("PETAB_MCP_FIT_AMICI_ATOL", 1e-10, env),
        "PETAB_MCP_FIT_AMICI_RTOL": _env_float("PETAB_MCP_FIT_AMICI_RTOL", None, env),
        "PETAB_MCP_FIT_AMICI_ATOL_STEADY_STATE": _env_float(
            "PETAB_MCP_FIT_AMICI_ATOL_STEADY_STATE", None, env
        ),
        "PETAB_MCP_FIT_AMICI_RTOL_STEADY_STATE": _env_float(
            "PETAB_MCP_FIT_AMICI_RTOL_STEADY_STATE", None, env
        ),
        "PETAB_MCP_FIT_AMICI_N_THREADS": _env_int("PETAB_MCP_FIT_AMICI_N_THREADS", 1, env),
    }


def fit_workers(n_starts: int) -> int:
    """Effective parallel workers for an ``n_starts`` multistart — never more than there are starts,
    always at least one, as the server computes it."""
    return max(1, min(FIT_N_WORKERS, max(1, int(n_starts))))


def fit_time_budget(n_starts: int, *, pool: bool = False) -> float:
    """Wall-clock (seconds) the stall watchdog sizes itself to for one full ``n_starts``
    multistart fit: the number of sequential worker WAVES times the per-start
    cap, plus compile slack, times ``FIT_BUDGET_MARGIN`` so a fit that legitimately uses its
    entire nominal budget (every wave hitting its per-start cap) still has headroom over normal
    MCP/network/serialization overhead before being mistaken for a hang. Callers size their stall
    timeouts to this so a within-budget fit is never flagged as a hang.

    ``pool`` counts one wave, as a fit on a sweep's fit pool takes (see the module docstring). Only
    the driver's last-step reserve passes it; a timeout keeps the job's waves."""
    n = max(1, int(n_starts))
    waves = 1 if pool else math.ceil(n / fit_workers(n))
    return (FIT_COMPILE_MARGIN_S + waves * FIT_PER_START_TIMEOUT_S) * FIT_BUDGET_MARGIN


#: Hard upper bound on any single derived fit budget. `n_starts` is chosen by the MODEL and the
#: server accepts any int >= 1 (`_coerce_n_starts`), so a hallucinated `n_starts=1000000` would
#: otherwise compute a ~115-DAY timeout and switch off the stall watchdog for that step. This clamps the derived TIMEOUT, which is not the same as capping n_starts:
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

#: Margin the STALL WATCHDOG sits above a fit-sized timeout inside the same step: the barebones
#: arm's sandbox execute timeout (``agent.barebones``).
#:
#: Ties are not safe. When the watchdog and the timeout inside it are set to the same number the
#: watchdog ALWAYS wins -- it is armed a graph-dispatch earlier -- so an overrunning command is
#: cancelled mid-flight as "stalled" and its own, gentler timeout path is unreachable. The panel's
#: floor carries the same margin, though its tool calls no longer sit inside it: they are cut on
#: the server's heartbeats (``runner.clock.CALL_IDLE_WINDOW_S``), far inside any watchdog window.
FIT_WATCHDOG_MARGIN = _env_float("PETAB_MCP_FIT_WATCHDOG_MARGIN", 1.15)


def fit_time_budget_for(asked, default: int) -> float:
    """:func:`fit_time_budget` for a step that ASKED for ``asked`` starts, or ``default``: the
    stall watchdog's floor (``agent.driver``), read off the step's pending ``tool_calls`` entry,
    and the session-wide fallback timeout ``runner.main`` sets.

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
