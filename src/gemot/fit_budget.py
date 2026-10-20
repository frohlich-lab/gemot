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

pyPESTO runs the multistart across ``FIT_N_WORKERS`` worker processes when more than one core is
available (the batch runner dispatches examples one at a time, so a fit has the whole box), so the
wall-clock is the number of sequential WAVES, not the start count:

    waves(n_starts)      = ceil(n_starts / fit_workers(n_starts))
    fit_time_budget(...) = FIT_COMPILE_MARGIN_S + waves * FIT_PER_START_TIMEOUT_S

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
    AMICI's tolerances what it integrates, the task settings how its starts are batched. Read from
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
    }


def fit_workers(n_starts: int) -> int:
    """Effective parallel workers for an ``n_starts`` multistart — never more than there are starts,
    always at least one, as the server computes it."""
    return max(1, min(FIT_N_WORKERS, max(1, int(n_starts))))


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
