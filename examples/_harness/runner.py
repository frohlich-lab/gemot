#!/usr/bin/env python3
"""Scenario-agnostic runner for the model-repair demos.

Given a :class:`~scenario.Scenario`, the pipeline is always the same:

  (a) clean the example directory (recreate ``work/`` from the pristine, broken
      ``problem/`` copy — this also discards any prior provenance; the SBML-hash-keyed
      AMICI build cache is kept OUTSIDE ``work/`` and reused across runs),
  (b) start the ``petab-mcp`` server on ``work/problem.yaml`` (HTTP transport, one
      long-lived stateful process shared by the whole run), and
  (c) drive the chosen driver — ``deepagents`` (an OpenRouter-hosted model with the
      three-analyst panel from :mod:`agent`) or ``scripted`` (the scenario's canonical
      repair, no LLM) — then judge the result INDEPENDENTLY via ``scenario.check``.

HTTP (not stdio) is used because AMICI's C++ build toolchain writes to the server's
stdout while compiling, which would corrupt a stdio JSON-RPC channel; that stdout is
captured to ``work/amici_build.log`` while uvicorn/Python logging goes to
``work/server.log``.

An example's ``run.py`` is a thin wrapper: build the ``Scenario`` and call
:func:`run_scenario`.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import datetime
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from .scenario import Baseline, JudgeContext, Scenario


# --------------------------------------------------------------------------- MCP glue
def _unwrap(result):
    """Turn a CallToolResult into the tool's Python return value."""
    sc = getattr(result, "structuredContent", None)
    if sc is not None:
        # FastMCP wraps non-dict returns as {"result": <value>}.
        if isinstance(sc, dict) and set(sc.keys()) == {"result"}:
            return sc["result"]
        return sc
    texts = []
    for block in getattr(result, "content", None) or []:
        text = getattr(block, "text", None)
        if text is not None:
            texts.append(text)
    joined = "\n".join(texts)
    try:
        return json.loads(joined)
    except Exception:
        return joined


def _make_call(session: ClientSession):
    """Return an ``await call(tool, **args)`` bound to ``session`` (unwraps the result)."""

    async def call(tool: str, /, **args):
        result = await session.call_tool(tool, args or {})
        return _unwrap(result)

    return call


class _ToolClock:
    """Client-observed wall-time during which at least one MCP tool call is in flight.

    This is time *as the runner sees it* — it includes the HTTP round-trip and any
    server-side queueing, not purely server execution — but for these demos it is
    dominated by the work inside the tool calls (AMICI compile + pyPESTO fits + SQL),
    so subtracting it from the wall clock is a good proxy for isolating LLM time.


    Tracks the UNION of busy intervals, not the sum of per-call durations: the LangGraph
    agent (deepagents' ToolNode) dispatches the tool calls in a single turn concurrently, so
    several ``session.call_tool`` coroutines overlap. Summing each one's duration double-counts the
    overlap and can exceed the wall clock (a fit + a SQL query running at once counts twice).
    A busy-interval measure instead answers "how much wall time was the server working," which
    is what we subtract from the wall clock to isolate LLM/inference time, and it can never
    exceed the wall clock.
    """

    __slots__ = ("seconds", "calls", "_active", "_since")

    def __init__(self) -> None:
        self.seconds = 0.0
        self.calls = 0
        self._active = 0  # in-flight call count
        self._since = 0.0  # monotonic time the busy window (0 -> 1) opened

    def enter(self) -> None:
        if self._active == 0:
            self._since = time.monotonic()
        self._active += 1
        self.calls += 1

    def leave(self) -> None:
        self._active -= 1
        if self._active == 0:
            self.seconds += time.monotonic() - self._since


def _wrap_tool_clock(session: ClientSession) -> _ToolClock:
    """Time every ``session.call_tool`` into a returned clock (best-effort).

    Both the runner's own calls and the deepagents panel's calls route through
    ``session.call_tool`` (langchain-mcp-adapters invokes the bound session), so shadowing
    the instance method captures ALL tool time — which for these demos is dominated by the
    AMICI compile + every pyPESTO fit + the SQL queries. Subtracting it from the wall clock
    isolates the LLM/inference time. If the method can't be shadowed the clock simply stays
    at zero (timing is reported as unavailable) rather than breaking the run.

    asyncio is single-threaded and cooperative, so the ``enter``/``leave`` counter updates
    (no ``await`` between reading and mutating ``_active``) are atomic w.r.t. each other; the
    overlap is only ever inside the awaited ``original(...)``.
    """
    clock = _ToolClock()
    original = session.call_tool

    async def timed(*args, **kwargs):
        clock.enter()
        try:
            return await original(*args, **kwargs)
        finally:
            clock.leave()

    with contextlib.suppress(Exception):
        session.call_tool = timed  # type: ignore[method-assign]
    return clock


# --------------------------------------------------------------------------- provenance
def _now_iso() -> str:
    """Current UTC time as an ISO-8601 string (for the metrics history + trace join)."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _git_sha(path: Path) -> str | None:
    """Short HEAD sha (+``+dirty`` if the tree has uncommitted changes), or ``None``.

    Provenance for the metrics history so a run is attributable to a commit; best-effort so
    a detached/absent git checkout never breaks the run.
    """
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(path),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return None
    try:
        dirty = subprocess.call(
            ["git", "diff", "--quiet", "--ignore-submodules"], cwd=str(path),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except Exception:
        dirty = 0
    return f"{sha}+dirty" if dirty else sha


# --------------------------------------------------------------------------- steps
def cleanup(scenario: Scenario) -> Path:
    """Recreate work/ from the pristine broken problem (also drops old provenance)."""
    work = scenario.example_dir / "work"
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(scenario.problem_dir, work)
    print(f"[cleanup] fresh working copy at {work.relative_to(scenario.example_dir.parent)}")
    return work


async def _judge(scenario: Scenario, call, baseline: Baseline, work: Path, meta: dict,
                 clock: "_ToolClock", t0: float) -> "tuple[int, dict]":
    """Judge the repaired model against the true model by BIC (structure-agnostic).

    Success ⇔ the repaired model can't be rejected relative to the true (data-generating)
    model: ``BIC(candidate) - BIC(true) <= scenario.bic_threshold``. We don't require any
    particular rate-law FORM (models need not be structurally identifiable) and we don't
    special-case invented species — BIC's ``k·ln(n)`` penalty makes an extra parameter earn
    its keep. A χ² goodness-of-fit p-value (residuals vs the known noise) is reported
    alongside. The scenario's optional ``diagnostics`` hook adds report-only detail.

    Returns ``(exit_code, result)``: the process exit code (0 on success, 1 otherwise) and
    the machine-readable result dict (also written to ``work/result.json``), so the caller
    can stamp the outcome onto the run span.
    """
    from .scenario import bic, chi2_gof_pvalue

    structure = await call("get_model_structure")
    reactions = structure.get("reactions", []) or []
    current_species = set(structure.get("species", []) or [])
    new_species = sorted(current_species - baseline.species)

    final = await call("get_fit_summary")
    final_nll = float(final["neg_log_likelihood"])
    n = int(final.get("n_data_points") or 0)
    # k = number of ESTIMATED parameters (estimate=1), NOT len(fitted_parameters): the latter
    # also counts fixed params (noise σ, and the reconcile-added parametric initial
    # conditions), which would inflate the penalty and diverge from how reference.json counts
    # k. Both sides must count free parameters the same way.
    params = await call("get_parameters")
    k = sum(1 for p in params if p.get("estimate")) if isinstance(params, list) else 0
    chi2 = final.get("chi2")
    chi2 = float(chi2) if chi2 is not None else None
    head = int((await call("current_version")).get("version", 0))
    diff = await call("diff_versions", from_version=0, to_version=head) if head else {}
    errors = await call("get_errors", limit=20)
    delta = baseline.nll - final_nll

    # BIC gate against the true-model reference.
    ref = scenario.reference()
    bic_cand = bic(final_nll, k, n) if n else None
    bic_ref = bic(ref["nll"], ref["k"], ref["n"]) if ref else None
    delta_bic = (bic_cand - bic_ref) if (bic_cand is not None and bic_ref is not None) else None
    if delta_bic is not None:
        success = delta_bic <= scenario.bic_threshold
    else:  # no reference / no data points — can't apply the gate
        success = False
    # χ² goodness-of-fit vs the known noise (reported, not gating).
    dof = max(0, n - k)
    p_gof = chi2_gof_pvalue(chi2, dof)

    ctx = JudgeContext(call=call, baseline=baseline, final_nll=final_nll, structure=structure,
                       new_species=new_species, delta=delta, n=n, k=k, chi2=chi2)
    diag_lines: list[str] = []
    diag_fields: dict = {}
    if scenario.diagnostics is not None:
        try:
            diag_lines, diag_fields = await scenario.diagnostics(ctx)
        except Exception as exc:  # diagnostics are report-only; never fail the run on them
            diag_lines = [f"(diagnostics error: {type(exc).__name__}: {exc})"]

    print("\n" + "=" * 70)
    print(f"RESULT — {scenario.name}")
    print("=" * 70)
    print(f"  reactions now:        {[r.get('id') for r in reactions]}")
    for line in diag_lines:
        print(f"  {line}")
    print(f"  new species:          {new_species or 'none'}")
    print(f"  fit: nll={final_nll:.6g}  chi2={chi2:.6g}  k={k}  n={n}  (baseline nll {baseline.nll:.6g}, Δnll {delta:.6g})"
          if chi2 is not None else
          f"  fit: nll={final_nll:.6g}  k={k}  n={n}  (baseline nll {baseline.nll:.6g}, Δnll {delta:.6g})")
    if delta_bic is not None:
        print(f"  BIC:  candidate {bic_cand:.6g} | true-model {bic_ref:.6g} | "
              f"ΔBIC {delta_bic:+.6g}  (accept if <= {scenario.bic_threshold:g})")
    else:
        print(f"  BIC:  unavailable (reference {'missing' if ref is None else 'ok'}, n={n})")
    if p_gof is not None:
        print(f"  chi2 GoF:             p={p_gof:.3g} (dof={dof}) "
              f"— {'consistent with noise' if p_gof >= 0.05 else 'worse than noise'}")
    # Wall-clock split: time inside MCP tool calls (dominated by the AMICI compile +
    # pyPESTO fits + SQL) vs LLM+overhead (the remainder — inference + orchestration).
    wall = time.monotonic() - t0
    compute_s = clock.seconds
    llm_s = max(0.0, wall - compute_s)
    if clock.calls:
        print(f"  time:                 total {wall:.0f}s | tool calls {compute_s:.0f}s "
              f"({clock.calls} calls) | LLM+overhead {llm_s:.0f}s")
    if errors:
        print(f"  rejected tool calls:  {len(errors)} (see get_errors)")
    ant_diff = diff.get("model_diff") if isinstance(diff, dict) else None
    if ant_diff:
        print("  model diff (v0 -> HEAD):")
        for line in str(ant_diff).splitlines():
            print("    " + line)
    if success:
        verdict = (f"not rejectable vs the true model (ΔBIC {delta_bic:+.3g} <= "
                   f"{scenario.bic_threshold:g}); the repair explains the data as well as the truth.")
    elif delta_bic is None:
        verdict = "could not evaluate BIC (missing reference.json or no data points)."
    else:
        verdict = (f"rejectable: ΔBIC {delta_bic:+.3g} > {scenario.bic_threshold:g} — the fit is "
                   "decisively worse than the true model.")
    print(f"\n  {'SUCCESS' if success else 'NOT FIXED'}: {verdict}")
    print("=" * 70)

    result = {
        **meta,
        "scenario": scenario.name,
        "baseline_nll": baseline.nll,
        "final_nll": final_nll,
        "delta": delta,
        "chi2": chi2,
        "k": k,
        "n": n,
        "bic": bic_cand,
        "bic_reference": bic_ref,
        "delta_bic": delta_bic,
        "bic_threshold": scenario.bic_threshold,
        "chi2_dof": dof,
        "chi2_gof_pvalue": p_gof,
        "reference": ref,
        "reactions": [r.get("id") for r in reactions],
        "species": sorted(current_species),
        "new_species": new_species,
        "success": success,
        "head_version": head,
        "n_errors": len(errors) if isinstance(errors, list) else 0,
        "wall_seconds": round(wall, 1),
        "compute_seconds": round(compute_s, 1),
        "llm_seconds": round(llm_s, 1),
        "tool_calls": clock.calls,
        "run_ended_at": _now_iso(),
        **diag_fields,
    }
    (work / "result.json").write_text(json.dumps(result, indent=2))
    print(f"  (machine-readable result -> {(work / 'result.json').relative_to(scenario.example_dir.parent)})")
    return (0 if success else 1), result


# --------------------------------------------------------------------------- server
# Launched with: python -c <bootstrap> <yaml> <build_root> <ref_mode> <host> <port>
_SERVER_BOOTSTRAP = (
    "import sys\n"
    "import uvicorn\n"
    "_Config = uvicorn.Config\n"
    # Silence uvicorn's per-request access log so the server's stdout carries only
    # AMICI's C++ build-toolchain output (captured to work/amici_build.log).
    "uvicorn.Config = lambda *a, **k: _Config(*a, **{**k, 'access_log': False})\n"
    "from petab_mcp.server import create_server\n"
    "yaml_path, build_root, ref_mode, host, port = sys.argv[1:6]\n"
    "srv = create_server(yaml_path, build_root=build_root, reference_mode=ref_mode)\n"
    "srv.settings.host = host\n"
    "srv.settings.port = int(port)\n"
    "srv.run(transport='streamable-http')\n"
)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_until_listening(host: str, port: int, proc: subprocess.Popen, log: Path,
                          timeout: float = 120.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early (code {proc.returncode}); see {log}")
        with socket.socket() as s:
            s.settimeout(0.5)
            if s.connect_ex((host, port)) == 0:
                return
        time.sleep(0.2)
    raise TimeoutError(f"server did not start listening on {host}:{port} within {timeout}s")


async def _drive(scenario: Scenario, session: ClientSession, work: Path,
                 args: argparse.Namespace) -> "tuple[int, dict]":
    """Preflight, run the chosen driver, and judge — all on one live session.

    Returns ``(exit_code, result)`` (see :func:`_judge`).
    """
    t0 = time.monotonic()
    clock = _wrap_tool_clock(session)  # split server/compute time from LLM time
    call = _make_call(session)

    desc = await call("describe_problem")
    print(f"[preflight] model '{desc.get('model_id')}' | "
          f"reactions={desc.get('n_reactions_detail')} | "
          f"observables={desc.get('observables')} | "
          f"free params={desc.get('free_parameters')}")

    # Record the initial species set so the judge can reject "repairs" that invent a new,
    # unobserved species to bend the curve rather than fixing the mechanism.
    baseline_structure = await call("get_model_structure")
    baseline_species = set(baseline_structure.get("species", []) or [])

    print("[preflight] scoring baseline (first AMICI compile is slow)...")
    baseline_fit = await call("get_fit_summary")
    baseline = Baseline(
        nll=float(baseline_fit["neg_log_likelihood"]),
        species=baseline_species,
        structure=baseline_structure,
    )
    print(f"[preflight] baseline neg_log_likelihood = {baseline.nll:.6g}")

    run_started_at = _now_iso()
    git_sha = _git_sha(scenario.example_dir)

    # --- scripted driver: deterministic canonical repair, no LLM, no tracing --------------
    if args.driver == "scripted":
        print(f"[scripted] applying canonical fix: {scenario.fix_summary or '(see scenario)'}")
        await scenario.scripted_fix(call, args.n_starts)
        meta = {
            "driver": args.driver, "model": None, "stop_reason": "completed",
            "git_sha": git_sha, "example_version": scenario.version,
            "recursion_limit": None, "n_starts": args.n_starts, "temperature": None,
            "run_started_at": run_started_at,
            "phoenix_project": None, "phoenix_trace_id": None,
        }
        exit_code, result = await _judge(scenario, call, baseline, work, meta, clock, t0)
        return exit_code, result

    # --- deepagents driver: the LLM panel diagnoses + repairs -----------------------------
    # Arize Phoenix tracing (on by default; disable with --no-phoenix): instrument the
    # LangChain stack BEFORE the agent runs so every LLM + tool-call span is captured.
    tracing = None
    if args.phoenix:
        from .tracing import setup_phoenix_tracing

        tracing = setup_phoenix_tracing(project_name=args.phoenix_project)

    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit(
            "No OpenRouter API key: set OPENROUTER_API_KEY (or pass --api-key). "
            "Get one at https://openrouter.ai/keys, or use --driver scripted for a "
            "no-LLM self-test."
        )
    print(f"[agent] driving {args.model} via OpenRouter + deepagents "
          f"(recursion_limit={args.recursion_limit})")
    from .agent import run_deepagents
    from .tracing import record_run_result, run_span, trace_id_hex

    # A run-level span parents the whole agent trace, so each run is one self-describing
    # trace (model + fit metrics + verdict on the root); its trace id is stamped into
    # result.json so the across-example aggregator can join usage back to it.
    span_attrs = {
        "scenario": scenario.name, "example_version": scenario.version,
        "model": args.model, "driver": args.driver, "temperature": args.temperature,
        "recursion_limit": args.recursion_limit, "n_starts": args.n_starts,
        "git_sha": git_sha, "run_started_at": run_started_at,
    }
    exit_code, result = 1, {}
    try:
        with run_span(tracing, name="model_repair_run", attributes=span_attrs) as span:
            agent_out = await run_deepagents(
                session,
                model=args.model,
                base_url=args.base_url,
                api_key=api_key,
                system_prompt=scenario.system_prompt,
                task=scenario.task,
                recursion_limit=args.recursion_limit,
                temperature=args.temperature,
            )
            stop_reason = agent_out.get("stop_reason", "completed")
            if stop_reason != "completed":
                print(f"[agent] stopped early: {stop_reason} "
                      f"(raise --recursion-limit if it ran out of steps)")
            meta = {
                "driver": args.driver, "model": args.model, "stop_reason": stop_reason,
                "git_sha": git_sha, "example_version": scenario.version,
                "recursion_limit": args.recursion_limit,
                "n_starts": args.n_starts, "temperature": args.temperature,
                "run_started_at": run_started_at,
                "phoenix_project": tracing.project_name if tracing else None,
                "phoenix_trace_id": trace_id_hex(span),
            }
            exit_code, result = await _judge(scenario, call, baseline, work, meta, clock, t0)
            record_run_result(span, result)
    finally:
        if tracing is not None:
            tracing.flush()  # export any buffered spans before we (maybe) exit
            print(f"[phoenix] traces recorded under project '{tracing.project_name}' "
                  f"-> {tracing.ui_url}")
            if args.phoenix_keep_alive:
                print("[phoenix] keeping the process alive so the traces stay browsable; "
                      "press Ctrl-C to exit.")
                try:
                    while True:
                        await asyncio.sleep(3600)
                except (KeyboardInterrupt, asyncio.CancelledError):
                    print("\n[phoenix] shutting down.")

    return exit_code, result


# --------------------------------------------------------------------------- main
async def _main_async(scenario: Scenario, args: argparse.Namespace) -> "tuple[int, dict]":
    work = cleanup(scenario)

    work_yaml = work / "problem.yaml"
    # AMICI builds are keyed by SBML hash and never go stale, so keep them OUTSIDE work/
    # (wiped each run) to avoid recompiling identical models on repeat runs.
    build_root = scenario.example_dir / ".amici_build"
    host, port = "127.0.0.1", _free_port()
    server_cmd = [
        sys.executable, "-c", _SERVER_BOOTSTRAP,
        str(work_yaml), str(build_root), args.reference_validation, host, str(port),
    ]
    server_log_path = work / "server.log"
    amici_log_path = work / "amici_build.log"
    server_log = server_log_path.open("w")
    amici_log = amici_log_path.open("w")

    print(f"[server] launching on http://{host}:{port}/mcp")
    print(f"[server] server log -> {server_log_path.name} | AMICI build log -> {amici_log_path.name}")
    # Split the streams: fd1 (stdout) carries the C++ build toolchain output that would
    # otherwise corrupt a stdio channel -> amici_build.log; fd2 (stderr) carries uvicorn
    # request logs + Python/AMICI logging -> server.log.
    try:
        proc = subprocess.Popen(server_cmd, stdout=amici_log, stderr=server_log)
    except Exception:
        for handle in (server_log, amici_log):
            with contextlib.suppress(Exception):
                handle.close()
        raise
    try:
        _wait_until_listening(host, port, proc, server_log_path)
        url = f"http://{host}:{port}/mcp"
        async with streamablehttp_client(url) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await _drive(scenario, session, work, args)
    finally:
        # Tear down without masking any in-flight exception.
        with contextlib.suppress(Exception):
            proc.terminate()
            proc.wait(timeout=10)
        if proc.poll() is None:
            with contextlib.suppress(Exception):
                proc.kill()
                proc.wait(timeout=5)
        for handle in (server_log, amici_log):
            with contextlib.suppress(Exception):
                handle.close()


def build_parser(scenario: Scenario) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=f"{scenario.name} repair demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--driver", choices=["deepagents", "scripted"], default="deepagents",
                   help="deepagents: drive an OpenRouter model; scripted: deterministic repair, no LLM.")
    p.add_argument("--model", default="qwen/qwen3-235b-a22b",
                   help="OpenRouter model id (deepagents driver).")
    p.add_argument("--base-url", default="https://openrouter.ai/api/v1",
                   help="OpenAI-compatible base URL for the LLM provider (default: OpenRouter).")
    p.add_argument("--api-key", default=None,
                   help="OpenRouter API key (default: the OPENROUTER_API_KEY environment variable).")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="LLM sampling temperature (deepagents driver); recorded per run for sweeps.")
    p.add_argument("--recursion-limit", type=int, default=150, help="deepagents/LangGraph step budget.")
    p.add_argument("--n-starts", type=int, default=20,
                   help="multistart count for the auto-fit on the repair commit.")
    p.add_argument("--reference-validation", choices=["required", "best_effort", "off"], default="off",
                   help="Server reference-validation mode (off = no network; default for a hermetic demo).")
    # --- Arize Phoenix agent tracing (on by default; deepagents driver only) ---
    p.add_argument("--no-phoenix", dest="phoenix", action="store_false",
                   help="Disable Arize Phoenix tracing (it is ON by default).")
    default_project = scenario.phoenix_project or f"petab-mcp-{scenario.name}"
    p.add_argument("--phoenix-project", default=os.environ.get("PHOENIX_PROJECT_NAME", default_project),
                   help="Phoenix project name to group this run's traces under.")
    p.add_argument("--phoenix-keep-alive", action="store_true",
                   help="After the run, keep the process (and any locally launched Phoenix app) "
                        "alive until Ctrl-C so the traces stay browsable.")
    p.set_defaults(phoenix=True)
    # --- Weights & Biases metrics logging (off by default; the sweep's analysis UI) ---
    p.add_argument("--wandb", action="store_true",
                   help="Log this run's outcome (pass / ΔBIC / χ² / timings + config) to "
                        "Weights & Biases. Orthogonal to Phoenix tracing — use both: W&B for "
                        "the sweep metrics UI, Phoenix for the agent trace.")
    p.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "petab-mcp-sweep"),
                   help="W&B project to log runs under (default: $WANDB_PROJECT or petab-mcp-sweep).")
    return p


def run_scenario(scenario: Scenario, argv: list[str] | None = None) -> int:
    """Parse args, run the pipeline for ``scenario``, and return a process exit code."""
    args = build_parser(scenario).parse_args(argv)
    try:
        exit_code, result = asyncio.run(_main_async(scenario, args))
    except KeyboardInterrupt:
        return 130
    # W&B logging is a post-run, best-effort step (covers run.py direct AND the sweep's
    # run.py subprocesses) — never let it change the exit code.
    if getattr(args, "wandb", False):
        try:
            from .wandb_logging import log_result
            log_result(result, project=args.wandb_project)
        except Exception as exc:  # pragma: no cover - telemetry must not break the run
            print(f"[wandb] skipped ({type(exc).__name__}: {exc})")
    return exit_code
