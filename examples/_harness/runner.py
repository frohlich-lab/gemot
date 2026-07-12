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


# --------------------------------------------------------------------------- steps
def cleanup(scenario: Scenario) -> Path:
    """Recreate work/ from the pristine broken problem (also drops old provenance)."""
    work = scenario.example_dir / "work"
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(scenario.problem_dir, work)
    print(f"[cleanup] fresh working copy at {work.relative_to(scenario.example_dir.parent)}")
    return work


async def _judge(scenario: Scenario, call, baseline: Baseline, work: Path, meta: dict) -> int:
    """Gather the post-repair state, delegate the domain verdict to ``scenario.check``,
    print a RESULT report, write ``result.json``, and return an exit code."""
    structure = await call("get_model_structure")
    reactions = structure.get("reactions", []) or []
    current_species = set(structure.get("species", []) or [])
    new_species = sorted(current_species - baseline.species)

    final = await call("get_fit_summary")
    final_nll = float(final["neg_log_likelihood"])
    head = int((await call("current_version")).get("version", 0))
    diff = await call("diff_versions", from_version=0, to_version=head) if head else {}
    errors = await call("get_errors", limit=20)
    delta = baseline.nll - final_nll

    ctx = JudgeContext(
        call=call,
        baseline=baseline,
        final_nll=final_nll,
        structure=structure,
        new_species=new_species,
        delta=delta,
    )
    verdict = await scenario.check(ctx)

    print("\n" + "=" * 70)
    print(f"RESULT — {scenario.name}")
    print("=" * 70)
    print(f"  reactions now:        {[r.get('id') for r in reactions]}")
    for line in verdict.lines:
        print(f"  {line}")
    print(f"  new species:          {new_species or 'none'}")
    print(f"  baseline nll:         {baseline.nll:.6g}")
    print(f"  final nll:            {final_nll:.6g}")
    strong = delta >= scenario.min_delta
    print(f"  improvement (delta):  {delta:.6g}  (need >= {scenario.min_delta:g}"
          + ("; clear)" if strong else "; NOT enough)"))
    if errors:
        print(f"  rejected tool calls:  {len(errors)} (see get_errors)")
    ant_diff = diff.get("model_diff") if isinstance(diff, dict) else None
    if ant_diff:
        print("  model diff (v0 -> HEAD):")
        for line in str(ant_diff).splitlines():
            print("    " + line)
    print(f"\n  {'SUCCESS' if verdict.success else 'NOT FIXED'}: {verdict.verdict}")
    print("=" * 70)

    result = {
        **meta,
        "scenario": scenario.name,
        "baseline_nll": baseline.nll,
        "final_nll": final_nll,
        "delta": delta,
        "min_delta": scenario.min_delta,
        "strong": strong,
        "reactions": [r.get("id") for r in reactions],
        "species": sorted(current_species),
        "new_species": new_species,
        "success": verdict.success,
        "head_version": head,
        "n_errors": len(errors) if isinstance(errors, list) else 0,
        **verdict.fields,
    }
    (work / "result.json").write_text(json.dumps(result, indent=2))
    print(f"  (machine-readable result -> {(work / 'result.json').relative_to(scenario.example_dir.parent)})")
    return 0 if verdict.success else 1


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
                 args: argparse.Namespace) -> int:
    """Preflight, run the chosen driver, and judge — all on one live session."""
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

    stop_reason = "completed"
    tracing = None
    if args.driver == "scripted":
        print(f"[scripted] applying canonical fix: {scenario.fix_summary or '(see scenario)'}")
        await scenario.scripted_fix(call, args.n_starts)
    else:
        # Arize Phoenix tracing (on by default; disable with --no-phoenix): instrument the
        # LangChain stack BEFORE the agent runs so every LLM + tool-call span is captured.
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

        result = await run_deepagents(
            session,
            model=args.model,
            base_url=args.base_url,
            api_key=api_key,
            system_prompt=scenario.system_prompt,
            task=scenario.task,
            recursion_limit=args.recursion_limit,
        )
        stop_reason = result.get("stop_reason", "completed")
        if stop_reason != "completed":
            print(f"[agent] stopped early: {stop_reason} "
                  f"(raise --recursion-limit if it ran out of steps)")

    meta = {
        "driver": args.driver,
        "model": args.model if args.driver != "scripted" else None,
        "stop_reason": stop_reason,
    }
    exit_code = await _judge(scenario, call, baseline, work, meta)

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

    return exit_code


# --------------------------------------------------------------------------- main
async def _main_async(scenario: Scenario, args: argparse.Namespace) -> int:
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
    p.add_argument("--model", default="openai/gpt-oss-120b",
                   help="OpenRouter model id (deepagents driver).")
    p.add_argument("--base-url", default="https://openrouter.ai/api/v1",
                   help="OpenAI-compatible base URL for the LLM provider (default: OpenRouter).")
    p.add_argument("--api-key", default=None,
                   help="OpenRouter API key (default: the OPENROUTER_API_KEY environment variable).")
    p.add_argument("--recursion-limit", type=int, default=80, help="deepagents/LangGraph step budget.")
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
    return p


def run_scenario(scenario: Scenario, argv: list[str] | None = None) -> int:
    """Parse args, run the pipeline for ``scenario``, and return a process exit code."""
    args = build_parser(scenario).parse_args(argv)
    try:
        return asyncio.run(_main_async(scenario, args))
    except KeyboardInterrupt:
        return 130
