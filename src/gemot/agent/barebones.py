"""``run_barebones`` — the NULL arm: one stock deepagent, a sandboxed shell, the PEtab files on disk.

The control for the harness design: the same model, problem, Python environment and node as a
panel run, but the ``deepagents`` agent as installed (file tools, ``execute``, todo list,
summarization, its own ``general-purpose`` subagent) working directly on ``work/`` and asked to
deliver a finished PEtab problem in place. It keeps the panel driver's infrastructure (provider
pin, retries and checkpoint resume, output ceiling, runaway guard, token/wall/stall caps, token
accounting) and the panel's curated bio tools; it has none of the design (no MCP tools, versions,
analysts, finish gate or dispatch guards). Prompts are in :mod:`gemot.prompts`
(``BAREBONES_*``).

``execute`` runs under bubblewrap: system trees and the Python environment read-only, the
workspace read-write at ``/work``, no network, no ``$HOME``, an allowlisted environment.
:func:`sandbox_preflight` proves that on the node before the first LLM call and there is no
unsandboxed fallback. Every command is recorded in ``workspace_audit`` as provenance.

Scoring is the caller's (:mod:`gemot.runner.barebones`). Design decisions and their
reasons are in ``docs/RATIONALE.md`` (the ``#203`` entries).
"""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from pathlib import Path
from typing import Any

from .driver import (
    _LLM_MAX_RETRIES,
    _LLM_TIMEOUT_S,
    _MAX_RUN_TOKENS,
    _NO_EMISSION_YET,
    _RUN_RESUME_ATTEMPTS,
    _STEP_STALL_TIMEOUT_S,
    _chat_model_class,
    _content_filter_stop_reason,
    _env_token_limit,
    _extra_body,
    _state_message_count,
    _transient_provider_reason,
)
from ..prompts import BAREBONES_BIO_TOOLS_TMPL
from .biocurator import LITERATURE_BIO_TOOLS, discover_bio_tools
from .circuit_breaker import (
    _disabled_tool_filter,
    _new_breaker_state,
    _wrap_tools_with_circuit_breaker,
)
from .guards import _reasoning_runaway_guard, new_guard_stats
from .literature import _wrap_literature_recorder
from .sandbox import default_env_prefix, new_shell_audit, sandbox_backend_class
from .subagents import _general_purpose_enabled
from .token_usage import _token_accountant

#: How much of the agent's final reply ``result.json`` keeps.
_FINAL_REPLY_CHARS = 4000


def _final_reply(state: Any) -> str | None:
    """The text of the last AI message in a ``stream_mode="values"`` emission, or ``None``."""
    if not isinstance(state, dict):
        return None
    msgs = state.get("messages") or []
    for msg in reversed(msgs):
        if getattr(msg, "type", None) == "ai" or (
            isinstance(msg, dict) and msg.get("role") == "assistant"
        ):
            content = (
                getattr(msg, "content", None) if not isinstance(msg, dict) else msg.get("content")
            )
            if isinstance(content, list):
                content = "\n".join(
                    str(b.get("text", "")) if isinstance(b, dict) else str(b) for b in content
                )
            text = str(content or "").strip()
            return text[:_FINAL_REPLY_CHARS] if text else None
    return None


async def run_barebones(
    workspace: Path,
    *,
    model: str,
    base_url: str,
    api_key: str,
    system_prompt: str,
    task: str,
    recursion_limit: int,
    temperature: float = 1.0,
    execute_timeout_s: float,
    n_cpus: int = 1,
    max_wall_seconds: float | None = None,
    max_run_tokens: int | None = _MAX_RUN_TOKENS,
    bio_sessions: Any | None = None,
    bio_literature: bool = False,
) -> dict[str, Any]:
    """Drive one stock deepagent over ``workspace`` until it stops, and return the accounting.

    The caller has run :func:`sandbox_preflight`. ``execute_timeout_s`` bounds one shell command
    and sizes the stall watchdog; ``n_cpus`` sets the sandbox's worker/thread knobs;
    ``bio_sessions`` are the panel's bio-knowledge MCP sessions, whose curated tools this agent
    gets as plain tools (calls run in this process, so the sandbox stays offline). The loop is
    :func:`.driver.run_deepagents`'s minus the finish gate and silent-stop bounce: a graph that
    ends is ``completed``; transients and stalls resume from the checkpoint; refusals are terminal.
    """
    from deepagents import create_deep_agent

    from ..fit_budget import FIT_WATCHDOG_MARGIN

    audit = new_shell_audit()
    backend = sandbox_backend_class()(
        workspace,
        env_prefix=default_env_prefix(),
        n_cpus=n_cpus,
        timeout=execute_timeout_s,
        audit=audit,
    )

    bio_tools = await discover_bio_tools(
        bio_sessions, literature=bio_literature, holder="the barebones agent"
    )
    bio_names = sorted(n for t in bio_tools if (n := getattr(t, "name", None)))
    breaker_state = _new_breaker_state()
    _wrap_tools_with_circuit_breaker(
        bio_tools, breaker_state, transport_disable_names=set(bio_names)
    )
    literature_log: list = []
    _wrap_literature_recorder(
        [t for t in bio_tools if getattr(t, "name", None) in LITERATURE_BIO_TOOLS], literature_log
    )
    if bio_tools:
        system_prompt = system_prompt + BAREBONES_BIO_TOOLS_TMPL.format(
            n=len(bio_names), names=", ".join(bio_names)
        )
    print(
        f"[null] {len(bio_names)} bio knowledge tool(s)"
        + (f": {bio_names}" if bio_names else " (none resolved)")
    )

    guard_stats: dict[str, Any] = new_guard_stats()
    max_input_tokens = _env_token_limit("PETAB_MCP_MAX_INPUT_TOKENS", 1_048_576, minimum=1)
    max_output_tokens = _env_token_limit("PETAB_MCP_MAX_OUTPUT_TOKENS", 131_072, minimum=0)
    prompt_cache_key = f"petab-null-{uuid.uuid4().hex[:12]}"
    # Same construction as the panel's lead (see run_deepagents).
    llm = _chat_model_class()(
        name="ChatOpenAI",
        profile={"max_input_tokens": max_input_tokens},
        model=model,
        temperature=temperature,
        base_url=base_url,
        api_key=api_key,
        timeout=_LLM_TIMEOUT_S,
        max_retries=_LLM_MAX_RETRIES,
        stream_usage=True,
        **({"max_tokens": max_output_tokens} if max_output_tokens else {}),
        model_kwargs={"prompt_cache_key": prompt_cache_key},
        extra_body=_extra_body(model),
    )

    from json import JSONDecodeError

    from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError

    _TRANSIENT_ERRORS: tuple = (
        APIConnectionError,
        APITimeoutError,
        InternalServerError,
        RateLimitError,
        JSONDecodeError,
    )
    with contextlib.suppress(Exception):
        import httpx

        _TRANSIENT_ERRORS = _TRANSIENT_ERRORS + (httpx.TransportError,)

    # deepagents' own `general-purpose` subagent stays ON (RATIONALE.md); recorded, not assumed,
    # because the panel driver disables it through a process-global profile registration.
    gp_enabled = _general_purpose_enabled(llm)
    guard_stats["general_purpose_subagent"] = gp_enabled
    guard_stats["subagent_names"] = ["general-purpose"] if gp_enabled is not False else []
    if gp_enabled is False:
        print(
            "[null] WARNING: deepagents' `general-purpose` subagent is disabled in this "
            "process — this run is not the stock agent."
        )

    checkpointer = None
    with contextlib.suppress(Exception):
        from langgraph.checkpoint.memory import MemorySaver

        checkpointer = MemorySaver()

    agent = create_deep_agent(
        model=llm,
        tools=bio_tools,
        system_prompt=system_prompt,
        backend=backend,
        # Infrastructure only: the runaway guard and the breaker's disabled-tool filter.
        middleware=[_disabled_tool_filter(breaker_state), _reasoning_runaway_guard(guard_stats)],
        **({"checkpointer": checkpointer} if checkpointer is not None else {}),
    )

    token_accountant = _token_accountant()
    config: dict[str, Any] = {"recursion_limit": recursion_limit, "callbacks": [token_accountant]}
    if checkpointer is not None:
        config["configurable"] = {"thread_id": "petab-null-run"}

    # One super-step is one shell command or one LLM call; the watchdog sits above the longer.
    step_limit_base = max(_STEP_STALL_TIMEOUT_S, execute_timeout_s * FIT_WATCHDOG_MARGIN)
    deadline = (time.monotonic() + max_wall_seconds) if max_wall_seconds is not None else None
    # Reserve one step of headroom, as the panel driver does.
    step_headroom = (
        min(execute_timeout_s, max_wall_seconds * 0.5) if max_wall_seconds is not None else 0.0
    )
    soft_deadline = (deadline - step_headroom) if deadline is not None else None

    stream_input: Any = {"messages": [{"role": "user", "content": task}]}
    resume_left = _RUN_RESUME_ATTEMPTS if checkpointer is not None else 0
    stall_backoff = 1.0
    last_message_count: Any = _NO_EMISSION_YET
    final_reply: str | None = None
    stop_reason = "completed"
    while True:
        retriable: str | None = None
        agen = None
        wall_budget_exceeded = token_budget_exceeded = context_budget_exceeded = False
        step_wait = step_limit_base * stall_backoff
        replay_pending = stream_input is None
        try:
            agen = agent.astream(stream_input, config=config, stream_mode="values").__aiter__()
            while True:
                try:
                    step_limit = step_limit_base * stall_backoff
                    step_wait = (
                        step_limit
                        if deadline is None
                        else max(1.0, min(step_limit, deadline - time.monotonic()))
                    )
                    state = await asyncio.wait_for(agen.__anext__(), timeout=step_wait)
                except StopAsyncIteration:
                    break
                n_msgs = _state_message_count(state)
                replay = replay_pending and last_message_count is _NO_EMISSION_YET
                replay_pending = False
                if not replay and (n_msgs is None or n_msgs != last_message_count):
                    stall_backoff = 1.0
                last_message_count = n_msgs
                if (reply := _final_reply(state)) is not None:
                    final_reply = reply
                if soft_deadline is not None and time.monotonic() >= soft_deadline:
                    wall_budget_exceeded = True
                    break
                if (
                    max_run_tokens is not None
                    and token_accountant.prompt + token_accountant.completion >= max_run_tokens
                ):
                    token_budget_exceeded = True
                    break
                if token_accountant.max_prompt >= 0.90 * max_input_tokens:
                    context_budget_exceeded = True
                    break
            stop_reason = (
                "wall_time_budget_exceeded"
                if wall_budget_exceeded
                else "token_budget_exceeded"
                if token_budget_exceeded
                # A budget stop with a bare label, as in the panel's driver.
                else "context_budget_exceeded"
                if context_budget_exceeded
                else "completed"
            )
        except asyncio.TimeoutError:
            if soft_deadline is not None and time.monotonic() >= soft_deadline:
                wall_budget_exceeded = True
                stop_reason = "wall_time_budget_exceeded"
            else:
                retriable = (
                    f"stalled: no step completed in {step_wait:.0f}s "
                    "(hung provider call or wedged shell command)"
                )
        except _TRANSIENT_ERRORS as exc:
            retriable = f"{type(exc).__name__}: {exc}"
        except Exception as exc:
            if (filtered := _content_filter_stop_reason(exc)) is not None:
                stop_reason = filtered
            elif (transient := _transient_provider_reason(exc)) is not None:
                retriable = transient
            else:
                stop_reason = f"{type(exc).__name__}: {exc}"
        finally:
            if agen is not None:
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(agen.aclose(), timeout=10)
        if wall_budget_exceeded:
            print(
                f"[null] wall-time budget ({max_wall_seconds:.0f}s) reached — stopping at this "
                "step boundary; the judge scores whatever is on disk."
            )
            break
        if token_budget_exceeded:
            print(
                f"[null] token budget exceeded: "
                f"{token_accountant.prompt + token_accountant.completion:,} tokens over "
                f"{max_run_tokens:,} in {token_accountant.n_calls} LLM calls — stopping."
            )
            break
        if context_budget_exceeded:
            print(
                f"[null] context window approached: largest prompt "
                f"{token_accountant.max_prompt:,} tokens >= 90% of the {max_input_tokens:,}-token "
                "window — stopping."
            )
            break
        if retriable is None:
            break
        if soft_deadline is not None and time.monotonic() >= soft_deadline:
            stop_reason = f"wall_time_budget_exceeded (during retry: {retriable})"
            break
        if (
            max_run_tokens is not None
            and token_accountant.prompt + token_accountant.completion >= max_run_tokens
        ):
            stop_reason = "token_budget_exceeded"
            break
        # A stall after the largest prompt crossed the line skipped the boundary check above.
        if token_accountant.max_prompt >= 0.90 * max_input_tokens:
            stop_reason = "context_budget_exceeded"
            break
        if resume_left <= 0:
            stop_reason = retriable
            break
        resume_left -= 1
        if retriable.startswith("stalled:"):
            stall_backoff *= 2.0
        backoff = min(5.0 * (_RUN_RESUME_ATTEMPTS - resume_left), 30.0)
        print(
            f"[null] {retriable} — resuming from last checkpoint in {backoff:.0f}s "
            f"({resume_left} resume(s) left, stall window x{stall_backoff:.0f})"
        )
        await asyncio.sleep(backoff)
        stream_input = None

    token_usage = token_accountant.totals()
    if token_usage["n_llm_calls"] == 0 and audit["n_shell_commands"] > 0:
        print(
            "[null] WARNING: token accounting captured 0 LLM calls despite shell activity — "
            "token_usage in result.json is understated."
        )
    print(
        f"[null] {audit['n_shell_commands']} shell command(s), "
        f"{audit['shell_seconds']:.0f}s in the sandbox, {token_usage['n_llm_calls']} LLM call(s)"
    )
    return {
        "stop_reason": stop_reason,
        "final_reply": final_reply,
        "workspace_audit": audit,
        "openrouter_extra_body": _extra_body(model),
        "openrouter_endpoints": token_accountant.endpoints(),
        "token_usage": token_usage,
        "max_output_tokens": max_output_tokens,
        "max_input_tokens": max_input_tokens,
        "execute_timeout_s": execute_timeout_s,
        "prompt_cache_key": prompt_cache_key,
        # Same keys the panel records for its biocurator, so one query reads both arms.
        "biocurator_tools": bio_names,
        "biocurator_literature": literature_log,
        "n_tool_errors": breaker_state["n_tool_errors"],
        "disabled_tools": sorted(breaker_state["disabled"]),
        **guard_stats,
    }
