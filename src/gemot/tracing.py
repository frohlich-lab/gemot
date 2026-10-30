"""Arize Phoenix tracing for the deepagents driver.

`Phoenix <https://phoenix.arize.com>`_ is an open-source LLM-observability tool. This
module wires OpenTelemetry tracing into the LangChain / LangGraph stack that
``deepagents`` runs on, so every LLM call and every MCP tool call the agent makes is
captured as a span and is inspectable in the Phoenix UI (prompts, completions, tool
arguments, token counts, latency, and the full agent trace tree).

Tracing is **mandatory** — the Phoenix packages are declared dependencies of the deepagents
driver and there is no flag to turn it off. The function still degrades gracefully (a clear
message, never a crash) if the tracing packages happen to be missing.

Two ways to collect the spans:

* **A Phoenix collector is already running** (self-hosted ``phoenix serve``, the Docker
  image, or Phoenix Cloud): set ``PHOENIX_COLLECTOR_ENDPOINT`` (and, for Cloud,
  ``PHOENIX_API_KEY``) and the spans are exported there. This is the recommended path —
  the traces outlive the run.
* **Nothing is running**: a local Phoenix app is launched in-process
  (http://localhost:6006) for the duration of the run. Because that collector lives
  inside this process, the traces are only browsable while the run is alive — pass
  ``run.py --phoenix-keep-alive`` to hold the process open after the run.

Instrumentation is process-global (it patches LangChain's callback manager), so
``setup_phoenix_tracing`` must be called *before* the agent runs; ``run.py`` does this
right before handing control to ``deepagents``.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass
class TracingHandle:
    """What ``setup_phoenix_tracing`` returns when tracing is active.

    ``local_session`` is the object returned by ``phoenix.launch_app()`` when a local app
    was started (``None`` when exporting to an external collector); keeping a reference to
    it keeps the in-process app alive. ``flush`` forces any buffered spans to be exported
    (call it before the process exits so nothing is lost).
    """

    endpoint: str
    project_name: str
    ui_url: str | None
    local_session: Any = None
    tracer_provider: Any = None

    def flush(self) -> None:
        provider = self.tracer_provider
        if provider is None:
            return
        # BatchSpanProcessor buffers; force it out before the process tears down.
        force_flush = getattr(provider, "force_flush", None)
        if callable(force_flush):
            try:
                force_flush()
            except Exception:  # pragma: no cover - best-effort on shutdown
                pass


def _default_endpoint() -> str:
    """The collector endpoint to export spans to.

    Honours the standard Phoenix / OTel env vars so a run can point at a self-hosted
    server or Phoenix Cloud without any code change; otherwise defaults to a local
    Phoenix on its conventional port.
    """
    for var in ("PHOENIX_COLLECTOR_ENDPOINT", "OTEL_EXPORTER_OTLP_ENDPOINT"):
        value = os.environ.get(var)
        if value:
            return value.rstrip("/")
    return "http://localhost:6006"


def _base_endpoint(endpoint: str) -> str:
    """Collector base URL: strip a trailing OTLP path so callers can append their own.

    A user may set ``PHOENIX_COLLECTOR_ENDPOINT`` either to the Phoenix base
    (``http://host:6006``) or to a full OTLP traces path (``http://host:6006/v1/traces``).
    Normalizing to the base here lets us append ``/v1/traces`` exactly once (avoiding a
    doubled ``/v1/traces/v1/traces``) and gives the correct UI URL (the base) for free.
    """
    url = endpoint.rstrip("/")
    for suffix in ("/v1/traces", "/v1/otlp"):
        if url.endswith(suffix):
            return url[: -len(suffix)]
    return url


def phoenix_base_url(url: str) -> str:
    """Normalize a Phoenix endpoint to its base URL for ``Client(base_url=...)`` — strip a
    trailing OTLP path (``/v1/traces``, ``/v1/otlp``). The single home for this rule; the
    metrics / experiments / explorer helpers delegate here so it can't drift across copies.
    """
    return _base_endpoint(url)


# ------------------------------------------------------------------ span input stripping
# The LangChain instrumentor turns every LangGraph node and every middleware hook into a
# CHAIN span whose ``input.value`` is the serialized graph state -- which embeds the whole
# ``messages`` list. Such hooks fire on every agent step, so the transcript is stored once per
# hook per step, and since the transcript grows with every step the total scales as
# O(steps^2). Measured on the July sweep collector DB: 27.1 GB of its 37.6 GB was these spans'
# ``input.value``, against 4 KB of genuinely-theirs ``output.value``.
#
# Stripping is by span name rather than by a size threshold so it can never touch a span we
# read: any TOOL span (whose ``input.value`` is the tool arguments), the ``model_build_run``
# root span (whose ``input.value`` is this module's own ``petab.*`` run metadata), or a
# subagent's delegation prompt. The cost is that a rename upstream silently stops the saving --
# ``tests/test_tracing_span_inputs.py`` pins these against the node names a real LangChain
# agent graph produces, so that shows up as a test failure. Note the names are LangChain graph
# node names, not class names: the middleware ones come from ``AgentMiddleware.name``, an
# overridable property that merely *defaults* to the class name.
#
# The graph's other per-step node, ``tools``, is deliberately absent: its ``input.value`` is
# the tool call being dispatched, ~0.5 KB, not a transcript copy.
_STRIPPED_INPUT_SPANS = frozenset(
    {
        "model",  # LangChain's model node
        "TodoListMiddleware.after_model",  # deepagents
        "_ToolDispatchGuard.after_model",  # ours, the lead's, see gemot.agent.guards
        "_SubagentTurnLimit.after_model",  # ours, every analyst's, see gemot.agent.guards
    }
)

# LLM spans are stripped of ``input.value`` too, by kind rather than by name (a span's name is
# its chat-model class). Their transcript is ``llm.input_messages``, which the Phoenix UI reads
# (kept as a delta, see below); ``input.value`` is a second copy of it as LangChain-serialized
# messages, and that copy also carries what no reader wants: every MCP result twice (the ToolMessage
# ``artifact`` keeps the full structured result even after deepagents evicted a large one to
# its filesystem) and every earlier turn's ``reasoning_details`` (already in that turn's own
# ``output.value``). On the 2026-10-25 raimundez run it was 97 MB of the LLM spans' 131 MB.
_STRIPPED_INPUT_KINDS = frozenset({"LLM"})

# Only the payload goes; the span, its parentage and its timings stay, so the trace tree is
# unchanged. Dropping these spans outright is NOT an option -- ``ChatOpenAI`` is their child
# and would be orphaned.
_STRIPPED_INPUT_KEYS = ("input.value", "input.mime_type")

# ---------------------------------------------------------------------- LLM input deltas
# What the strip leaves on an LLM span is ``llm.input_messages``: the whole request, which repeats
# the conversation so far, so a conversation of n calls stores ~n/2 copies of its final input --
# O(steps^2) again, and most of the 48-143 MB per run that was left after the strip. So an LLM span
# stores only the messages after its common prefix with the previous call of its conversation,
# re-indexed from 0, and where that prefix is:
#
#   gemot.input_prefix_len   n, the leading messages it shares with the previous call
#   gemot.input_prefix_span  that call's span id
#
# and :func:`rebuild_input_messages` puts the input back as ``input_prev[:n] + delta``. A span
# without the two carries its full input: the first call of a conversation, a checkpoint (below),
# a call without a system message, and every trace written before this.
#
# Lossless by construction, because the prefix is compared message by message: whatever made an
# input other than an extension of the previous one shows up as a shorter prefix, never as lost
# messages. Compaction rewriting the history to [system, summary, tail], a turn-limit notice that
# is sent with one request only, a runaway retry resending the same input -- each was reproduced
# on the real graph and is rebuilt exactly (tests/test_tracing_span_inputs.py). Outputs are not
# touched, so a reply a guard changed before it reached state (a deduplicated tool call, a
# discarded runaway turn) is still on its own span.
#
# A conversation is one agent's thread within one trace: ``lc_agent_name`` (the lead, or the
# analyst), the ``checkpoint_ns`` of a subagent delegation (``tools:<id>``, one per `task` call; the
# lead's is ``model:<id>`` and changes every step, so it is left out), and the system prompt. A call
# without a system message -- deepagents' summarizer -- is one-shot and keeps its full input. The
# key only chooses which call to diff against, and the span records which one that was, so a key
# that groups wrongly costs saving, never correctness.
#
# The cost: the Phoenix UI shows a delta span's new messages only. The whole request of any span is
# :func:`rebuild_input_messages_from_dataframe` over its trace.
_INPUT_MESSAGES = "llm.input_messages."
_PREFIX_LEN = "gemot.input_prefix_len"
_PREFIX_SPAN = "gemot.input_prefix_span"

# A conversation writes its full input again every this many calls, and every conversation does
# after a failed export: a lost span breaks every later input of its chain, and a collector can
# lose one without the export failing. 20 is an analyst delegation's whole turn budget
# (ANALYST_MAX_TURNS), so a delegation is one chain, and a 100-turn lead pays ~2x its final input
# in checkpoints on top of ~1x in deltas, against ~50x written in full.
_FULL_INPUT_EVERY = 20


def _message_digest(message: dict[str, Any]) -> bytes:
    """A message's identity for the prefix comparison: all of its attributes, not just its text,
    so two messages with the same content and different tool calls are different."""
    return hashlib.blake2b(
        json.dumps(message, sort_keys=True, default=str).encode(), digest_size=16
    ).digest()


def _conversation_key(span: Any, attributes: dict[str, Any], system: dict[str, Any]) -> Any:
    """The conversation an LLM call belongs to (see above), or None for a one-shot call."""
    if system.get("message.role") != "system":
        return None
    try:
        metadata = json.loads(attributes.get("metadata") or "{}")
    except (TypeError, ValueError):
        metadata = {}
    metadata = metadata if isinstance(metadata, dict) else {}
    namespace = str(metadata.get("checkpoint_ns") or "")
    return (
        span.context.trace_id,
        metadata.get("lc_agent_name"),
        namespace if namespace.startswith("tools:") else "",
        _message_digest(system),
    )


class _InputDeltas:
    """Rewrites each LLM span's ``llm.input_messages`` as a delta on its conversation's previous
    call. Holds one digest per message of each conversation's last input, not the messages, so a
    finished conversation costs a few KB for the rest of the process."""

    def __init__(self) -> None:
        # conversation -> (its last call's span id, that call's message digests, calls in the
        # chain since its last full input)
        self._last: dict[Any, tuple[str, list[bytes], int]] = {}

    def reset(self) -> None:
        """Make every conversation's next call write its full input."""
        self._last.clear()

    def encode(self, span: Any, attributes: dict[str, Any]) -> dict[str, Any]:
        if span.context is None:
            return attributes
        messages: dict[int, dict[str, Any]] = {}
        rest: dict[str, Any] = {}
        for key, value in attributes.items():
            if not key.startswith(_INPUT_MESSAGES):
                rest[key] = value
                continue
            index, _, field = key[len(_INPUT_MESSAGES) :].partition(".")
            if not index.isdigit():
                return attributes
            messages.setdefault(int(index), {})[field] = value
        # Only an unbroken 0..m-1 list re-indexes and comes back as it was.
        if not messages or max(messages) != len(messages) - 1:
            return attributes
        ordered = [messages[i] for i in range(len(messages))]
        key = _conversation_key(span, attributes, ordered[0])
        if key is None:
            return attributes
        digests = [_message_digest(m) for m in ordered]
        last = self._last.get(key)
        n = 0
        if last is not None and last[2] < _FULL_INPUT_EVERY:
            # At least 1: the system prompt is part of the key.
            while n < min(len(last[1]), len(digests)) and last[1][n] == digests[n]:
                n += 1
        self._last[key] = (format(span.context.span_id, "016x"), digests, last[2] + 1 if n else 1)
        if not n:
            return attributes
        rest[_PREFIX_LEN] = n
        rest[_PREFIX_SPAN] = last[0]
        for i, message in enumerate(ordered[n:]):
            for field, value in message.items():
                rest[f"{_INPUT_MESSAGES}{i}.{field}"] = value
        return rest


def rebuild_input_messages(
    spans: Mapping[str, tuple[Any, Any, Any]],
) -> dict[str, list | None]:
    """Every span's full ``llm.input_messages``, put back together from the deltas the exporter
    wrote (see ``_InputDeltas``).

    ``spans`` maps a span id to what that span stores: its input messages as a list, in whichever
    form the reader has them (Phoenix's spans dataframe, its database, OTel's flat keys grouped by
    index), its ``gemot.input_prefix_len`` and its ``gemot.input_prefix_span``. Both are absent
    (``None``, or NaN from a dataframe) on a span that holds its full input. A span whose chain
    runs into a span not in ``spans`` maps to ``None``, never to a shorter input.
    """
    full: dict[str, list | None] = {}
    for span_id in spans:
        chain: list[str] = []
        current = span_id
        while current not in full and current not in chain:
            if current not in spans:
                full[current] = None
                break
            chain.append(current)
            previous = spans[current][2]
            if not isinstance(previous, str):
                break
            current = previous
        for sid in reversed(chain):
            messages, n, previous = spans[sid]
            messages = list(messages) if isinstance(messages, (list, tuple)) else []
            if not isinstance(previous, str):
                full[sid] = messages
                continue
            base = full.get(previous)
            full[sid] = None if base is None or len(base) < int(n) else base[: int(n)] + messages
    return {span_id: full[span_id] for span_id in spans}


def rebuild_input_messages_from_dataframe(df: Any) -> dict[str, list | None]:
    """:func:`rebuild_input_messages` for the LLM spans of a Phoenix spans dataframe, as
    ``Client().spans.get_spans_dataframe`` returns it (indexed by span id).

    Phoenix flattens only OpenInference's own attributes into columns, so the two ``gemot.*`` ones
    come back nested, as one ``attributes.gemot`` dict (arize-phoenix 20.6). Looked for as flat
    columns they are missing, every span seems to hold its full input, and the deltas come back as
    if they were the inputs. So both shapes are read.
    """
    spans: dict[str, tuple[Any, Any, Any]] = {}
    for span_id, row in df.iterrows():
        if row.get("attributes.openinference.span.kind") != "LLM":
            continue
        nested = row.get("attributes.gemot")
        n, previous = (
            nested.get(key.removeprefix("gemot."))
            if isinstance(nested, dict)
            else row.get(f"attributes.{key}")
            for key in (_PREFIX_LEN, _PREFIX_SPAN)
        )
        spans[str(span_id)] = (row.get("attributes.llm.input_messages"), n, previous)
    return rebuild_input_messages(spans)


def _strip_span_inputs(span: Any, deltas: _InputDeltas | None = None) -> Any:
    """Return ``span`` with the bulky ``input.*`` attributes removed, or unchanged if it
    is neither one of the wrapper spans nor an LLM span. With ``deltas``, an LLM span's input
    messages are also cut down to the delta on its conversation's previous call.

    These checks FIRST: this runs for every span in every export batch, and the large majority
    return here unchanged, so nothing more expensive than a set lookup belongs above them.
    """
    kind = (span.attributes or {}).get("openinference.span.kind")
    if span.name not in _STRIPPED_INPUT_SPANS and kind not in _STRIPPED_INPUT_KINDS:
        return span
    from opentelemetry.sdk.trace import ReadableSpan

    attributes = {
        key: value
        for key, value in (span.attributes or {}).items()
        if key not in _STRIPPED_INPUT_KEYS
    }
    if deltas is not None and kind == "LLM":
        attributes = deltas.encode(span, attributes)
    return ReadableSpan(
        name=span.name,
        context=span.context,
        parent=span.parent,
        resource=span.resource,
        attributes=attributes,
        events=span.events,
        links=span.links,
        kind=span.kind,
        status=span.status,
        start_time=span.start_time,
        end_time=span.end_time,
        instrumentation_scope=span.instrumentation_scope,
    )


def _make_stripping_exporter(inner: Any) -> Any:
    """Wrap a ``SpanExporter`` so wrapper-span inputs are dropped, and LLM inputs cut to their
    deltas, on the way out.

    Filtering at the exporter, not in a second ``SpanProcessor``: a processor added after the
    default one would have to mutate the span while the batch processor's exporter thread may
    already be serializing it. The exporter is also the one place that knows whether a span
    reached the collector, which the delta chains depend on.
    """
    from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

    class _StrippingSpanExporter(SpanExporter):
        def __init__(self, wrapped: Any) -> None:
            self._wrapped = wrapped
            self._deltas = _InputDeltas()
            self._lock = threading.Lock()

        def export(self, spans: Any) -> "SpanExportResult":
            # Encoding and exporting under one lock: a batch encoded against spans whose own export
            # then failed would point at spans the collector never got.
            with self._lock:
                try:
                    result = self._wrapped.export(
                        [_strip_span_inputs(s, self._deltas) for s in spans]
                    )
                except Exception:
                    self._deltas.reset()
                    raise
                if result != SpanExportResult.SUCCESS:
                    self._deltas.reset()
                return result

        def shutdown(self) -> None:
            self._wrapped.shutdown()

        def force_flush(self, timeout_millis: int = 30_000) -> bool:
            return self._wrapped.force_flush(timeout_millis)

    return _StrippingSpanExporter(inner)


def _install_stripping_processor(tracer_provider: Any, traces_endpoint: str) -> bool:
    """Replace the provider's default span processor with one that strips wrapper inputs.

    Returns True if it was installed. Only attempted on Phoenix's ``TracerProvider``, whose
    ``add_span_processor`` documents that it removes the default processor: on a plain OTel
    provider the default would survive and every span would be exported twice -- once
    unstripped -- which is worse than not stripping at all.
    """
    try:
        from phoenix.otel import BatchSpanProcessor, HTTPSpanExporter
        from phoenix.otel import TracerProvider as PhoenixTracerProvider

        if not isinstance(tracer_provider, PhoenixTracerProvider):
            return False
        # `add_span_processor` only drops the default processor while the provider holds one,
        # which it records in the private `_default_processor`: `True` up to arize-phoenix-otel
        # 0.16, the processor itself (else None) from 0.17. If it holds none, ours would be
        # *appended* and every span would export twice -- one copy unstripped. Refuse rather than
        # risk that; the count is asserted in tests/test_tracing_span_inputs.py.
        #
        # Truthiness, NOT `is True`: that check was written against 0.16, and from 0.17 it refused
        # every provider, so every run of the October indra sweeps exported its transcript in full
        # (238 of the 2026-10-25 raimundez run's 389 MB were the stripped spans' inputs).
        if not getattr(tracer_provider, "_default_processor", None):
            return False
        exporter = _make_stripping_exporter(HTTPSpanExporter(traces_endpoint))
        tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter=exporter))
        return True
    except Exception as exc:
        # Telemetry must never take the run down: `drive.py` calls setup_phoenix_tracing
        # unguarded, and add_span_processor joins the default batch processor's worker
        # thread on shutdown, which can raise against an unreachable collector.
        print(f"[phoenix] input-stripping span processor not installed ({exc})")
        return False


def setup_phoenix_tracing(
    project_name: str = "petab-mcp-agent",
    *,
    endpoint: str | None = None,
    launch: bool = True,
) -> TracingHandle | None:
    """Turn on Phoenix/OpenInference tracing for the LangChain-based agent.

    Parameters
    ----------
    project_name:
        Groups this run's spans under a named project in the Phoenix UI.
    endpoint:
        Collector endpoint to export spans to. Defaults to ``PHOENIX_COLLECTOR_ENDPOINT``
        / ``OTEL_EXPORTER_OTLP_ENDPOINT`` if set, else ``http://localhost:6006``.
    launch:
        If ``True`` (the default) and no external collector endpoint is configured, launch
        a local Phoenix app in-process (requires the ``arize-phoenix`` package) so there is
        always somewhere to view the traces.

    Returns a :class:`TracingHandle` on success, or ``None`` if the tracing packages are
    missing (a hint is printed) — the caller then simply runs without tracing.
    """
    # THE INTERLOCK FIRST, before anything that can `return None`. A batch run must never end
    # up untraced, and there are three ways to get there: no endpoint configured (checked
    # below), and the two ImportError paths under it — a partial or mid-upgrade environment
    # where arize-phoenix-otel or the LangChain instrumentor is missing prints a hint and
    # returns None. The slurm gates cannot see that: they check the COLLECTOR, which is alive.
    # So a package gap would reproduce the 2026-08-16 outcome exactly — full budget spent, zero
    # spans — with all three layers reporting success.
    _require = os.environ.get("PETAB_REQUIRE_COLLECTOR", "0").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
        "",
    )

    def _refuse(why: str) -> None:
        raise RuntimeError(
            f"PETAB_REQUIRE_COLLECTOR is set but tracing cannot be established: {why}. "
            "Refusing to run untraced: the run would cost its full compute budget and leave "
            "nothing to analyse."
        )

    try:
        from phoenix.otel import register
    except ImportError:
        print(
            "[phoenix] `arize-phoenix-otel` is not installed; running without tracing.\n"
            "  Install the runner requirements to enable it, or:\n"
            "    pip install arize-phoenix arize-phoenix-otel openinference-instrumentation-langchain"
        )
        if _require:
            _refuse("`arize-phoenix-otel` is not installed")
        return None

    try:
        from openinference.instrumentation.langchain import LangChainInstrumentor
    except ImportError:
        print(
            "[phoenix] `openinference-instrumentation-langchain` is not installed; "
            "running without tracing. Install with:\n"
            "    pip install openinference-instrumentation-langchain"
        )
        if _require:
            _refuse("`openinference-instrumentation-langchain` is not installed")
        return None

    # Only auto-launch a local app when the user didn't point us at an external collector.
    explicit_endpoint = (
        endpoint
        or os.environ.get("PHOENIX_COLLECTOR_ENDPOINT")
        or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    )
    # A BATCH run must never take the throwaway-local-app path. An in-process Phoenix dies with
    # the task and takes every span with it, so the run burns its whole budget and produces
    # nothing analysable — measured on 2026-08-16 at 40 tasks / ~26 CPU-hours, discovered only
    # when the trace pull came back 0/40. slurm/array_task.sh and slurm/make_sweep.py both refuse
    # to start such a run; this is the same rule one layer down, so that stripping the endpoint
    # between the task gate and here cannot silently reinstate the fallback.
    #
    # SCOPE, precisely: array_task.sh is the only thing that sets this variable today, so this
    # is a backstop for the sweep path, NOT protection for callers that bypass it. A hand-run
    # `python run.py` or recover_run.py still gets the local app unless it exports
    # PETAB_REQUIRE_COLLECTOR=1 itself — deliberate (interactive runs want the local app), but
    # it does mean "the third layer covers bypassing callers" would be false.
    if _require and not explicit_endpoint:
        _refuse(
            "no collector endpoint is configured (PHOENIX_COLLECTOR_ENDPOINT / "
            "OTEL_EXPORTER_OTLP_ENDPOINT are unset), and a throwaway in-process Phoenix dies "
            "with this process. Start the shared collector (slurm/phoenix_ctl.sh start)"
        )
    local_session = None
    if launch and not explicit_endpoint:
        try:
            import phoenix as px

            local_session = px.launch_app()
            print(f"[phoenix] launched local Phoenix app at {local_session.url}")
        except ImportError:
            print(
                "[phoenix] launching a local Phoenix app needs the `arize-phoenix` package "
                "(the UI/server). Install it, or point PHOENIX_COLLECTOR_ENDPOINT at a "
                "running Phoenix. Running without tracing."
            )
            return None
        except Exception as exc:  # pragma: no cover - launch is environment-dependent
            print(f"[phoenix] failed to launch local Phoenix app ({exc}); running without tracing.")
            return None

    # Normalize to the collector base so we append the OTLP path exactly once, whether the
    # user gave us a bare host or a full ``/v1/traces`` URL.
    base_endpoint = _base_endpoint(endpoint or _default_endpoint())
    traces_endpoint = f"{base_endpoint}/v1/traces"

    # `register` wires up the OTel tracer provider + OTLP exporter and sets it as global.
    # `batch=True` installs a BatchSpanProcessor instead of the default SimpleSpanProcessor:
    # SimpleSpanProcessor exports one HTTP request PER SPAN synchronously, which — across a
    # sweep of many concurrent runs all pointed at one shared collector — floods the collector
    # (its query API times out) and adds per-call latency to the agent. Batching buffers spans
    # and exports them in far fewer requests; `TracingHandle.flush` forces the buffer out before
    # the process exits so nothing is lost. Different arize-phoenix-otel releases expose slightly
    # different kwargs, so fall back progressively if a kwarg is rejected — the fallback preserves
    # the target by exporting the base endpoint through the env var `register` reads, so we never
    # silently export to localhost when the user configured a remote collector.
    try:
        tracer_provider = register(
            project_name=project_name,
            endpoint=traces_endpoint,
            batch=True,
            set_global_tracer_provider=True,
        )
    except TypeError:
        os.environ.setdefault("PHOENIX_COLLECTOR_ENDPOINT", base_endpoint)
        try:
            tracer_provider = register(project_name=project_name, batch=True)
        except TypeError:  # very old release without `batch` — accept SimpleSpanProcessor
            tracer_provider = register(project_name=project_name)

    # Swap the default span processor for one that drops the redundant wrapper-span inputs
    # (see _STRIPPED_INPUT_SPANS). Best-effort: if the provider isn't Phoenix's, tracing
    # still works, it just exports the full ~4x-duplicated transcript as before.
    if not _install_stripping_processor(tracer_provider, traces_endpoint):
        print(
            "[phoenix] could not install the input-stripping span processor; "
            "traces will carry duplicated agent transcripts and grow quickly."
        )

    # Instrument LangChain (deepagents/LangGraph run on it) so LLM calls and tool calls
    # are captured. Passing the provider explicitly avoids relying on global-lookup order.
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider, skip_dep_check=True)

    ui_url = local_session.url if local_session is not None else base_endpoint
    print(f"[phoenix] tracing enabled (project='{project_name}') -> view traces at {ui_url}")
    return TracingHandle(
        endpoint=traces_endpoint,
        project_name=project_name,
        ui_url=ui_url,
        local_session=local_session,
        tracer_provider=tracer_provider,
    )


# --------------------------------------------------------------------------- run span
# A run-level span that PARENTS the whole agent trace, so each run is one self-describing
# trace: open it in Phoenix and the root carries the model, the fit metrics (chi2, BIC,
# ΔBIC), the verdict, and the timings, while Phoenix's own per-trace token rollup (summed
# from the child LLM spans) shows the usage. The queryable ``petab.*`` attributes make the
# same numbers available in the spans dataframe for the across-example aggregator
# (``metrics.py``) to pull without re-deriving them.

# Result keys copied onto the run span as queryable ``attributes.petab.<key>`` values.
_RESULT_METRIC_KEYS = (
    "scenario",
    "example_version",
    "model",
    "driver",
    "stop_reason",
    "success",
    # Which bar `success` was decided by (see metrics._RESULT_FIELDS): group by it before pooling.
    "judge_mode",
    "baseline_nll",
    "final_nll",
    "delta",
    "chi2",
    "chi2_dof",
    "chi2_gof_pvalue",
    "k",
    "k_estimated",
    "n",
    "k_eff",
    "bic",
    "bic_reference",
    "delta_bic",
    "bic_threshold",
    "criterion",
    "criterion_note",
    "criterion_value",
    "criterion_reference",
    "delta_criterion",
    "head_version",
    "n_errors",
    "tool_calls",
    "wall_seconds",
    "compute_seconds",
    "llm_seconds",
    "compile_seconds",
    "fit_seconds",
    "n_compiles",
    "n_fits",
    "recursion_limit",
    "n_starts",
    "temperature",
    "git_sha",
    "ablation_arm",
    "ablation_state",
)


def _otel_value(value: Any) -> Any:
    """Coerce a value to something OTel accepts (str/bool/int/float); stringify the rest."""
    if isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _set_attrs(span: Any, attrs: dict[str, Any]) -> None:
    """Best-effort ``span.set_attribute`` over a dict, skipping ``None`` values."""
    for key, value in attrs.items():
        if value is None:
            continue
        with contextlib.suppress(Exception):  # never let telemetry break the run
            span.set_attribute(key, _otel_value(value))


@contextlib.contextmanager
def run_span(handle: TracingHandle | None, *, name: str, attributes: dict[str, Any]):
    """Open an OpenInference CHAIN span that becomes the trace root for the agent run.

    Every LLM/tool span the agent emits while this is the current span nests under it.
    Yields the span, or ``None`` when tracing (or ``opentelemetry``) is unavailable so the
    caller can no-op transparently — ``record_run_result``/``trace_id_hex`` also accept
    ``None``.
    """
    if handle is None:
        yield None
        return
    try:
        from opentelemetry import trace
    except Exception:  # pragma: no cover - opentelemetry ships with phoenix.otel
        yield None
        return
    tracer = trace.get_tracer("petab_mcp.harness")
    with tracer.start_as_current_span(name) as span:
        _set_attrs(span, {"openinference.span.kind": "CHAIN"})
        _set_attrs(span, {f"petab.{k}": v for k, v in attributes.items()})
        _set_attrs(
            span,
            {
                "input.value": json.dumps(attributes, default=str),
                "input.mime_type": "application/json",
            },
        )
        yield span


def trace_id_hex(span: Any) -> str | None:
    """The 32-char hex trace id of ``span`` (to join result.json to its Phoenix trace)."""
    if span is None:
        return None
    try:
        tid = span.get_span_context().trace_id
        return format(tid, "032x") if tid else None
    except Exception:
        return None


def record_run_result(span: Any, result: dict[str, Any]) -> None:
    """Stamp the judge's outcome onto the run span: queryable ``petab.*`` attributes, a
    parsed-in-Phoenix ``metadata`` blob, an ``output.value`` panel, and a coarse verdict."""
    if span is None:
        return
    _set_attrs(span, {f"petab.{k}": result.get(k) for k in _RESULT_METRIC_KEYS})
    # The HELD-OUT score and the reference-free adequacy bar, as the same scalar `val_*` /
    # `adequacy_*` pairs the history CSV uses — a span attribute has to be a scalar, and a sweep
    # queries `attributes.petab.val_delta_nll` exactly the way it queries `attributes.petab.delta_bic`.
    # metrics' keys-only flatteners (`_KEY_FLATTENERS`) OWN those names and hand back the mapping, so
    # this does not restate them: a new column reaches both surfaces or neither. (metrics is otherwise
    # the CONSUMER of what this module writes; it is imported here for that single naming authority.
    # The dependency is mutual — metrics imports `phoenix_base_url` from here — so both directions
    # are deliberately function-local to keep import time acyclic.)
    from . import metrics

    for name in metrics._KEY_FLATTENERS:
        _set_attrs(span, {f"petab.{k}": v for k, v in getattr(metrics, name)(result).items()})
    _set_attrs(span, {"petab.verdict": "success" if result.get("success") else "not_fixed"})
    with contextlib.suppress(Exception):
        span.set_attribute("metadata", json.dumps(result, default=str))
    outcome = {
        k: result.get(k)
        for k in (
            "success",
            "chi2",
            "chi2_gof_pvalue",
            "delta_bic",
            "bic",
            "final_nll",
            "stop_reason",
        )
    }
    _set_attrs(
        span,
        {
            "output.value": json.dumps(outcome, default=str),
            "output.mime_type": "application/json",
        },
    )
