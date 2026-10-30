"""The span input stripping in ``tracing.py``.

Four parts. The first pins the wrapper span *names* against the installed LangChain /
deepagents / our own guards: that stripping is name-based, so a rename upstream would silently
stop the saving and let the collector DB go back to storing the agent transcript once per hook
per step. The second checks the strip itself removes only ``input.*``, and only on those spans
and on LLM spans. The third checks it is installed at all on the Phoenix provider ``register``
returns. The fourth checks that an LLM span's input, written as a delta on its conversation's
previous call, is rebuilt exactly, on the real graph and through everything that rewrites a
request.

No ``importorskip``: ``dev`` installs the ``tracing`` extra. While these tests skipped in CI the
third part could not fail, and it was the one that broke.
"""

import json
import re

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExportResult
from opentelemetry.trace import SpanContext

from gemot.tracing import (
    _FULL_INPUT_EVERY,
    _PREFIX_LEN,
    _PREFIX_SPAN,
    _STRIPPED_INPUT_SPANS,
    _make_stripping_exporter,
    _strip_span_inputs,
    rebuild_input_messages,
    rebuild_input_messages_from_dataframe,
)


# --------------------------------------------------------------------------- name pinning


def _real_graph_nodes():
    """Nodes of the graphs the DRIVER builds — `create_deep_agent`, not bare `create_agent` —
    with the lead's middleware and an analyst's.

    Building the wrong constructor was the flaw in the first version of these tests: they
    asserted against a graph that nothing in production runs, so `create_deep_agent`'s own
    middleware could add an unstripped per-step wrapper and every test would still pass. Leaving
    out the analysts' middleware was the second: `_SubagentTurnLimit.after_model` carried every
    analyst transcript, unstripped, for as long as only the lead's graph was checked here.
    """
    from deepagents import create_deep_agent
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

    from gemot.agent.guards import _subagent_turn_limit, _tool_dispatch_guard

    import inspect

    params = inspect.signature(create_deep_agent).parameters
    kwargs = {"model": GenericFakeChatModel(messages=iter([])), "tools": []}
    if "middleware" in params:
        kwargs["middleware"] = [_tool_dispatch_guard({}), _subagent_turn_limit(3)]
    agent = create_deep_agent(**kwargs)
    return set(agent.get_graph().nodes)


# Per-step nodes deliberately NOT stripped, each for its own reason:
#   tools                              -- input.value is the tool call being dispatched (~0.5 KB)
#   PatchToolCallsMiddleware.before_agent -- a `before_agent` hook: fires once per agent
#       invocation, not once per step, so it cannot grow as O(steps^2). Measured at 87 KB
#       average / 0.12 GB across the whole July corpus, against 27.1 GB for the three stripped.
#   _ParentNotifierMiddleware.after_agent -- deepagents 0.7.27's, on every deep agent: an
#       `after_agent` hook, so once per invocation for the same reason, and deepagents already
#       records its input as `{}` (`TracePolicy(process_inputs=omit_payload)`).
_DELIBERATELY_KEPT = {
    "tools",
    "PatchToolCallsMiddleware.before_agent",
    "_ParentNotifierMiddleware.after_agent",
}


def test_the_real_graph_has_no_unstripped_per_step_wrapper():
    """A new per-step wrapper node must be a conscious decision, not a silent regression."""
    nodes = _real_graph_nodes() - {"__start__", "__end__"}
    unaccounted = nodes - _STRIPPED_INPUT_SPANS - _DELIBERATELY_KEPT
    assert not unaccounted, (
        f"unstripped nodes {sorted(unaccounted)} in the production graph: decide whether each "
        "carries a transcript copy, and either strip it or add it to _DELIBERATELY_KEPT with a "
        "reason"
    )


def test_the_stripped_names_still_match_real_nodes():
    """At least one stripped name must actually be a node the driver's graph emits.

    Not ALL of them: `TodoListMiddleware.after_model` is installed only under some deepagents
    harness profiles (19,007 such spans exist in the July corpus, none in the August one), so a
    stripped name that is currently absent is harmless. A stripped set matching NOTHING would
    mean the strip has silently become a no-op, which is the regression worth catching.
    """
    matched = _STRIPPED_INPUT_SPANS & _real_graph_nodes()
    assert matched, (
        f"none of {sorted(_STRIPPED_INPUT_SPANS)} is a node of the real graph "
        f"({sorted(_real_graph_nodes())}) -- the input stripping is a no-op"
    )


# ------------------------------------------------------------------------------ stripping


def _span(name, attributes):
    return ReadableSpan(name=name, attributes=attributes)


def test_strips_only_the_input_attributes():
    span = _span(
        "model",
        {
            "input.value": "x" * 100_000,
            "input.mime_type": "application/json",
            "output.value": "the useful bit",
            "llm.token_count.prompt": 1234,
        },
    )
    stripped = _strip_span_inputs(span)

    assert "input.value" not in stripped.attributes
    assert "input.mime_type" not in stripped.attributes
    # Everything the span genuinely owns survives.
    assert stripped.attributes["output.value"] == "the useful bit"
    assert stripped.attributes["llm.token_count.prompt"] == 1234
    assert stripped.name == "model"


def test_preserves_span_identity_and_timings():
    """The trace tree must be unchanged -- ChatOpenAI is a child of these spans."""
    span = ReadableSpan(
        name="_ToolDispatchGuard.after_model",
        attributes={"input.value": "big"},
        start_time=111,
        end_time=222,
    )
    stripped = _strip_span_inputs(span)

    assert stripped.parent == span.parent
    assert stripped.context == span.context
    assert stripped.kind == span.kind
    assert (stripped.start_time, stripped.end_time) == (111, 222)


def test_leaves_other_spans_untouched():
    """The spans both extract scripts and the explorer actually read."""
    for name, kind in (
        ("make_changes", "TOOL"),
        ("task", "TOOL"),
        ("model_build_run", "CHAIN"),
        ("biocurator", "CHAIN"),
    ):
        span = _span(
            name,
            {
                "openinference.span.kind": kind,
                "input.value": "keep me",
                "input.mime_type": "application/json",
            },
        )
        assert _strip_span_inputs(span) is span, f"{name} must not be stripped"
        assert span.attributes["input.value"] == "keep me"


def test_an_llm_span_keeps_its_messages_and_loses_their_serialized_copy():
    """`llm.input_messages` is the transcript the explorer and the Phoenix UI read; the
    `input.value` beside it was a second, larger copy (97 of 131 MB on the 2026-10-25 raimundez
    run's LLM spans), so only that goes."""
    span = _span(
        "ChatOpenAI",
        {
            "openinference.span.kind": "LLM",
            "input.value": '{"messages": [[...]]}',
            "input.mime_type": "application/json",
            "llm.input_messages.0.message.role": "system",
            "llm.input_messages.0.message.content": "You are the MODELING LEAD",
            "llm.output_messages.0.message.content": "done",
            "output.value": '{"generations": [[...]]}',
            "llm.token_count.prompt": 457970,
        },
    )
    stripped = _strip_span_inputs(span)

    assert "input.value" not in stripped.attributes
    assert "input.mime_type" not in stripped.attributes
    for key in (
        "openinference.span.kind",
        "llm.input_messages.0.message.role",
        "llm.input_messages.0.message.content",
        "llm.output_messages.0.message.content",
        "output.value",
        "llm.token_count.prompt",
    ):
        assert stripped.attributes[key] == span.attributes[key], key


def test_exporter_strips_in_flight_and_delegates():
    exported = []

    class _Recorder:
        def export(self, spans):
            exported.extend(spans)
            return SpanExportResult.SUCCESS

        def shutdown(self):
            exported.append("shutdown")

        def force_flush(self, timeout_millis=30_000):
            return True

    exporter = _make_stripping_exporter(_Recorder())
    result = exporter.export(
        [
            _span("model", {"input.value": "drop", "output.value": "keep"}),
            _span("ChatOpenAI", {"input.value": "keep"}),
        ]
    )

    assert result is SpanExportResult.SUCCESS
    assert "input.value" not in exported[0].attributes
    assert exported[0].attributes["output.value"] == "keep"
    assert exported[1].attributes["input.value"] == "keep"

    exporter.shutdown()
    assert exported[-1] == "shutdown"


# ---------------------------------------------------------------------------- wiring


def test_install_replaces_the_default_processor_rather_than_adding_to_it():
    """The load-bearing post-condition: exactly ONE processor afterwards.

    `_install_stripping_processor` relies on Phoenix's `add_span_processor` dropping the
    default processor it installed. If a release stops doing that, ours is appended instead
    and every span exports twice -- one copy unstripped, strictly worse than not stripping.
    That behaviour hangs off a private `_default_processor`, so pin it here.

    The `True` half matters as much: arize-phoenix-otel 0.17 turned that attribute from a flag
    into the processor, the install refused every provider from then on, and every October
    indra sweep traced its transcripts in full. This test saw it only once `dev` installed
    phoenix-otel; it had been skipping in CI.
    """
    from phoenix.otel import TracerProvider

    from gemot.tracing import _install_stripping_processor

    endpoint = "http://127.0.0.1:59999/v1/traces"
    provider = TracerProvider(endpoint=endpoint, verbose=False)
    assert len(provider._active_span_processor._span_processors) == 1

    assert _install_stripping_processor(provider, endpoint) is True

    processors = provider._active_span_processor._span_processors
    assert len(processors) == 1, "default processor survived -- spans would export twice"
    assert type(processors[0].span_exporter).__name__ == "_StrippingSpanExporter"


def test_install_declines_on_a_provider_it_cannot_replace_the_default_on():
    """A plain OTel provider must be refused, not appended to."""
    from opentelemetry.sdk.trace import TracerProvider as PlainTracerProvider

    from gemot.tracing import _install_stripping_processor

    provider = PlainTracerProvider()
    assert _install_stripping_processor(provider, "http://127.0.0.1:59999/v1/traces") is False
    assert len(provider._active_span_processor._span_processors) == 0


# --------------------------------------------------------------------------- input deltas

_INPUT_KEY = re.compile(r"llm\.input_messages\.(\d+)\.(.+)")
_SYSTEM, _TASK = ("system", "You are the MODELING LEAD"), ("user", "build a model")


def _messages(attributes):
    """A span's ``llm.input_messages`` as a list, one dict of attributes per message (OTel keeps
    them as flat keys)."""
    grouped = {}
    for key, value in attributes.items():
        if match := _INPUT_KEY.fullmatch(key):
            grouped.setdefault(int(match.group(1)), {})[match.group(2)] = value
    return [grouped[i] for i in sorted(grouped)]


def _id(span):
    return format(span.context.span_id, "016x")


def _stored(spans):
    """What each exported span stores, in the form `rebuild_input_messages` takes."""
    return {
        _id(s): (
            _messages(s.attributes),
            s.attributes.get(_PREFIX_LEN),
            s.attributes.get(_PREFIX_SPAN),
        )
        for s in spans
    }


def _llm_span(span_id, messages, *, agent="lead", namespace="model:1"):
    attributes = {
        "openinference.span.kind": "LLM",
        "metadata": json.dumps({"lc_agent_name": agent, "checkpoint_ns": namespace}),
        "output.value": "the reply",
    }
    for i, (role, content) in enumerate(messages):
        attributes[f"llm.input_messages.{i}.message.role"] = role
        attributes[f"llm.input_messages.{i}.message.content"] = content
    context = SpanContext(trace_id=1, span_id=span_id, is_remote=False)
    return ReadableSpan(name="ChatOpenAI", context=context, attributes=attributes)


class _Collector:
    """The wrapped exporter: keeps what reached it, and answers with ``results`` in turn (raising
    one that is an exception), then with SUCCESS."""

    def __init__(self, *results):
        self.spans, self._results = [], list(results)

    def export(self, spans):
        result = self._results.pop(0) if self._results else SpanExportResult.SUCCESS
        if isinstance(result, Exception):
            raise result
        if result == SpanExportResult.SUCCESS:
            self.spans.extend(spans)
        return result

    def shutdown(self):
        pass

    def force_flush(self, timeout_millis=30_000):
        return True


def _export(calls, collector=None):
    """Export one LLM span per ``(span_id, messages, kwargs)`` call, each as its own batch."""
    collector = collector or _Collector()
    exporter = _make_stripping_exporter(collector)
    for span_id, messages, kwargs in calls:
        try:
            exporter.export([_llm_span(span_id, messages, **kwargs)])
        except RuntimeError:
            pass
    return collector


def test_each_call_stores_only_what_it_adds_to_the_previous_one():
    """The O(steps^2) this removes: every call used to store the whole conversation again."""
    calls = [
        [_SYSTEM, _TASK],
        [_SYSTEM, _TASK, ("assistant", "a1"), ("tool", "t1")],
        [_SYSTEM, _TASK, ("assistant", "a1"), ("tool", "t1"), ("assistant", "a2"), ("tool", "t2")],
    ]
    spans = _export([(i, m, {}) for i, m in enumerate(calls, start=1)]).spans

    assert _PREFIX_LEN not in spans[0].attributes, "a conversation's first call is written in full"
    assert len(_messages(spans[0].attributes)) == 2
    assert (spans[1].attributes[_PREFIX_LEN], spans[1].attributes[_PREFIX_SPAN]) == (
        2,
        _id(spans[0]),
    )
    assert [m["message.content"] for m in _messages(spans[2].attributes)] == ["a2", "t2"]
    assert spans[2].attributes["output.value"] == "the reply", "outputs stay on every span"
    rebuilt = rebuild_input_messages(_stored(spans))
    assert list(rebuilt.values()) == [_messages(_llm_span(0, m).attributes) for m in calls]


def test_a_message_only_one_request_carried_is_kept_on_that_call():
    """The analysts' turn-limit notice goes into the outgoing request and never into state, so the
    next call's input does not extend this one's. Keeping only each conversation's last input
    would lose every notice but the last; a shorter prefix keeps them all."""
    history = [_SYSTEM, _TASK, ("assistant", "a1"), ("tool", "t1")]
    calls = [
        [*history, ("user", "2 of your 4 turns remain")],
        [*history, ("assistant", "a2"), ("tool", "t2"), ("user", "FINAL TURN")],
    ]
    spans = _export([(i, m, {}) for i, m in enumerate(calls, start=1)]).spans

    assert spans[1].attributes[_PREFIX_LEN] == 4
    rebuilt = rebuild_input_messages(_stored(spans))
    assert list(rebuilt.values()) == [_messages(_llm_span(0, m).attributes) for m in calls]


def test_a_call_without_a_system_prompt_keeps_its_full_input():
    """deepagents' summarizer runs under the lead's agent name and namespace but without a system
    prompt: it is one-shot, and must neither join the lead's chain nor break it. The compacted
    call after it shares only the system prompt with the lead's last one."""
    spans = _export(
        [
            (1, [_SYSTEM, _TASK, ("assistant", "a1")], {}),
            (2, [("user", "summarize: build a model, a1")], {}),
            (3, [_SYSTEM, ("user", "summary: a1"), ("assistant", "a2")], {}),
        ]
    ).spans

    assert _PREFIX_LEN not in spans[1].attributes
    assert (spans[2].attributes[_PREFIX_LEN], spans[2].attributes[_PREFIX_SPAN]) == (
        1,
        _id(spans[0]),
    )


def test_conversations_are_told_apart_by_agent_delegation_and_system_prompt():
    """Two delegations of one analyst are two conversations (``checkpoint_ns`` = ``tools:<id>``);
    the lead's steps are one, although its namespace changes every step."""
    math = ("system", "You are a mathematician")
    spans = _export(
        [
            (1, [_SYSTEM, _TASK], {"namespace": "model:1"}),
            (2, [math, ("user", "q1")], {"agent": "mathematician", "namespace": "tools:a"}),
            (3, [math, ("user", "q2")], {"agent": "mathematician", "namespace": "tools:b"}),
            (4, [_SYSTEM, _TASK, ("assistant", "a1")], {"namespace": "model:2"}),
            (
                5,
                [math, ("user", "q1"), ("assistant", "x")],
                {"agent": "mathematician", "namespace": "tools:a"},
            ),
        ]
    ).spans

    assert [s.attributes.get(_PREFIX_SPAN) for s in spans] == [
        None,
        None,
        None,
        _id(spans[0]),
        _id(spans[1]),
    ]


def test_the_full_input_is_written_again_every_few_calls():
    """Bounds what one span the collector lost silently can break: the calls up to the next full
    input, not the rest of the conversation."""
    calls, messages = [], [_SYSTEM, _TASK]
    for i in range(1, 2 * _FULL_INPUT_EVERY + 2):
        messages = [*messages, ("assistant", f"a{i}")]
        calls.append((i, messages, {}))
    spans = _export(calls).spans

    full = [i for i, s in enumerate(spans, start=1) if _PREFIX_LEN not in s.attributes]
    assert full == [1, _FULL_INPUT_EVERY + 1, 2 * _FULL_INPUT_EVERY + 1]
    assert rebuild_input_messages(_stored(spans))[_id(spans[-1])] == _messages(
        _llm_span(0, messages).attributes
    )


def test_a_failed_export_makes_every_conversation_start_over():
    """A span in a failed batch never reached the collector, so the next call must not point at it."""
    history = [_SYSTEM, _TASK]
    calls = [(i, history + [("assistant", f"a{j}") for j in range(i)], {}) for i in range(1, 6)]
    collector = _Collector(
        SpanExportResult.SUCCESS,
        SpanExportResult.FAILURE,
        SpanExportResult.SUCCESS,
        RuntimeError("collector unreachable"),
    )
    spans = _export(calls, collector).spans

    assert [int(_id(s), 16) for s in spans] == [1, 3, 5]
    assert [s.attributes.get(_PREFIX_SPAN) for s in spans] == [None, None, None]


def test_a_chain_with_a_missing_span_rebuilds_to_none_not_to_a_shorter_input():
    calls = [
        (i, [_SYSTEM, _TASK] + [("assistant", f"a{j}") for j in range(i)], {}) for i in range(1, 4)
    ]
    stored = _stored(_export(calls).spans)
    del stored[format(1, "016x")]

    assert rebuild_input_messages(stored) == {format(2, "016x"): None, format(3, "016x"): None}


def test_an_input_the_span_limit_truncated_is_left_as_it_is():
    """Past OTel's span attribute limit (10,000 under phoenix.otel, 128 on a plain OTel provider)
    the oldest attributes are dropped, so the input starts mid-list. Re-indexed, it would come back
    as a different list; written as it is, it stays what was recorded."""
    span = _llm_span(2, [_SYSTEM, _TASK, ("assistant", "a1")])
    truncated = {
        k: v for k, v in span.attributes.items() if not k.startswith("llm.input_messages.0.")
    }
    collector = _Collector()
    exporter = _make_stripping_exporter(collector)
    exporter.export([_llm_span(1, [_SYSTEM, _TASK])])
    exporter.export([ReadableSpan(name=span.name, context=span.context, attributes=truncated)])

    assert collector.spans[1].attributes == truncated


def _traced_run():
    """One run of the graph the driver builds -- ``create_deep_agent``, a lead and an analyst, the
    guards that rewrite a request or its result -- traced by the real OpenInference instrumentor,
    against a scripted model. Returns (the spans as recorded, the spans as exported).

    The script makes every case that turns an input into something other than an extension of the
    previous one happen at least once: a reasoning runaway the guard retries with the same input,
    a duplicated tool call the dispatch guard drops from state, the analyst's turn-limit notices,
    and compaction (the model's profile gives it a window the history overflows).
    """
    from deepagents import create_deep_agent
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    # OpenInference's provider, which phoenix.otel's subclasses: its 10,000-attribute span limit,
    # not OTel's default 128, which truncates a long input before the exporter sees it.
    from openinference.instrumentation import TracerProvider
    from openinference.instrumentation.langchain import LangChainInstrumentor
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from gemot.agent.guards import (
        _reasoning_runaway_guard,
        _subagent_turn_limit,
        _tool_dispatch_guard,
        new_guard_stats,
    )

    ids = iter(range(1000))

    def turn(name, args, copies=1):
        call = {"name": name, "args": args, "id": f"c{next(ids)}", "type": "tool_call"}
        return AIMessage("", tool_calls=[call] * copies)

    def todo(text, copies=1):
        todos = [{"content": f"{text} " + "x" * 6000, "status": "pending"}]
        return turn("write_todos", {"todos": todos}, copies)

    scripts = {
        "PERSONA-LEAD": [
            todo("plan"),
            turn(
                "task", {"description": "check identifiability", "subagent_type": "mathematician"}
            ),
            AIMessage("", response_metadata={"finish_reason": "length"}),
            todo("refine", copies=2),
            *[todo(f"step {i}") for i in range(10)],
            AIMessage("done"),
        ],
        "PERSONA-MATH": [turn("ls", {"path": "/"}) for _ in range(3)] + [AIMessage("report")],
    }

    class Scripted(BaseChatModel):
        profile: dict | None = {"max_input_tokens": 16_000}

        @property
        def _llm_type(self):
            return "scripted"

        def bind_tools(self, tools, **kwargs):
            return self

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            if not isinstance(messages[0], SystemMessage):
                reply = AIMessage("summary so far")  # deepagents' summarizer
            else:
                script = next(v for k, v in scripts.items() if k in messages[0].content)
                reply = script.pop(0) if script else AIMessage("done")
            return ChatResult(generations=[ChatGeneration(message=reply)])

    recorded, exported = InMemorySpanExporter(), InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(recorded))
    provider.add_span_processor(SimpleSpanProcessor(_make_stripping_exporter(exported)))
    stats = new_guard_stats()
    instrumentor = LangChainInstrumentor()
    instrumentor.instrument(tracer_provider=provider, skip_dep_check=True)
    try:
        agent = create_deep_agent(
            model=Scripted(),
            tools=[],
            system_prompt="PERSONA-LEAD",
            name="lead",
            subagents=[
                {
                    "name": "mathematician",
                    "description": "identifiability",
                    "system_prompt": "PERSONA-MATH",
                    "tools": [],
                    "middleware": [_subagent_turn_limit(4)],
                }
            ],
            middleware=[_reasoning_runaway_guard(stats), _tool_dispatch_guard(stats)],
        )
        agent.invoke({"messages": [HumanMessage("build a model")]}, config={"recursion_limit": 100})
    finally:
        instrumentor.uninstrument()
    assert stats["reasoning_runaways"] == 1 and stats["dropped_dup_tool_calls"] == 1
    return recorded.get_finished_spans(), exported.get_finished_spans()


def test_every_input_is_rebuilt_exactly_on_the_real_graph():
    """The round trip, on spans the real instrumentor recorded from the real graph: every LLM
    span's input is rebuilt exactly from what was exported, and the cases that keep a final
    conversation from being the whole record each occurred and survived."""

    def llm(spans):
        return [s for s in spans if s.attributes.get("openinference.span.kind") == "LLM"]

    recorded, exported = _traced_run()
    want = {_id(s): _messages(s.attributes) for s in llm(recorded)}
    stored = _stored(llm(exported))
    assert rebuild_input_messages(stored) == want

    def agent(span_id):
        span = next(s for s in recorded if _id(s) == span_id)
        return json.loads(span.attributes["metadata"])["lc_agent_name"]

    deltas = {sid: (n, prev) for sid, (_, n, prev) in stored.items() if prev is not None}
    assert deltas, "no span was written as a delta"
    assert all(agent(sid) == agent(prev) for sid, (_, prev) in deltas.items())
    one_shot = [sid for sid, m in want.items() if m[0]["message.role"] != "system"]
    assert one_shot and all(stored[sid][2] is None for sid in one_shot), "compaction's summarizer"
    shorter = {agent(sid) for sid, (n, prev) in deltas.items() if n < len(want[prev])}
    assert shorter == {"lead", "mathematician"}, "compaction, and the analyst's turn notices"
    assert any(not stored[sid][0] for sid in deltas), "the runaway retry resends the same input"

    def size(messages):
        return len(json.dumps(messages))

    assert sum(size(m) for m, _, _ in stored.values()) < 0.5 * sum(size(m) for m in want.values())


def test_a_phoenix_dataframe_is_read_with_its_prefix_nested_or_flat():
    """arize-phoenix 20.6 returns the two ``gemot.*`` attributes as one nested ``attributes.gemot``
    column, seen on a local collector. Read as flat columns, every span looked like it held its
    full input and the rebuild handed back the deltas as the inputs."""
    import pandas as pd

    system, task, a1 = (
        {"message.role": "system", "message.content": "lead"},
        {"message.role": "user", "message.content": "build a model"},
        {"message.role": "assistant", "message.content": "a1"},
    )
    rows = [
        ("a", [system, task], None, None),
        ("b", [a1], 2, "a"),
        ("c", float("nan"), 3, "b"),  # a retry: nothing new, so no input messages at all
    ]
    nested = pd.DataFrame(
        [
            {
                "context.span_id": span_id,
                "attributes.openinference.span.kind": "LLM",
                "attributes.llm.input_messages": messages,
                "attributes.gemot": None
                if n is None
                else {"input_prefix_len": n, "input_prefix_span": previous},
            }
            for span_id, messages, n, previous in rows
        ]
        + [{"context.span_id": "t", "attributes.openinference.span.kind": "TOOL"}]
    ).set_index("context.span_id")
    flat = nested.drop(columns="attributes.gemot").assign(
        **{
            "attributes.gemot.input_prefix_len": [n for _, _, n, _ in rows] + [None],
            "attributes.gemot.input_prefix_span": [p for _, _, _, p in rows] + [None],
        }
    )

    want = {"a": [system, task], "b": [system, task, a1], "c": [system, task, a1]}
    assert rebuild_input_messages_from_dataframe(nested) == want
    assert rebuild_input_messages_from_dataframe(flat) == want
