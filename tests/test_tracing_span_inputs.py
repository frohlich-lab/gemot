"""The wrapper-span input stripping in ``tracing.py``.

Two halves. The first pins the span *names* against the installed LangChain / deepagents /
our own guard: the stripping is name-based, so a rename upstream would silently stop the
saving and let the collector DB go back to storing the agent transcript four times per step.
The second checks the strip itself removes only ``input.*``, and only on those spans.
"""

import pytest

# Phoenix/OTel live in the `harness` extra, NOT in `dev` (pyproject: dev -> petab-mcp[agent]),
# and the packaging comment is explicit that no test may require them. Skip, don't error.
# Guard the SUBMODULE, not the top-level name: `opentelemetry` is a namespace package that
# imports cleanly even when nothing is installed under it, so importorskip("opentelemetry")
# passes and the real import then fails at collection -- which is how CI broke.
pytest.importorskip("opentelemetry.sdk.trace", reason="harness-only dependency")

from opentelemetry.sdk.trace import ReadableSpan  # noqa: E402
from opentelemetry.sdk.trace.export import SpanExportResult  # noqa: E402

from gemot.tracing import (  # noqa: E402
    _STRIPPED_INPUT_SPANS,
    _make_stripping_exporter,
    _strip_span_inputs,
)


# --------------------------------------------------------------------------- name pinning


def _real_graph_nodes():
    """Nodes of the graph the DRIVER builds — `create_deep_agent`, not bare `create_agent`.

    Building the wrong constructor was the flaw in the first version of these tests: they
    asserted against a graph that nothing in production runs, so `create_deep_agent`'s own
    middleware could add an unstripped per-step wrapper and every test would still pass.
    """
    from deepagents import create_deep_agent
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

    from gemot.agent.guards import _tool_dispatch_guard

    import inspect

    params = inspect.signature(create_deep_agent).parameters
    kwargs = {"model": GenericFakeChatModel(messages=iter([])), "tools": []}
    if "middleware" in params:
        kwargs["middleware"] = [_tool_dispatch_guard({})]
    agent = create_deep_agent(**kwargs)
    return set(agent.get_graph().nodes)


# Per-step nodes deliberately NOT stripped, each for its own reason:
#   tools                              -- input.value is the tool call being dispatched (~0.5 KB)
#   PatchToolCallsMiddleware.before_agent -- a `before_agent` hook: fires once per agent
#       invocation, not once per step, so it cannot grow as O(steps^2). Measured at 87 KB
#       average / 0.12 GB across the whole July corpus, against 27.1 GB for the three stripped.
_DELIBERATELY_KEPT = {"tools", "PatchToolCallsMiddleware.before_agent"}


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
    for name in ("ChatOpenAI", "set_model", "task", "model_build_run", "biocurator"):
        span = _span(name, {"input.value": "keep me", "input.mime_type": "application/json"})
        assert _strip_span_inputs(span) is span, f"{name} must not be stripped"
        assert span.attributes["input.value"] == "keep me"


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
    That behaviour hangs off a private `_default_processor` flag, so pin it here.
    """
    pytest.importorskip("phoenix.otel", reason="harness-only dependency")
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
