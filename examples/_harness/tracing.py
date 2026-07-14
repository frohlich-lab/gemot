"""Arize Phoenix tracing for the deepagents driver.

`Phoenix <https://phoenix.arize.com>`_ is an open-source LLM-observability tool. This
module wires OpenTelemetry tracing into the LangChain / LangGraph stack that
``deepagents`` runs on, so every LLM call and every MCP tool call the agent makes is
captured as a span and is inspectable in the Phoenix UI (prompts, completions, tool
arguments, token counts, latency, and the full agent trace tree).

Tracing is **on by default** — the Phoenix packages are declared dependencies of the
deepagents driver — and can be turned off with ``run.py --no-phoenix``. The function
still degrades gracefully (a clear message, never a crash) if the tracing packages happen
to be missing.

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
import json
import os
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
    try:
        from phoenix.otel import register
    except ImportError:
        print(
            "[phoenix] `arize-phoenix-otel` is not installed; running without tracing.\n"
            "  Install the runner requirements, or:\n"
            "    pip install arize-phoenix arize-phoenix-otel openinference-instrumentation-langchain\n"
            "  (or pass --no-phoenix to silence this)."
        )
        return None

    try:
        from openinference.instrumentation.langchain import LangChainInstrumentor
    except ImportError:
        print(
            "[phoenix] `openinference-instrumentation-langchain` is not installed; "
            "running without tracing. Install with:\n"
            "    pip install openinference-instrumentation-langchain"
        )
        return None

    # Only auto-launch a local app when the user didn't point us at an external collector.
    explicit_endpoint = endpoint or os.environ.get("PHOENIX_COLLECTOR_ENDPOINT") \
        or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
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
            print(f"[phoenix] failed to launch local Phoenix app ({exc}); "
                  "running without tracing.")
            return None

    # Normalize to the collector base so we append the OTLP path exactly once, whether the
    # user gave us a bare host or a full ``/v1/traces`` URL.
    base_endpoint = _base_endpoint(endpoint or _default_endpoint())
    traces_endpoint = f"{base_endpoint}/v1/traces"

    # `register` wires up the OTel tracer provider + OTLP exporter and sets it as global.
    # Different arize-phoenix-otel releases expose slightly different kwargs, so fall back
    # to a minimal call if the richer one is rejected. The fallback preserves the target by
    # exporting the base endpoint through the env var `register` reads, so we never silently
    # export to localhost when the user configured a remote collector.
    try:
        tracer_provider = register(
            project_name=project_name,
            endpoint=traces_endpoint,
            set_global_tracer_provider=True,
        )
    except TypeError:
        os.environ.setdefault("PHOENIX_COLLECTOR_ENDPOINT", base_endpoint)
        tracer_provider = register(project_name=project_name)

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
    "scenario", "example_version", "model", "driver", "stop_reason", "success",
    "baseline_nll", "final_nll", "delta",
    "chi2", "chi2_dof", "chi2_gof_pvalue",
    "k", "n", "bic", "bic_reference", "delta_bic", "bic_threshold",
    "head_version", "n_errors", "tool_calls",
    "wall_seconds", "compute_seconds", "llm_seconds",
    "recursion_limit", "n_starts", "temperature", "git_sha",
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
        _set_attrs(span, {
            "input.value": json.dumps(attributes, default=str),
            "input.mime_type": "application/json",
        })
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
    _set_attrs(span, {"petab.verdict": "success" if result.get("success") else "not_fixed"})
    with contextlib.suppress(Exception):
        span.set_attribute("metadata", json.dumps(result, default=str))
    outcome = {k: result.get(k) for k in
               ("success", "chi2", "chi2_gof_pvalue", "delta_bic", "bic",
                "final_nll", "stop_reason")}
    _set_attrs(span, {
        "output.value": json.dumps(outcome, default=str),
        "output.mime_type": "application/json",
    })
