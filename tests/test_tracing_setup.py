"""Native tracing collects spans without requiring LangChain or bypassing batch gates."""

from __future__ import annotations

import builtins
import sys
from types import SimpleNamespace

import pytest

from gemot import tracing


@pytest.fixture
def setup_dependencies(monkeypatch):
    for key in (
        "PHOENIX_COLLECTOR_ENDPOINT",
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "PETAB_REQUIRE_COLLECTOR",
    ):
        monkeypatch.delenv(key, raising=False)
    calls = []
    provider = SimpleNamespace(force_flush=lambda: calls.append(("flush", {})))

    def register(**kwargs):
        calls.append(("register", kwargs))
        return provider

    class Instrumentor:
        def instrument(self, **kwargs):
            calls.append(("instrument", kwargs))

    def strip(tracer_provider, endpoint):
        calls.append(("strip", {"provider": tracer_provider, "endpoint": endpoint}))
        return True

    monkeypatch.setitem(sys.modules, "phoenix.otel", SimpleNamespace(register=register))
    monkeypatch.setitem(
        sys.modules,
        "openinference.instrumentation.langchain",
        SimpleNamespace(LangChainInstrumentor=Instrumentor),
    )
    monkeypatch.setattr(tracing, "_install_stripping_processor", strip)
    return calls, provider


def _block_import(monkeypatch, prefix):
    original = builtins.__import__
    attempts = []

    def import_module(name, *args, **kwargs):
        if name.startswith(prefix):
            attempts.append(name)
            raise ImportError(f"test intentionally excludes {name}")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_module)
    return attempts


def test_native_setup_never_imports_or_instruments_langchain(monkeypatch, setup_dependencies):
    calls, provider = setup_dependencies
    attempts = _block_import(monkeypatch, "openinference.instrumentation.langchain")
    monkeypatch.setenv("PETAB_REQUIRE_COLLECTOR", "1")
    handle = tracing.setup_phoenix_tracing(
        "native-project",
        endpoint="http://collector.example:6006/v1/traces",
        instrument_langchain=False,
    )
    assert attempts == []
    assert [name for name, _ in calls] == ["register"]
    assert calls[0][1]["endpoint"] == "http://collector.example:6006/v1/traces"
    assert calls[0][1]["batch"] is True
    assert handle.tracer_provider is provider
    assert handle.project_name == "native-project"
    handle.flush()
    assert calls[-1][0] == "flush"


def test_default_setup_keeps_panel_instrumentation(setup_dependencies):
    calls, provider = setup_dependencies
    handle = tracing.setup_phoenix_tracing(endpoint="http://collector.example:6006")
    assert handle.tracer_provider is provider
    assert [name for name, _ in calls] == ["register", "strip", "instrument"]
    assert calls[-1][1] == {"tracer_provider": provider, "skip_dep_check": True}


@pytest.mark.parametrize("required", [False, True])
def test_panel_missing_instrumentor_retains_existing_backstop(
    monkeypatch, setup_dependencies, required
):
    calls, _ = setup_dependencies
    attempts = _block_import(monkeypatch, "openinference.instrumentation.langchain")
    monkeypatch.setenv("PETAB_REQUIRE_COLLECTOR", "1" if required else "0")
    if required:
        with pytest.raises(RuntimeError, match="openinference-instrumentation-langchain"):
            tracing.setup_phoenix_tracing(endpoint="http://collector.example:6006")
    else:
        assert tracing.setup_phoenix_tracing(endpoint="http://collector.example:6006") is None
    assert len(attempts) == 1
    assert calls == []


@pytest.mark.parametrize("instrument_langchain", [False, True])
def test_required_external_collector_cannot_fall_back_to_local_app(
    monkeypatch, setup_dependencies, instrument_langchain
):
    calls, _ = setup_dependencies
    monkeypatch.setenv("PETAB_REQUIRE_COLLECTOR", "1")
    with pytest.raises(RuntimeError, match="no collector endpoint is configured"):
        tracing.setup_phoenix_tracing(instrument_langchain=instrument_langchain)
    assert calls == []


@pytest.mark.parametrize("required", [False, True])
def test_native_missing_phoenix_retains_package_backstop(monkeypatch, setup_dependencies, required):
    calls, _ = setup_dependencies
    attempts = _block_import(monkeypatch, "phoenix.otel")
    monkeypatch.setenv("PETAB_REQUIRE_COLLECTOR", "1" if required else "0")
    kwargs = {"endpoint": "http://collector.example:6006", "instrument_langchain": False}
    if required:
        with pytest.raises(RuntimeError, match="arize-phoenix-otel"):
            tracing.setup_phoenix_tracing(**kwargs)
    else:
        assert tracing.setup_phoenix_tracing(**kwargs) is None
    assert len(attempts) == 1
    assert calls == []
