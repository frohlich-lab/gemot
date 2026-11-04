"""Native CLI runs keep preflight, judging and Phoenix without constructing a panel.

CLI configuration is checked before a previous work directory can be replaced. The runner
then enters one native session under its ordinary root trace and judges the committed model
through the same independent judge as the other drivers.
"""

from __future__ import annotations

import asyncio
import importlib
import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from gemot import agent, models, tracing
from gemot.runner import cli as C
from gemot.runner import drive as D
from gemot.runner import provenance as P
from gemot.scenario import Scenario


_DRIVERS = ("codex", "claude")


def _scenario(tmp_path):
    async def unused_fix(*args):
        raise AssertionError("a CLI run must not use the scripted repair")

    return Scenario("toy", tmp_path / "problem", unused_fix)


def _transport(driver):
    return importlib.import_module(f"gemot.agent.{driver}")


@pytest.mark.parametrize("driver", _DRIVERS)
def test_cli_keeps_raw_model_and_checks_before_pipeline(monkeypatch, tmp_path, driver):
    seen = []

    def check(executable):
        seen.append(("check", executable))
        return "cli-test-version"

    async def main(scenario, args):
        seen.append(("run", args))
        return 0, {}

    monkeypatch.setattr(_transport(driver), f"check_{driver}_cli", check)
    monkeypatch.setattr(C, "_main_async", main)
    # A known OpenRouter alias must pass unchanged to the selected native CLI.
    assert C.run_scenario(_scenario(tmp_path), ["--driver", driver, "--model", "luna"]) == 0
    assert seen[0] == ("check", driver)
    args = seen[1][1]
    assert args.model == "luna" and args.reasoning_effort is None
    assert getattr(args, f"{driver}_version") == "cli-test-version"
    assert getattr(args, f"{driver}_timeout") is None
    assert not hasattr(args, f"{driver}_max_input_tokens")
    assert args.phoenix is True


@pytest.mark.parametrize("driver", _DRIVERS)
@pytest.mark.parametrize(
    "options",
    [
        [],
        ["--model", "native-model", "--api-key", "wrong-transport-key"],
        ["--model", "native-model", "--biocurator-model", "another-model"],
        ["--model", "native-model", "--{driver}-timeout", "0"],
        ["--model", "native-model", "--{driver}-timeout", "nan"],
        ["--model", "native-model", "--{driver}-timeout", "inf"],
        ["--model", "native-model", "--max-wall-seconds", "0"],
        ["--model", "native-model", "--max-wall-seconds", "nan"],
        ["--model", "native-model", "--max-wall-seconds", "inf"],
        ["--model", "native-model", "--{driver}-max-input-tokens", "200000"],
    ],
)
def test_invalid_native_configuration_stops_before_cleanup(monkeypatch, tmp_path, driver, options):
    def forbidden(*args):
        pytest.fail("invalid native settings reached preflight or workspace cleanup")

    monkeypatch.setattr(C, "_main_async", forbidden)
    monkeypatch.setattr(_transport(driver), f"check_{driver}_cli", forbidden)
    options = [option.replace("{driver}", driver) for option in options]
    with pytest.raises(SystemExit) as error:
        C.run_scenario(_scenario(tmp_path), ["--driver", driver, *options])
    assert error.value.code == 2


@pytest.mark.parametrize(
    "driver, effort", [("codex", "max"), ("claude", "none"), ("claude", "minimal")]
)
def test_unsupported_effort_stops_before_start(monkeypatch, tmp_path, driver, effort):
    def forbidden(*args):
        pytest.fail("unsupported effort reached the runner")

    monkeypatch.setattr(C, "_main_async", forbidden)
    with pytest.raises(SystemExit) as error:
        C.run_scenario(
            _scenario(tmp_path),
            ["--driver", driver, "--model", "native-model", "--reasoning-effort", effort],
        )
    assert error.value.code == 2


@pytest.mark.parametrize("driver", _DRIVERS)
def test_missing_cli_preserves_previous_work(monkeypatch, tmp_path, driver):
    previous = tmp_path / "work" / "result.json"
    previous.parent.mkdir()
    previous.write_text('{"success": true}')

    def missing(executable):
        raise RuntimeError("CLI needs upgrading")

    async def forbidden(*args):
        pytest.fail("a failed CLI preflight entered workspace cleanup")

    monkeypatch.setattr(_transport(driver), f"check_{driver}_cli", missing)
    monkeypatch.setattr(C, "_main_async", forbidden)
    with pytest.raises(SystemExit) as error:
        C.run_scenario(_scenario(tmp_path), ["--driver", driver, "--model", "native-model"])
    assert error.value.code == 2
    assert json.loads(previous.read_text()) == {"success": True}


@pytest.mark.parametrize("driver", ["deepagents", "barebones", "scripted"])
def test_existing_drivers_keep_their_defaults(monkeypatch, tmp_path, driver):
    seen = []

    async def main(scenario, args):
        seen.append(args)
        return 0, {}

    monkeypatch.setattr(C, "_main_async", main)
    if driver == "barebones":
        from gemot.runner import barebones

        monkeypatch.setattr(barebones, "_main_async_barebones", main)
    assert C.run_scenario(_scenario(tmp_path), ["--driver", driver]) == 0
    assert seen[0].model == models.resolve(models.DEFAULT)
    assert seen[0].reasoning_effort == "max"


@pytest.mark.parametrize("driver", _DRIVERS)
def test_native_runner_records_phoenix_and_judges_without_panel(monkeypatch, tmp_path, driver):
    from gemot.agent import native_cli

    scenario = _scenario(tmp_path)
    args = C.build_parser(scenario).parse_args(
        ["--driver", driver, "--model", "native-model", "--reasoning-effort", "high"]
    )
    setattr(args, f"{driver}_version", "cli-test-version")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(P, "_git_sha", lambda *a: "git-test")
    monkeypatch.setattr(P, "_git_diff_digest", lambda *a: None)
    monkeypatch.setattr(P, "_imported_provenance", lambda *a: None)
    monkeypatch.setattr(P, "_fit_settings", lambda *a: {})
    monkeypatch.setattr(P, "_omnipath_provenance", lambda: None)
    seen = []
    span = object()
    handle = SimpleNamespace(
        project_name=args.phoenix_project,
        ui_url="http://collector.test",
        flush=lambda: seen.append("flush"),
    )

    def setup(*, project_name, instrument_langchain):
        assert instrument_langchain is False
        assert project_name == args.phoenix_project
        seen.append("setup")
        return handle

    @contextmanager
    def root_span(tracing_handle, *, name, attributes):
        assert tracing_handle is handle and seen == ["setup"]
        assert name == "model_build_run"
        assert attributes["driver"] == driver and attributes["temperature"] is None
        seen.append("span")
        yield span

    def trace_id(value):
        assert value is span
        return "a" * 32

    async def native_run(session, **kwargs):
        assert seen == ["setup", "span"]
        assert kwargs["tracing"] is handle
        assert kwargs["parent_span"] is span
        assert kwargs["driver"] == driver and kwargs["model"] == "native-model"
        assert kwargs["reasoning_effort"] == "high"
        assert "api_key" not in kwargs and "base_url" not in kwargs
        seen.append("native")
        return {
            "stop_reason": "completed",
            "native_final_text": "model committed",
            "token_usage": {"total_tokens": 123},
        }

    async def no_panel(*args, **kwargs):
        pytest.fail("native CLI driver constructed a deepagents panel")

    async def judge(scenario, call, baseline, work, meta, clock, t0):
        seen.append("judge")
        result = {**meta, "success": True}
        (work / "result.json").write_text(json.dumps(result))
        return 0, result

    def record(value, result):
        assert value is span and result["success"]
        assert result["phoenix_trace_id"] == "a" * 32
        seen.append("record")

    async def call(name, arguments=None, **kwargs):
        replies = {"describe_problem": {"model_authored": False}, "get_model_structure": {}}
        return SimpleNamespace(structured_content=replies.get(name, {}))

    monkeypatch.setattr(tracing, "setup_phoenix_tracing", setup)
    monkeypatch.setattr(tracing, "run_span", root_span)
    monkeypatch.setattr(tracing, "trace_id_hex", trace_id)
    monkeypatch.setattr(tracing, "record_run_result", record)
    monkeypatch.setattr(native_cli, "run_native_cli", native_run)
    monkeypatch.setattr(agent, "run_deepagents", no_panel)
    monkeypatch.setattr(D, "_judge", judge)
    monkeypatch.setattr(D, "_write_live_marker", lambda *a: None)
    code, result = asyncio.run(D._drive(scenario, SimpleNamespace(call_tool=call), tmp_path, args))
    assert code == 0 and seen == ["setup", "span", "native", "judge", "record", "flush"]
    assert result == json.loads((tmp_path / "result.json").read_text())
    assert result["phoenix_project"] == args.phoenix_project
    assert result["phoenix_trace_id"] == "a" * 32
    assert result["model_transport"] == f"{driver}_cli"
    assert result[f"{driver}_executable"] == driver
    assert result[f"{driver}_version"] == "cli-test-version"
    assert result[f"{driver}_timeout"] is None
    assert result["temperature"] is None and result["recursion_limit"] is None
    assert "max_input_tokens" not in result
    assert result["native_final_text"] == "model committed"
