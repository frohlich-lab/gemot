"""The ablation registry is only authoritative if nothing can bypass it.

Two classes of test here. The first pins the registry's own behaviour (resolution, `arm()`, the
printed catalogue). The second is the drift guard: it greps ``src/`` for environment variables that
gate behaviour and fails if one is neither registered nor explicitly declared non-ablation. Without
it, "the single entrypoint" holds only for as long as everyone remembers to update the registry.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import SERVER_SRC
from gemot import ablations

SRC = Path(__file__).resolve().parents[1] / "src"


def _modules():
    """``(label, path)`` for every module of this harness and of the installed server, which reads
    switches of its own."""
    for package in (SRC / "gemot", SERVER_SRC):
        for path in package.rglob("*.py"):
            yield str(path.relative_to(package.parent)), path


# --------------------------------------------------------------------------- the registry itself
def test_every_ablation_defaults_to_the_baseline_arm(monkeypatch):
    """With no ablation variable set, every switch reads its declared default and arm() is clean."""
    for a in ablations.ABLATIONS:
        monkeypatch.delenv(a.env, raising=False)
    assert ablations.state() == {a.name: a.default_on for a in ablations.ABLATIONS}
    assert ablations.arm() == "baseline"


@pytest.mark.parametrize("spelling", ["0", "false", "FALSE", "no", "off", " Off "])
def test_off_words_turn_a_default_on_switch_off(monkeypatch, spelling):
    monkeypatch.setenv("PETAB_DATA_REGIMES", spelling)
    assert ablations.enabled("data_regimes") is False


@pytest.mark.parametrize("spelling", ["1", "true", "yes", "on", "anything"])
def test_non_off_words_leave_a_default_on_switch_on(monkeypatch, spelling):
    monkeypatch.setenv("PETAB_DATA_REGIMES", spelling)
    assert ablations.enabled("data_regimes") is True


def test_the_printed_setting_actually_flips_each_arm(monkeypatch):
    """Execute the catalogue's own instructions.

    `describe()` tells the reader to export `a.setting`, and nothing else checks that the string
    works. It is easy to get wrong, because the value that flips an arm is not uniform: a switch
    whose variable names a PATH is disabled by the empty string, and `=0` would set it to a
    directory called "0" while leaving it on. A catalogue that documents a no-op is worse than
    none, so: parse each printed setting, apply it, and assert the arm moved off its default.
    """
    for a in ablations.ABLATIONS:
        var, _, value = a.setting.partition("=")
        assert var == a.env, f"{a.name}: setting names {var}, not its env var {a.env}"
        monkeypatch.setenv(var, value)
        assert a.enabled() is (not a.default_on), (
            f"{a.name}: the documented setting `{a.setting}` did not flip it"
        )
        assert not a.is_default()
        assert f"{a.name}=" in ablations.arm()
        monkeypatch.delenv(var)


def test_arm_names_only_the_deviations(monkeypatch):
    """A new ablation left at its default must not rename existing arms, or old runs stop being
    comparable to new ones the moment a switch is added."""
    monkeypatch.delenv("PETAB_BIO_LITERATURE", raising=False)
    monkeypatch.delenv("PETAB_DATA_REGIMES", raising=False)
    assert ablations.arm() == "baseline"  # both at their defaults
    monkeypatch.setenv("PETAB_DATA_REGIMES", "1")
    assert ablations.arm() == "data_regimes=on"
    monkeypatch.setenv("PETAB_BIO_LITERATURE", "1")
    assert ablations.arm() == "data_regimes=on,bio_literature=on"


def test_unknown_ablation_raises_rather_than_reading_as_off():
    """A typo must not silently select the baseline — that is an A/B whose halves are identical."""
    with pytest.raises(KeyError, match="unknown ablation"):
        ablations.enabled("data_regime")  # missing 's'


def test_state_records_every_switch_not_only_the_deviations(monkeypatch):
    """A run silent about a switch is indistinguishable from one predating it."""
    monkeypatch.delenv("PETAB_DATA_REGIMES", raising=False)
    monkeypatch.delenv("PETAB_BIO_LITERATURE", raising=False)
    assert set(ablations.state()) == {a.name for a in ablations.ABLATIONS}


def test_cli_lists_every_arm_and_json_round_trips(capsys):
    assert ablations._main([]) == 0
    text = capsys.readouterr().out
    for a in ablations.ABLATIONS:
        assert a.name in text and a.setting in text and a.effect[:40] in text

    assert ablations._main(["--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["arm"] == ablations.arm()
    assert [x["name"] for x in payload["ablations"]] == [a.name for a in ablations.ABLATIONS]


def test_module_runs_as_a_script():
    """`python -m gemot.ablations` is the advertised entrypoint, so run it as one."""
    out = subprocess.run(
        [sys.executable, "-m", "gemot.ablations"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Ablations" in out.stdout and "data_regimes" in out.stdout


# --------------------------------------------------------------------------- the drift guard
# Environment variables read in `src/` that are NOT ablations. Each one must be here for a stated
# reason, so that adding a genuine arm without registering it fails this test rather than passing
# unnoticed. Paths, credentials and telemetry endpoints change WHERE things go, not what the agent
# does; the bio cache and OmniPath snapshot are plumbing (see the ablations module docstring).
_NOT_ABLATIONS = {
    "AMICI_LOG": "AMICI log level, not a behavioural switch",
    "BENCHMARK_MODELS_PETAB": "path to the benchmark collection",
    "BENCHMARK_MODELS_PETAB_CACHE": "path to the benchmark download cache",
    "OPENROUTER_API_KEY": "credential",
    "OTEL_EXPORTER_OTLP_ENDPOINT": "telemetry endpoint",
    "PATH": "the shell's own",
    "PETAB_BIO_CACHE_DIR": "plumbing: where a bio answer is read from, not what it says",
    "PETAB_MCP_BUILD_ROOT": "path for AMICI build artefacts",
    "PETAB_MCP_CRITERION": "the complexity penalty the server ranks models by "
    "(bic/caic/aic/aicc/hqc/tic). "
    "By the membership test above this IS an arm — it changes what answer "
    "the run prefers — but `Ablation` is binary (`default_on`/`enabled()`, "
    "`state() -> dict[str, bool]`, `arm()` off on/off) and cannot express a "
    "four-valued switch. It is RECORDED like one meanwhile: `criterion` "
    "rides on every fit result and lands in result.json, the metrics CSV, "
    "the W&B summary and the run span, so two criterion arms never pool. "
    "What it lacks is the catalogue entry, which needs a multi-valued "
    "Ablation — register it here the moment the registry grows one",
    "PETAB_MCP_PROVENANCE_DIR": "path for provenance logs",
    "PETAB_MCP_MAX_INPUT_TOKENS": "infrastructure: the model's real context window, fed to the "
    "summarization middleware as a profile so its trigger is a "
    "fraction of the true window instead of a flat 170k fallback — "
    "changes when compaction fires, not what any answer is",
    "PETAB_MCP_MAX_OUTPUT_TOKENS": "infrastructure, the output-side twin of MAX_INPUT_TOKENS: the "
    "model's real max_completion_tokens, sent so a turn gets the "
    "ceiling the model advertises rather than whatever default the "
    "provider substitutes (Z.AI serves glm-5.3-flash at 65536, half "
    "its stated 131072). Declared here rather than registered as an "
    "arm because it does not change the task, the prompt, the tools "
    "or the scoring — only how much room one reply has. Stated "
    "plainly though: it CAN move an outcome, because a reply "
    "truncated at the provider default emits nothing at all "
    "(finish_reason 'length', no tool call). That is a capability "
    "ceiling being set correctly, not a behavioural knob to A/B",
    "PETAB_MCP_REFERENCE_VALIDATION": "server validation mode; also a run.py flag, not an arm",
    "PETAB_MCP_REPO_ROOT": "path override for repo_root()",
    "PETAB_MCP_SCRATCH": "where array_task.sh stages a task's working directory; read back by "
    "analysis.default_work_root() to find it again. Read-side only — by the "
    "time anything reads it here the run is over",
    "PETAB_MCP_SKILLS_DIR": "path to the skills directory",
    "PETAB_MCP_WORK_ROOT": "the shared-server refactor's name for PETAB_MCP_SCRATCH; same path, "
    "same read-side-only role",
    "PETAB_MCP_YAML": "path to the PEtab problem",
    "PETAB_REQUIRE_COLLECTOR": "refuse to fall back to a throwaway in-process Phoenix when no "
    "shared collector is configured — a safety interlock on the "
    "RECORD of a run, not on what the agent does, so it cannot move "
    "a result (a run either happens traced or does not happen)",
    "PHOENIX_COLLECTOR_ENDPOINT": "telemetry endpoint",
    "PHOENIX_PROJECT_NAME": "telemetry project name",
    "WANDB_PROJECT": "telemetry project name",
}

_ENV_READ = re.compile(r"""(?:environ\.get|getenv)\(\s*["']([A-Z][A-Z0-9_]*)["']""")


def test_no_unregistered_environment_switch_gates_behaviour():
    """Every env var read in either package is either a registered ablation or declared
    non-ablation.

    This is what makes `python -m gemot.ablations` an authoritative catalogue rather than
    a list someone remembered to update: an arm can only stay undiscoverable for as long as it
    takes this test to run.
    """
    registered = {a.env for a in ablations.ABLATIONS}
    found: dict[str, str] = {}
    for label, path in _modules():
        for var in _ENV_READ.findall(path.read_text(encoding="utf-8", errors="replace")):
            found.setdefault(var, label)

    unaccounted = {
        v: where for v, where in found.items() if v not in registered and v not in _NOT_ABLATIONS
    }
    assert not unaccounted, (
        "environment variables read in either package that are neither registered in "
        "gemot.ablations nor listed in _NOT_ABLATIONS:\n"
        + "\n".join(f"  {v}  ({where})" for v, where in sorted(unaccounted.items()))
        + "\nIf it changes what a run DOES, register it as an Ablation so it is discoverable and "
        "lands in result.json. If not, add it to _NOT_ABLATIONS with the reason."
    )


@pytest.mark.parametrize(
    "spelling",
    ["0", "false", "FALSE", "no", "off", " Off ", "1", "true", "yes", "on", "anything", None],
)
def test_the_server_gate_agrees_with_the_registry(monkeypatch, spelling):
    """A duplicated predicate must agree with the registry on every spelling.

    ``petab_mcp.session`` gates its own data-regimes advice by reading ``PETAB_DATA_REGIMES``
    itself, because the server cannot import the harness (wrong direction) and the harness's
    ``prompts`` cannot import the server (import-lightness). If the two ever disagreed the result
    is a HALF-ABLATED run: the tool removed from the toolset while the server still tells the agent
    to call it — a wasted turn on an unknown-tool error and, worse, a contaminated control. Rather
    than trust the comment, run both.
    """
    if spelling is None:
        monkeypatch.delenv("PETAB_DATA_REGIMES", raising=False)
    else:
        monkeypatch.setenv("PETAB_DATA_REGIMES", spelling)

    # A FRESH interpreter each time: both gates are module-level constants, so an already-imported
    # copy would answer for whatever the environment was on first import.
    probe = (
        "import os,sys;"
        "sys.path.insert(0, %r);"
        "from gemot.ablations import enabled;"
        "from petab_mcp.session import _REGIMES_ADVICE;"
        "print(int(enabled('data_regimes')), int(bool(_REGIMES_ADVICE)))" % str(SRC)
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    registry, server = out.stdout.split()
    assert registry == server, (
        f"PETAB_DATA_REGIMES={spelling!r}: registry says {registry}, "
        f"petab_mcp.session says {server} — the duplicated predicate has drifted"
    )


@pytest.mark.parametrize(
    "spelling",
    ["0", "false", "FALSE", "no", "off", " Off ", "1", "true", "yes", "on", "anything", None],
)
def test_the_diff_edits_server_gate_agrees_with_the_registry(monkeypatch, spelling):
    """Same duplicated-predicate rule for the OTHER server-read arm: ``petab_mcp.server``
    reads PETAB_DIFF_EDITS itself (registration-time, to pick the make_changes schema), so the
    two spellings-of-truth must agree or a run is half-ablated — the prompt promising `edits`
    while the schema refuses it, or vice versa."""
    if spelling is None:
        monkeypatch.delenv("PETAB_DIFF_EDITS", raising=False)
    else:
        monkeypatch.setenv("PETAB_DIFF_EDITS", spelling)

    # A fresh interpreter, same as above: the registry side is read through module-level
    # constants elsewhere, and the probe must answer for THIS environment.
    probe = (
        "import os,sys;"
        "sys.path.insert(0, %r);"
        "from gemot.ablations import enabled;"
        "from petab_mcp.server import _diff_edits_enabled;"
        "print(int(enabled('diff_edits')), int(_diff_edits_enabled()))" % str(SRC)
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    registry, server = out.stdout.split()
    assert registry == server, (
        f"PETAB_DIFF_EDITS={spelling!r}: registry says {registry}, "
        f"petab_mcp.server says {server} — the duplicated predicate has drifted"
    )


def test_registered_ablations_are_actually_read_somewhere():
    """The converse: a registered arm that nothing reads is a lie in the catalogue."""
    text = "\n".join(
        p.read_text(encoding="utf-8", errors="replace")
        for _, p in _modules()
        if p.name != "ablations.py"
    )
    for a in ablations.ABLATIONS:
        assert f'enabled("{a.name}")' in text or a.env in text, (
            f"{a.name} is registered but never read outside the registry"
        )


def test_the_removed_bio_literature_flag_stays_removed():
    """`--bio-literature` was an arm settable only from the command line and recorded nowhere.

    It is now `PETAB_BIO_LITERATURE`. Re-adding the flag would restore two ways to set one arm,
    which is how it gets recorded from the wrong one.
    """
    assert (
        'add_argument("--bio-literature"' not in (SRC / "gemot" / "runner" / "cli.py").read_text()
    )
