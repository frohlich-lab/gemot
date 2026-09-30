"""The curator-only arm removes one seat without removing biology from the problem.

Prompts and the panel roster resolve at import, so exercise each configuration in a fresh
interpreter. The driver and runner checks use inert tools and a fake graph: no model calls,
remote biology services or numerical fits are needed to verify the ablation boundary.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from gemot.ablations import ABLATIONS
from test_run_loop_no_model import requires_harness

ROOT = Path(__file__).resolve().parents[1]


def _fresh(script: str, **switches: str) -> str:
    env = {**os.environ, **{a.env: str(int(a.default_on)) for a in ABLATIONS}, **switches}
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def test_only_curator_guidance_and_roster_change():
    script = r"""
import json
from pathlib import Path
from gemot import agent, prompts
from gemot.scenario import Scenario
scenario = Scenario("identified", Path("/example/problem"), lambda *args: None,
                    task="EGFR biology", known_constants="U2OS cells",
                    biocurator_entities=["EGFR"])
print(json.dumps({"roster": agent.SUBAGENT_NAMES, "prompt": scenario.system_prompt,
                  "task": scenario.agent_task, "analysts": agent.ANALYSTS,
                  "tools": sorted(agent.ALLOWED_TOOLS), "groups": prompts.prompt_group_versions(),
                  "enabled": prompts.BIOCURATOR_ENABLED}))
"""
    baseline = json.loads(_fresh(script))
    off = json.loads(_fresh(script, PETAB_BIOCURATOR="0"))
    assert baseline["enabled"] and not off["enabled"]
    assert baseline["roster"] == [*off["roster"], "biocurator"]
    assert off["roster"] == ["mathematician", "systems_scientist", "signal_engineer"]
    assert "three independent analysts" in off["prompt"]
    assert "biocurator" not in off["prompt"].lower()
    assert "biocurator" in baseline["prompt"].lower()
    assert off["task"] == baseline["task"]
    assert "EGFR" in off["task"] and "U2OS" in off["task"]
    assert off["analysts"] == baseline["analysts"]
    assert off["tools"] == baseline["tools"]
    changed_groups = {k for k in baseline["groups"] if off["groups"][k] != baseline["groups"][k]}
    assert changed_groups == {"lead"}
    blind = json.loads(_fresh(script, PETAB_BIOCURATOR="1", PETAB_BIOLOGY_BLIND="1"))
    assert not blind["enabled"] and "biocurator" not in blind["roster"]


@requires_harness
@pytest.mark.parametrize("literature", ["0", "1"])
def test_disabled_curator_skips_connections_discovery_and_model(literature):
    _fresh(
        r"""
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import pytest
from gemot import agent, prompts
from gemot.runner import main as runner
from gemot.runner import provenance
from gemot.runner.cli import build_parser
from gemot.scenario import Scenario
sys.path.insert(0, str(Path.cwd() / "tests"))
from test_run_loop_no_model import _run, _Probe

async def forbidden_discovery(*args, **kwargs):
    raise AssertionError("Disabled curator queried biology tools")

def forbidden_snapshot():
    raise AssertionError("Disabled curator recorded an unused biology snapshot")

with TemporaryDirectory() as directory, pytest.MonkeyPatch.context() as patch:
    source = Path(directory) / "example/problem"
    source.mkdir(parents=True)
    # Real cleanup must copy the supplied names, descriptions and numbers verbatim.
    supplied = {
        "observables.tsv": "observableId\tobservableName\nEGFR_obs\tEGFR abundance in U2OS\n",
        "measurements.tsv": "observableId\ttime\tmeasurement\nEGFR_obs\t0\t1.0\n",
        "conditions.tsv": "conditionId\tconditionName\ncontrol\tU2OS control\n",
    }
    for name, content in supplied.items():
        (source / name).write_text(content)
    scenario = Scenario("identified", source, lambda *args: None,
                        task="EGFR biology", known_constants="U2OS cells",
                        biocurator_entities=["EGFR"])
    patch.setattr(agent.driver, "discover_bio_tools", forbidden_discovery)
    patch.setattr(provenance, "_omnipath_provenance", forbidden_snapshot)
    result, graph = _run(agent, patch, _Probe(model_authored=True), confirm_after=1,
                         driver_options={"bio_sessions": [object()], "bio_literature": True,
                                         "curated_entities": scenario.biocurator_entities,
                                         "biocurator_model": "must_not_be_resolved",
                                         "system_prompt": scenario.system_prompt,
                                         "task": scenario.agent_task})
    assert result["biocurator_model"] is None
    for key in ("biocurator_tools", "biocurator_entities", "biocurator_literature"):
        assert result[key] == []
    expected = ["mathematician", "systems_scientist", "signal_engineer"]
    assert result["subagent_names"] == expected
    assert [s["name"] for s in graph.captured["subagents"]] == expected
    assert graph.captured["system_prompt"] == scenario.system_prompt
    assert "biocurator" not in graph.captured["system_prompt"].lower()
    assert graph.inputs[0]["messages"][0]["content"] == scenario.agent_task
    assert "make_changes" in {t.name for t in graph.captured["tools"]}

    @asynccontextmanager
    async def server(*args, **kwargs):
        yield SimpleNamespace(proc=SimpleNamespace(poll=lambda: None), session=object())

    @asynccontextmanager
    async def bio_sessions(urls):
        assert not urls, "Runner tried to open external biology connections"
        yield []

    seen = []
    async def drive(scenario, session, work, args, **kwargs):
        assert kwargs["bio_sessions"] == []
        assert "EGFR" in scenario.agent_task and "U2OS" in scenario.agent_task
        for name, content in supplied.items():
            assert (work / name).read_text() == content
        meta = provenance._agent_meta(provenance._base_meta(scenario, args), result, "completed")
        assert meta["ablations"]["biocurator"] is False
        assert meta["ablations"]["biology_blind"] is False
        assert "biocurator=off" in meta["ablation_arm"]
        assert "biocurator=off" in meta["ablation_state"]
        assert meta["biology_blinding"] is None and meta["omnipath_snapshot"] is None
        assert meta["subagent_names"] == expected and meta["biocurator_model"] is None
        assert meta["prompt_group_versions"] == prompts.prompt_group_versions()
        seen.append(meta)
        return 0, meta

    patch.setattr(runner, "_petab_server", server)
    patch.setattr(runner, "open_bio_sessions", bio_sessions)
    patch.setattr(runner, "_drive", drive)
    patch.setattr(runner, "archive_run", lambda *args: None)
    args = build_parser(scenario).parse_args(["--driver", "deepagents"])
    code, saved = asyncio.run(runner._main_async(scenario, args))
    assert code == 0 and seen == [saved]
""",
        PETAB_BIOCURATOR="0",
        PETAB_BIO_LITERATURE=literature,
    )
