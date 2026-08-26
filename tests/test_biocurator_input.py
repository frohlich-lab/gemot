"""Tests for the biocurator input channel (petab_harness.agent) — issue #91.

The deepagents ``task`` tool delivers only a free-text ``description`` to a subagent; a lead that
stuffs the biocurator's entity names into a ``prompt=`` argument has them SILENTLY DROPPED. The
biocurator (unlike the analysts) holds no petab tool and cannot self-serve, so that broke its whole
input. The fix (a) auto-injects the entity names the harness knows into the biocurator's persona
prompt, and (b) adds a lead-only middleware that REJECTS a ``task`` call carrying unexpected kwargs
back to the model instead of dispatching a subagent on truncated input.

The pure helpers are tested with no third-party dependency. The middleware is exercised directly
(its ``wrap_tool_call``/``awrap_tool_call`` methods) — skipped where langchain is absent, as it is
only an examples-harness dependency (mirrors tests/test_agent_tool_dispatch.py).
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path


from petab_harness import agent, prompts, scenario


# --------------------------------------------------------------------- _unexpected_task_kwargs
def test_unexpected_task_kwargs_flags_only_the_extras():
    assert agent._unexpected_task_kwargs({"description": "d", "subagent_type": "b"}) == []
    assert agent._unexpected_task_kwargs(
        {"description": "d", "subagent_type": "b", "prompt": "p"}) == ["prompt"]


def test_unexpected_task_kwargs_is_order_stable_and_multi():
    # insertion order is preserved; both extras are reported
    assert agent._unexpected_task_kwargs(
        {"prompt": "p", "message": "m", "description": "d"}) == ["prompt", "message"]


def test_unexpected_task_kwargs_handles_non_dict():
    assert agent._unexpected_task_kwargs(None) == []
    assert agent._unexpected_task_kwargs("nope") == []
    assert agent._unexpected_task_kwargs([1, 2]) == []


def test_task_arg_reject_message_names_the_keys_and_the_channel():
    msg = agent._task_arg_reject_message(["prompt"])
    assert "prompt" in msg
    assert "description" in msg          # points the model at the right channel
    assert "REJECTED" in msg


# --------------------------------------------------------------------- _biocurator_entity_block
def test_biocurator_entity_block_lists_names():
    block = agent._biocurator_entity_block(["Naive", "EarlyEffector"])
    assert "Naive" in block and "EarlyEffector" in block


def test_biocurator_entity_block_stays_blind():
    # The biocurator is task-blinded: the injected line must not leak the data/model/fitting frame.
    block = agent._biocurator_entity_block(["Naive", "Pathogen"]).lower()
    for banned in ("model", "data", "fit", "measure", "observable", "experiment", "simulat"):
        assert banned not in block, f"blind-breaking word {banned!r} leaked into the entity block"


def test_biocurator_entity_block_empty_when_no_names():
    assert agent._biocurator_entity_block([]) == ""
    assert agent._biocurator_entity_block(None) == ""


# --------------------------------------------------------------------- _build_biocurator_subagent
def test_build_biocurator_subagent_injects_names_into_the_prompt():
    key = agent._subagent_prompt_key()
    spec = agent._build_biocurator_subagent([], ["Naive", "Memory"])
    assert spec["name"] == "biocurator"
    assert spec[key].startswith(agent.BIOCURATOR_PROMPT)   # persona kept, block appended
    assert "Naive" in spec[key] and "Memory" in spec[key]


def test_build_biocurator_subagent_without_names_leaves_prompt_unchanged():
    key = agent._subagent_prompt_key()
    spec = agent._build_biocurator_subagent([])
    assert spec[key] == agent.BIOCURATOR_PROMPT


# ------------------------------------------------------- targeted follow-up branching
# Subagent calls are stateless one-shots: the lead can call the biocurator again with a
# FOLLOW-UP question in `description`, and the persona re-ran the whole "examine all entities"
# survey every time (all entities are auto-injected, not passed by the lead). The persona now
# BRANCHES — an open "characterize these" request still gets the full survey, but a specific
# question gets a focused answer — WITHOUT ever learning what the lead is doing (the blind).
def test_biocurator_prompt_branches_open_vs_specific():
    p = agent.BIOCURATOR_PROMPT
    # both request shapes are named, and the specific-question path is a FOCUSED answer, not the
    # full survey re-run.
    assert "OPEN" in p and "SPECIFIC question" in p
    assert "focused answer" in p
    assert "do NOT re-run the whole survey" in p
    # the open path still mandates the full survey (so a bare entity list is unchanged behaviour)
    assert "give the full survey" in p


def test_biocurator_report_contract_branches_the_report_form():
    """The report CONTRACT — the thing that forced 'one section per bullet above' every call —
    now branches, so a focused answer is a valid shape and not a survey."""
    c = prompts._BIOCURATOR_REPORT_CONTRACT
    assert "OPEN request" in c and "one section per bullet above" in c   # survey path kept
    assert "SPECIFIC question" in c and "need not walk all three bullets" in c   # focused path
    # the invariants that hold in BOTH shapes survive the branch
    assert "PMID/DOI" in c and "NEVER return an empty message" in c


def test_biocurator_prompt_stays_blind_after_the_branch():
    """The follow-up handling must not hand the biocurator the modeling frame.

    The persona now OPENS with its one sanctioned window — `list_observables`, whose
    observableName states what each assay MEASURES (owner decision, 2026-08-26) — so the
    measurement vocabulary in that opening paragraph is the exception, not a leak. What must
    still be absent is the MODELING frame: the model, the fits, the simulations, the parameters.
    The opening itself asserts the boundary ("you never see the model, the fits, or the data"),
    so those words appear exactly once, in the sentence that forbids them — strip the opening
    paragraph before scanning.

    Word boundaries, so legitimate desk vocabulary is not a false positive: 'databases' is not
    'data', and 'judge fit before relying on a paper' is 'fit' = suitability."""
    opening, _, rest = agent.BIOCURATOR_PROMPT.partition("\n")
    assert "list_observables" in opening and "never see the model" in opening, (
        "the sanctioned-window opening moved; re-anchor this test")
    frame = [r"\bmodel", r"\bdata\b", r"\bfitting\b", r"\bfitted\b",
             r"\bexperiment", r"\bsimulat", r"\bparameter"]
    leaked = [f for f in frame if re.search(f, rest, re.IGNORECASE)]
    assert not leaked, f"blind-breaking frame words leaked into the biocurator persona: {leaked}"


# --------------------------------------------------------------------- per-scenario curation
def test_scenario_has_biocurator_entities_field():
    """The Scenario dataclass carries a hand-curated biocurator_entities list (default empty)."""
    mod = scenario

    async def _noop(call, n):
        return None

    default = mod.Scenario(name="x", problem_dir=Path("/tmp"), scripted_fix=_noop)
    assert default.biocurator_entities == []
    # the mutable default must not be shared between instances
    default.biocurator_entities.append("LEAK")
    assert mod.Scenario(name="z", problem_dir=Path("/tmp"),
                        scripted_fix=_noop).biocurator_entities == []
    set_ = mod.Scenario(name="y", problem_dir=Path("/tmp"), scripted_fix=_noop,
                        biocurator_entities=["STAT5A", "STAT5B"])
    assert set_.biocurator_entities == ["STAT5A", "STAT5B"]


def test_every_example_curates_biocurator_entities():
    """Every example run.py must explicitly set biocurator_entities (even if []) — a guard so a
    newly-added example is deliberately curated rather than silently defaulting."""
    examples = Path(__file__).resolve().parents[1] / "examples"
    run_files = sorted(examples.glob("*/run.py"))
    assert run_files, "no example run.py files found"
    missing = [p.parent.name for p in run_files if "biocurator_entities=" not in p.read_text()]
    assert not missing, f"examples not setting biocurator_entities: {missing}"


# --------------------------------------------------------------------- _task_arg_guard middleware
# Building the middleware imports langchain (examples-harness only), so guard these tests.
from conftest import requires_langchain  # noqa: E402


class _FakeRequest:
    def __init__(self, name, args, id_="c1"):
        self.tool_call = {"name": name, "args": args, "id": id_, "type": "tool_call"}


@requires_langchain
def test_task_arg_guard_rejects_a_task_with_unexpected_kwargs():
    from langchain_core.messages import ToolMessage

    stats: dict = {}
    mw = agent._task_arg_guard(stats, max_rejections=3)
    dispatched = []

    def handler(req):
        dispatched.append(req)
        return "DISPATCHED"

    bad = _FakeRequest("task", {"description": "d", "subagent_type": "biocurator", "prompt": "e"})
    out = mw.wrap_tool_call(bad, handler)
    assert isinstance(out, ToolMessage)
    assert out.status == "error"
    assert dispatched == []                              # subagent never ran on truncated input
    assert stats["task_arg_rejections"] == 1
    assert stats["unexpected_task_kwargs"] == ["prompt"]


@requires_langchain
def test_task_arg_guard_passes_clean_task_and_ignores_non_task_calls():
    stats: dict = {}
    mw = agent._task_arg_guard(stats)
    dispatched = []

    def handler(req):
        dispatched.append(req.tool_call["name"])
        return "OK"

    clean = _FakeRequest("task", {"description": "Naive, Memory", "subagent_type": "biocurator"})
    # a non-task tool with an extra kwarg is none of this guard's business — it must pass through
    other = _FakeRequest("set_model", {"antimony": "x", "prompt": "y"})
    assert mw.wrap_tool_call(clean, handler) == "OK"
    assert mw.wrap_tool_call(other, handler) == "OK"
    assert dispatched == ["task", "set_model"]
    assert stats.get("task_arg_rejections", 0) == 0
    assert stats.get("unexpected_task_kwargs") in (None, [])


@requires_langchain
def test_task_arg_guard_relents_after_the_cap():
    from langchain_core.messages import ToolMessage

    stats: dict = {}
    mw = agent._task_arg_guard(stats, max_rejections=2)
    dispatched = []

    def handler(req):
        dispatched.append(req)
        return "DISPATCHED"

    bad = _FakeRequest("task", {"description": "d", "subagent_type": "biocurator", "prompt": "e"})
    assert isinstance(mw.wrap_tool_call(bad, handler), ToolMessage)   # reject 1
    assert isinstance(mw.wrap_tool_call(bad, handler), ToolMessage)   # reject 2
    assert dispatched == []
    # cap hit -> relent: the third bad call falls through and dispatches (extras dropped downstream)
    assert mw.wrap_tool_call(bad, handler) == "DISPATCHED"
    assert len(dispatched) == 1
    assert stats["task_arg_rejections"] == 2                          # counter stopped at the cap
    assert stats["unexpected_task_kwargs"] == ["prompt"]             # still recorded every time


@requires_langchain
def test_task_arg_guard_async_paths():
    from langchain_core.messages import ToolMessage

    stats: dict = {}
    mw = agent._task_arg_guard(stats, max_rejections=1)

    async def handler(req):
        return "DISPATCHED"

    bad = _FakeRequest("task", {"description": "d", "subagent_type": "biocurator", "prompt": "e"})
    out1 = asyncio.run(mw.awrap_tool_call(bad, handler))
    assert isinstance(out1, ToolMessage) and out1.status == "error"
    # cap=1 -> the next bad call relents and awaits the handler
    out2 = asyncio.run(mw.awrap_tool_call(bad, handler))
    assert out2 == "DISPATCHED"
    assert stats["task_arg_rejections"] == 1


@requires_langchain
def test_task_arg_guard_end_to_end_bounces_then_retry_dispatches():
    """The full issue-#91 loop against langchain's real agent graph: the lead's first `task` call
    carries a dropped `prompt=` (the entities) alongside a weak `description` and is REJECTED; its
    clean retry — entities now in `description` — dispatches. The guard sits OUTSIDE
    _tool_dispatch_guard, so the bounced call must NOT inflate `tools_executed`."""
    from langchain.agents import create_agent
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.tools import tool

    class ToolFakeModel(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    ran: list = []

    @tool
    def task(description: str, subagent_type: str) -> str:
        """Delegate to a subagent."""
        ran.append((description, subagent_type))
        return "subagent report"

    msgs = [
        AIMessage(content="", tool_calls=[{  # the broken call from the issue's t1
            "name": "task", "id": "c1", "type": "tool_call",
            "args": {"description": "markers", "subagent_type": "biocurator",
                     "prompt": "Naive, EarlyEffector"}}]),
        AIMessage(content="", tool_calls=[{  # the corrected retry
            "name": "task", "id": "c2", "type": "tool_call",
            "args": {"description": "Naive, EarlyEffector", "subagent_type": "biocurator"}}]),
        AIMessage(content="done"),
    ]
    stats = {"tools_executed": 0, "dropped_dup_tool_calls": 0, "idle_nudges": 0,
             "gave_up_no_tools": False, "unexpected_task_kwargs": [], "task_arg_rejections": 0}
    # same relative order as run_deepagents: _task_arg_guard OUTSIDE _tool_dispatch_guard.
    graph = create_agent(ToolFakeModel(responses=msgs), tools=[task],
                         middleware=[agent._task_arg_guard(stats),
                                     agent._tool_dispatch_guard(stats)])
    graph.invoke({"messages": [HumanMessage(content="build a model")]},
                 config={"recursion_limit": 30})

    assert ran == [("Naive, EarlyEffector", "biocurator")]   # only the clean retry ran
    assert stats["tools_executed"] == 1                       # the bounced call was NOT counted
    assert stats["task_arg_rejections"] == 1
    assert stats["unexpected_task_kwargs"] == ["prompt"]
