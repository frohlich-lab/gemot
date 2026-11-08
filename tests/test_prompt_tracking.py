"""The prompt surface is tracked in full, and every result records which version produced it.

``prompt_version`` used to hash 5 of the ~35 model-facing constants in ``prompts.py``. The
other 30 — the panel-facing tool descriptions that REPLACE the server docstrings, the
biocurator, the subagent descriptions, the nudges, the whole finish gate — were model-facing
and unregistered, so a run could not say which text produced it. These tests keep that closed:
the completeness check below fails on any prompt constant that escapes ``PROMPT_GROUPS``, so
adding a prompt without tracking it is a red test rather than a silent gap.
"""

from __future__ import annotations

import argparse
import types
from pathlib import Path

import pytest

from gemot import prompts as P

# Module-level names that are NOT model-facing text: the registry machinery itself, the separator
# it serializes with, and the lists of TOOL NAMES a prompt renders itself from: the lead's own
# reads, tracked through DEFAULT_SYSTEM_PROMPT, and each analyst's, tracked through its
# descriptions. Editing a tuple still moves the version. Everything else in `prompts` is prose
# sent to a model.
_NOT_PROMPTS = {
    "PROMPT_GROUPS",
    "_HASH_SEP",
    "LEAD_SELF_READ_TOOLS",
    "MATHEMATICIAN_READ_TOOLS",
    "SYSTEMS_SCIENTIST_READ_TOOLS",
    "SIGNAL_ENGINEER_READ_TOOLS",
}

# FRAGMENTS: constants that are concatenated INTO another prompt rather than sent on their own
# (both report contracts are appended to the personas). They have no registry entry of their
# own, so they are checked by containment instead of equality. Keep this set tiny and
# explicit — it is the one hole in an otherwise exact guard, and a new constant lands under the
# strict rule by default.
_FRAGMENTS = {
    "_REPORT_CONTRACT",
    "_BIOCURATOR_REPORT_CONTRACT",
    # Concatenated into DEFAULT_SYSTEM_PROMPT.
    "_FREE_EXPONENTS_GUIDANCE",
    # Concatenated into the three ANALYST prompts (the skill formerly fetched via
    # get_skill once per spawn — now prompt text, versioned through those prompts).
    "_PETAB_SKILL_INLINE",
    # Concatenated into DEFAULT_SYSTEM_PROMPT.
    "_DIFF_EDITS_GUIDANCE",
    # The rules read on MORE THAN ONE surface, each composed into every prompt that
    # states it (see the block at the top of prompts.py). They are fragments for the
    # same reason as the guidance above; the difference is that these land in several
    # registered prompts rather than one, which is the whole point of their existing.
    "_PARAMETER_HALF",
    "_SIGMA_CAP_ADVISORY",
    "_DSCORE_SIGN",
    "_MERGE_RULE",
    "_MERGE_HINTS_WHERE",
    "_RESIDUALS_DO_NOT_DECIDE_DONE",
    "_MODEL_EDIT_OPS",
    "_CHI2_GATE_RULE",
    # Read by the statements lead and the null arm (the barebones template).
    "_FREE_LITERALS",
    # Closes every analyst description, on both arms.
    "_ANALYST_READS_HEAD",
    # What an analyst can and cannot do, in every lead prompt that introduces the analysts.
    "_ANALYST_LIMITS",
}


def _texts_of(value) -> list[str]:
    """Every string inside a prompt constant, recursively (str / dict / list / tuple)."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [t for v in value.values() for t in _texts_of(v)]
    if isinstance(value, (list, tuple)):
        return [t for v in value for t in _texts_of(v)]
    return []


def _prompt_constants() -> dict[str, object]:
    """Every module-level prompt constant: a `str`, or a container of them.

    Lists and tuples are walked too. A prompt held in a sequence is exactly as model-facing as
    one held in a dict, and leaving sequences out would be a hole in the guard rather than a
    simplification.
    """
    found: dict[str, object] = {}
    for name, value in vars(P).items():
        if name.startswith("__") or name in _NOT_PROMPTS:
            continue
        if isinstance(value, (types.FunctionType, types.ModuleType, type)):
            continue
        texts = _texts_of(value)
        if texts and all(isinstance(t, str) for t in texts):
            found[name] = value
    return found


# -- completeness ----------------------------------------------------------
def test_every_prompt_constant_is_tracked():
    """No model-facing string in prompts.py sits outside PROMPT_GROUPS.

    This is the guard the whole tracking change exists for: the registry is only "the full
    surface" for as long as nothing new slips past it, and a hand-check does not survive
    contact with a future edit.

    It checks EXACT membership against the registry's leaves, not containment against one
    concatenated blob. Containment silently passes any string that happens to sit inside a
    tracked one — e.g. a `_IDLE_HINT = "Emit a tool call now"` extracted from `_IDLE_NUDGE`
    would look tracked while being invisible to `prompt_version`.
    """
    leaves = P.tracked_leaves()
    missing = []
    for name, value in _prompt_constants().items():
        for text in _texts_of(value):
            if not text or text in leaves:
                continue
            if name in _FRAGMENTS and any(text in leaf for leaf in leaves):
                continue  # composed into another prompt; see _FRAGMENTS
            missing.append(name)
            break
    assert not missing, (
        "these prompt constants are not in prompts.PROMPT_GROUPS, so prompt_version does not "
        f"cover them: {sorted(missing)}. Add each to the group it belongs to (or, if it is "
        "concatenated into another prompt rather than sent on its own, to _FRAGMENTS here)."
    )


def test_the_fragments_really_are_fragments():
    """_FRAGMENTS is the guard's only escape hatch, so it must not silently collect entries
    that are simply untracked: each one has to actually appear inside a registered prompt."""
    leaves = P.tracked_leaves()
    for name in _FRAGMENTS:
        text = getattr(P, name)
        assert any(text in leaf and text != leaf for leaf in leaves), (
            f"{name} is in _FRAGMENTS but is not composed into any registered prompt — it is "
            "just untracked"
        )


def test_groups_are_disjoint_and_flatten_cleanly():
    names = [n for g in P.PROMPT_GROUPS.values() for n in g]
    assert len(names) == len(set(names)), "a registry name appears in two groups"
    assert set(names) == set(P.all_prompts())


def test_no_registered_prompt_contains_a_control_character():
    """What `collect_prompts()` returns is PROMPT TEXT: it is stored in Phoenix's registry and
    read by humans, so it must contain nothing a text column would reject.

    It briefly did. `_render` served two jobs — framing the hash input and bundling a
    multi-entry table — so the NUL that makes the hash unambiguous ended up inside 5 of the 14
    registered strings (38 of them in `petab-tool-descriptions`). A Postgres-backed Phoenix
    rejects 0x00, and `register_prompts` swallows the failure into `errors`, so the registry
    would have quietly lost those five while `prompt_version` claimed to cover them. `_bundle`
    (readable) and `_hash_input` (NUL-framed) are now separate.
    """
    offenders = {
        name: sorted({ch for ch in text if ord(ch) < 32 and ch not in "\n\t"})
        for name, text in P.all_prompts().items()
    }
    offenders = {k: v for k, v in offenders.items() if v}
    assert not offenders, f"registered prompts carry control characters: {offenders}"


def test_the_hash_framing_is_still_unambiguous():
    """The NUL framing is only safe BECAUSE no prompt contains one (the test above). Assert the
    two facts together, so removing one without the other is caught."""
    assert P._HASH_SEP == "\x00"
    assert not any(P._HASH_SEP in text for text in P.tracked_leaves())


def test_the_surface_covers_every_kind_of_prompt():
    """Sanity on the groups themselves, including the barebones and anonymous modeling arms."""
    assert set(P.PROMPT_GROUPS) == {
        "lead",
        "analysts",
        "biocurator",
        "control",
        "barebones",
        "biology_blind",
        "native_cli",
    }
    # the previously-untracked ones, named explicitly so a regression is legible
    tracked = P.all_prompts()
    assert P.BIOCURATOR_PROMPT == tracked["petab-biocurator-system"]
    assert P.FINISH_DESCRIPTION == tracked["petab-finish-description"]
    assert P._IDLE_NUDGE in tracked["petab-nudges"]
    assert P.FINISH_REFUSED_NO_FIT in tracked["petab-finish-gate"]


def test_the_lead_gets_the_readout_note_with_its_constants():
    """``agent_task`` appends the note to a scenario that HAS constants (every benchmark problem
    ships blinded readouts) and to nothing else (the synthetic demos have no blinding)."""
    from gemot.scenario import Scenario

    kw = dict(problem_dir=Path("/x/problem"), scripted_fix=None)
    with_note = Scenario(name="a", known_constants="KNOWN CONSTANTS: k.", **kw).agent_task
    assert with_note.endswith("KNOWN CONSTANTS: k." + P.KNOWN_CONSTANTS_READOUT_NOTE)
    assert Scenario(name="b", **kw).agent_task == P.DEFAULT_TASK


# -- versioning ------------------------------------------------------------
def test_prompt_version_is_stable_and_12_chars():
    v = P.prompt_version()
    assert v == P.prompt_version() and len(v) == 12


def _bump_first_leaf(entry):
    """Copy `entry` with its first leaf string altered — entries may be nested dicts now."""
    if isinstance(entry, str):
        return entry + " CHANGED"
    first = next(iter(entry))
    return {**entry, first: _bump_first_leaf(entry[first])}


def test_prompt_version_changes_when_any_group_changes(monkeypatch):
    """Every group is load-bearing on the version — including the ones that used to be
    invisible to it (tools, control, biocurator). Bumping a nested LEAF also proves the hash
    reaches inside a bundled entry rather than only its top-level key."""
    before = P.prompt_version()
    for group in P.PROMPT_GROUPS:
        patched = {
            g: (_bump_first_leaf(e) if g == group else e) for g, e in P.PROMPT_GROUPS.items()
        }
        monkeypatch.setattr(P, "PROMPT_GROUPS", patched)
        assert P.prompt_version() != before, f"changing group {group!r} did not move the version"
        monkeypatch.undo()
    assert P.prompt_version() == before


def test_group_versions_isolate_the_change(monkeypatch):
    """A bump is attributable: only the edited group's hash moves."""
    before = P.prompt_group_versions()
    patched = dict(P.PROMPT_GROUPS)
    patched["control"] = {**patched["control"], "petab-finish-description": "different text"}
    monkeypatch.setattr(P, "PROMPT_GROUPS", patched)
    after = P.prompt_group_versions()
    assert after["control"] != before["control"]
    assert {g: h for g, h in after.items() if g != "control"} == {
        g: h for g, h in before.items() if g != "control"
    }


def test_collect_prompts_is_the_full_surface():
    from gemot.experiments import collect_prompts, prompt_version

    assert collect_prompts() == P.all_prompts()
    assert prompt_version() == P.prompt_version()
    # every registered prompt has a declared chat role
    from gemot.experiments import _PROMPT_ROLES

    assert set(collect_prompts()) <= set(_PROMPT_ROLES)


# -- every result records it ----------------------------------------------
def test_base_meta_stamps_the_prompt_version(tmp_path):
    """Not just Phoenix: a plain run.py / slurm run writes result.json through _base_meta, and
    used to record git_sha but nothing about the prompts."""
    from gemot.runner.provenance import _base_meta

    scenario = types.SimpleNamespace(example_dir=tmp_path, version=1, amici_tolerance_env=dict)
    args = argparse.Namespace(
        driver="deepagents", model="m", recursion_limit=10, n_starts=5, temperature=0.0
    )
    meta = _base_meta(scenario, args)

    assert meta["prompt_version"] == P.prompt_version()
    assert meta["prompt_group_versions"] == P.prompt_group_versions()
    # it sits beside the other run-provenance keys, not in place of them
    assert "git_sha" in meta and "example_version" in meta


def test_skills_are_tracked_like_prompts(tmp_path):
    """The lead is told to read `petab-model-building` before authoring and the signal analyst
    `residual-analysis` before querying, so a skill edit changes what the agent is TOLD — it is a
    prompt change in everything but the file it lives in, and has to be versioned like one."""
    pytest.importorskip("langchain_core", reason="agent stack not installed")
    from gemot.agent.tools import ORCHESTRATOR_TOOLS
    from gemot.experiments import collect_skills, skills_version
    from petab_mcp.skills import skills_version as server_skills_version

    # the lead can actually reach it (otherwise the instruction is a dead letter).
    # list_skills is deliberately NOT exposed: zero agent calls across every measured sweep,
    # and the lead prompt names the two skills it needs directly.
    assert "get_skill" in ORCHESTRATOR_TOOLS
    assert "list_skills" not in ORCHESTRATOR_TOOLS
    assert "get_skill('petab-model-building')" in P.DEFAULT_SYSTEM_PROMPT
    # A server lists `statements-grammar` only on the statements route and `antimony-authoring`
    # only on the text routes (frohlich-lab/PEtab-MCP#204, #209), so each lead names its own.
    assert "get_skill('statements-grammar')" in P.INDRA_SYSTEM_PROMPT
    assert "get_skill('statements-grammar')" not in P.DEFAULT_SYSTEM_PROMPT
    assert "antimony-authoring" not in P.INDRA_SYSTEM_PROMPT
    # The `independence-audit` pointer rode with the retired data_regimes arm, which was never on
    # in practice: the lead must not be told to read a skill about a tool it does not hold.
    assert "get_skill('independence-audit')" not in P.DEFAULT_SYSTEM_PROMPT
    assert "get_data_regimes" not in P.DEFAULT_SYSTEM_PROMPT

    skills = collect_skills()
    assert set(skills) == {
        "petab-skill-petab-model-building",
        "petab-skill-antimony-authoring",
        "petab-skill-independence-audit",
        "petab-skill-reading-fits",
        "petab-skill-statements-grammar",
    }
    assert all(text.strip() for text in skills.values())

    v = skills_version()
    assert v == server_skills_version() and len(v) == 12

    # it is SEPARATE from prompt_version, so a bump says WHICH surface moved
    assert v != P.prompt_version()

    # ...and it lands everywhere prompt_version does
    from gemot import metrics, wandb_logging
    from gemot.runner.provenance import _base_meta

    assert "skills_version" in metrics.HISTORY_FIELDS
    assert "skills_version" in wandb_logging._CONFIG_KEYS
    scenario = types.SimpleNamespace(example_dir=tmp_path, version=1, amici_tolerance_env=dict)
    args = argparse.Namespace(
        driver="deepagents", model="m", recursion_limit=10, n_starts=5, temperature=0.0
    )
    assert _base_meta(scenario, args)["skills_version"] == v


def test_the_petab_skill_is_inlined_from_the_installed_server():
    """`_inline_skill` returns "" for a file it cannot find, so a server package that moved would
    silently drop the skill from every analyst prompt, and move prompt_version with it."""
    from conftest import SERVER_SRC

    skill = SERVER_SRC / "_skills" / "petab-model-building" / "SKILL.md"
    assert skill.read_text(encoding="utf-8") in P._PETAB_SKILL_INLINE


def test_prompt_version_reaches_the_sweep_sinks(tmp_path):
    """result.json is not where a sweep is analysed from. The history CSV writes with
    `extrasaction="ignore"`, so a key missing from the field list is dropped SILENTLY — a
    sweep spanning a prompt edit would pool both versions with no column to split them on."""
    import csv

    from gemot import metrics, wandb_logging

    assert "prompt_version" in metrics.HISTORY_FIELDS
    assert "prompt_version" in wandb_logging._CONFIG_KEYS
    # ...and it is a run INPUT, so it belongs in W&B config rather than the summary metrics
    assert "prompt_version" not in wandb_logging._METRIC_KEYS

    # round-trip a result-shaped row through the writer the sweep actually uses
    out = tmp_path / "history.csv"
    metrics.append_history([{"scenario": "toy", "prompt_version": "deadbeef1234"}], out)
    row = next(iter(csv.DictReader(out.open())))
    assert row["prompt_version"] == "deadbeef1234"


def test_appending_to_a_pre_existing_history_does_not_shift_columns(tmp_path):
    """The history is append-only and outlives a schema change. Writing rows in the CURRENT
    field order under an OLDER header would shift every column after the insertion point —
    `prompt_version`/`skills_version` went in mid-tuple, so every later column would move."""
    import csv as _csv

    from gemot import metrics

    out = tmp_path / "history.csv"
    old_fields = [
        f for f in metrics.HISTORY_FIELDS if f not in ("prompt_version", "skills_version")
    ]
    with out.open("w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=old_fields)
        w.writeheader()
        w.writerow({"scenario": "old", "run_started_at": "T0", "git_sha": "abc"})

    metrics.append_history(
        [
            {
                "scenario": "new",
                "run_started_at": "T1",
                "git_sha": "def",
                "prompt_version": "pv",
                "skills_version": "sv",
            }
        ],
        out,
    )

    rows = list(_csv.DictReader(out.open()))
    assert [r["run_started_at"] for r in rows] == ["T0", "T1"]
    assert [r["git_sha"] for r in rows] == ["abc", "def"]


# --------------------------------------------------------------------------- no archaeology
# Agent-facing text states RULES. It must not cite this project's own development history — a sweep
# size, a measured rate, an issue number, what some earlier version did. Three reasons that is not a
# style preference:
#
#   1. It is unverifiable to the reader. The agent cannot check a number from our runs, so it acts
#      as an appeal to authority and crowds out the rule it decorates.
#   2. It dates. Text describing how the harness behaves today becomes wrong when that changes, and
#      nothing fails when it does.
#   3. It leaks the benchmark's frame — run counts, pass rates and scoring anecdotes tell the agent
#      it is being scored, and roughly on what.
#
# A number that genuinely teaches belongs as a property of the problem ("a collapse of three
# near-free pairs can cost hundreds of bic"), which is checkable against the model in front of the
# agent, rather than as a finding of ours.
_ARCHAEOLOGY = [
    (r"\b\d+[\s-]run\b", "a run count from our own sweeps"),
    (r"\bsweeps?\b", "refers to our sweeps"),
    (r"issue #\d+", "an issue number"),
    (r"\bmeasured (?:on|across|over)\b", "cites our own measurement"),
    (r"\bwe (?:found|observed|measured|saw)\b", "cites our own observation"),
    (r"\bhit rate\b", "a cache statistic from our runs"),
    (r"%\s*of runs\b", "a rate over our runs"),
    # Deliberately NOT "an earlier version" / "no longer" / "previously": the agent has its own
    # model versions, and "never an earlier version's metrics" is exactly the rule the lead prompt
    # needs to state. A guard that flags correct text gets silenced rather than obeyed, so this
    # matches only phrasings that can mean the HARNESS's history and nothing else.
    (
        r"\bthis (?:harness|tool|prompt|skill) (?:used to|once)\b",
        "describes the harness's own change history",
    ),
    (r"\ban earlier draft\b", "describes an earlier draft of this text"),
    (r"\bregression\b", "describes our own defect history"),
]


def _agent_facing_text():
    """Every string the agent can actually read: the tracked prompts and the shipped skills."""
    from petab_mcp import skills as _skills

    for name, text in P.all_prompts().items():
        if isinstance(text, str):
            yield f"prompt:{name}", text
    root = Path(_skills.__file__).resolve().parent / "_skills"
    for path in sorted(root.rglob("SKILL.md")):
        yield f"skill:{path.parent.name}", path.read_text(encoding="utf-8")


def test_agent_facing_text_contains_no_archaeology():
    import re

    offences = []
    for where, text in _agent_facing_text():
        for pattern, why in _ARCHAEOLOGY:
            for m in re.finditer(pattern, text, re.IGNORECASE):
                s, e = max(0, m.start() - 70), min(len(text), m.end() + 70)
                offences.append(f"{where}: {why}\n      ...{text[s:e].strip()}...")
    assert not offences, (
        "agent-facing text cites this project's own history instead of stating a rule:\n    "
        + "\n    ".join(offences)
        + "\n  State the rule and drop the provenance. If the number genuinely teaches something, "
        "phrase it as a property of the problem ('a collapse of three near-free pairs can cost "
        "hundreds of bic'), not as a finding of ours. Development history belongs in a code "
        "comment, a commit message, or BENCHMARK_NOTES.md."
    )
