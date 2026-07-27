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

from petab_harness import prompts as P

# Module-level names that are NOT model-facing text: the registry machinery itself and the
# separator it serializes with. Everything else in `prompts` is prose sent to a model.
_NOT_PROMPTS = {"PROMPT_GROUPS", "_HASH_SEP"}

# FRAGMENTS: constants that are concatenated INTO another prompt rather than sent on their own
# (both report contracts are appended to the personas). They have no registry entry of their
# own, so they are checked by containment instead of equality. Keep this set tiny and
# explicit — it is the one hole in an otherwise exact guard, and a new constant lands under the
# strict rule by default.
_FRAGMENTS = {"_REPORT_CONTRACT", "_BIOCURATOR_REPORT_CONTRACT"}


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
    """Sanity on the groups themselves — the five kinds of text the harness sends."""
    assert set(P.PROMPT_GROUPS) == {"lead", "analysts", "biocurator", "tools", "control"}
    # the previously-untracked ones, named explicitly so a regression is legible
    tracked = P.all_prompts()
    assert P.TOOL_DESCRIPTIONS["make_changes"] in tracked["petab-tool-descriptions"]
    assert P.BIOCURATOR_PROMPT == tracked["petab-biocurator-system"]
    assert P.MAKE_DECISION_DESCRIPTION == tracked["petab-make-decision"]
    assert P._IDLE_NUDGE in tracked["petab-nudges"]
    assert P.FINISH_REFUSED_NO_FIT in tracked["petab-finish-gate"]


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
            g: (_bump_first_leaf(e) if g == group else e)
            for g, e in P.PROMPT_GROUPS.items()
        }
        monkeypatch.setattr(P, "PROMPT_GROUPS", patched)
        assert P.prompt_version() != before, f"changing group {group!r} did not move the version"
        monkeypatch.undo()
    assert P.prompt_version() == before


def test_group_versions_isolate_the_change(monkeypatch):
    """A bump is attributable: only the edited group's hash moves."""
    before = P.prompt_group_versions()
    patched = dict(P.PROMPT_GROUPS)
    patched["control"] = {**patched["control"], "petab-make-decision": "different text"}
    monkeypatch.setattr(P, "PROMPT_GROUPS", patched)
    after = P.prompt_group_versions()
    assert after["control"] != before["control"]
    assert {g: h for g, h in after.items() if g != "control"} == {
        g: h for g, h in before.items() if g != "control"
    }


def test_collect_prompts_is_the_full_surface():
    from petab_harness.experiments import collect_prompts, prompt_version

    assert collect_prompts() == P.all_prompts()
    assert prompt_version() == P.prompt_version()
    # every registered prompt has a declared chat role
    from petab_harness.experiments import _PROMPT_ROLES

    assert set(collect_prompts()) <= set(_PROMPT_ROLES)


# -- the panel's tool text is owned by prompts.py --------------------------
def test_every_panel_tool_has_a_description():
    """A tool without a TOOL_DESCRIPTIONS entry falls back to a truncated SERVER docstring, so
    its panel-facing text lives outside prompts.py and outside the tracked surface. That is how
    `list_conditions` came to be the one tool the panel read from server.py."""
    pytest.importorskip("langchain_core", reason="agent stack not installed")
    from petab_harness.agent.tools import ALLOWED_TOOLS

    assert ALLOWED_TOOLS == set(P.TOOL_DESCRIPTIONS), (
        "ALLOWED_TOOLS and TOOL_DESCRIPTIONS must match exactly; "
        f"undescribed={sorted(ALLOWED_TOOLS - set(P.TOOL_DESCRIPTIONS))}, "
        f"stale={sorted(set(P.TOOL_DESCRIPTIONS) - ALLOWED_TOOLS)}"
    )


# -- every result records it ----------------------------------------------
def test_base_meta_stamps_the_prompt_version(tmp_path):
    """Not just Phoenix: a plain run.py / slurm run writes result.json through _base_meta, and
    used to record git_sha but nothing about the prompts."""
    from petab_harness.runner.provenance import _base_meta

    scenario = types.SimpleNamespace(example_dir=tmp_path, version=1)
    args = argparse.Namespace(driver="deepagents", model="m", recursion_limit=10,
                              n_starts=5, temperature=0.0)
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
    from petab_harness.agent.tools import ORCHESTRATOR_TOOLS
    from petab_harness.experiments import collect_skills, skills_version
    from petab_mcp.skills import skills_version as server_skills_version

    # the lead can actually reach them (otherwise the instruction is a dead letter)
    assert {"get_skill", "list_skills"} <= set(ORCHESTRATOR_TOOLS)
    assert "get_skill('petab-model-building')" in P.DEFAULT_SYSTEM_PROMPT

    skills = collect_skills()
    assert set(skills) == {"petab-skill-petab-model-building",
                           "petab-skill-antimony-authoring"}
    assert all(text.strip() for text in skills.values())

    v = skills_version()
    assert v == server_skills_version() and len(v) == 12

    # it is SEPARATE from prompt_version, so a bump says WHICH surface moved
    assert v != P.prompt_version()

    # ...and it lands everywhere prompt_version does
    from petab_harness import metrics, wandb_logging
    from petab_harness.runner.provenance import _base_meta

    assert "skills_version" in metrics.HISTORY_FIELDS
    assert "skills_version" in wandb_logging._CONFIG_KEYS
    scenario = types.SimpleNamespace(example_dir=tmp_path, version=1)
    args = argparse.Namespace(driver="deepagents", model="m", recursion_limit=10,
                              n_starts=5, temperature=0.0)
    assert _base_meta(scenario, args)["skills_version"] == v


def test_skills_version_moves_when_a_skill_changes(tmp_path):
    """A skill edit must be visible in the version, or the tracking is decorative."""
    from petab_mcp.skills import skills_version

    src = Path(__file__).parent.parent / "src" / "petab_mcp" / "_skills"
    before = skills_version(src)

    fake = tmp_path / "_skills" / "petab-model-building"
    fake.mkdir(parents=True)
    (fake / "SKILL.md").write_text(
        (src / "petab-model-building" / "SKILL.md").read_text() + "\nAn extra rule.\n")
    assert skills_version(tmp_path / "_skills") != before


def test_prompt_version_reaches_the_sweep_sinks(tmp_path):
    """result.json is not where a sweep is analysed from. The history CSV writes with
    `extrasaction="ignore"`, so a key missing from the field list is dropped SILENTLY — a
    sweep spanning a prompt edit would pool both versions with no column to split them on."""
    import csv

    from petab_harness import metrics, wandb_logging

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

    from petab_harness import metrics

    out = tmp_path / "history.csv"
    old_fields = [f for f in metrics.HISTORY_FIELDS
                  if f not in ("prompt_version", "skills_version")]
    with out.open("w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=old_fields)
        w.writeheader()
        w.writerow({"scenario": "old", "run_started_at": "T0", "git_sha": "abc"})

    metrics.append_history([{"scenario": "new", "run_started_at": "T1", "git_sha": "def",
                             "prompt_version": "pv", "skills_version": "sv"}], out)

    rows = list(_csv.DictReader(out.open()))
    assert [r["run_started_at"] for r in rows] == ["T0", "T1"]
    assert [r["git_sha"] for r in rows] == ["abc", "def"]
