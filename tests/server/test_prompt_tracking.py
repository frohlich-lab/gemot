"""The shipped SKILLS are tracked like prompts, and say only what a prompt may.

The skills are agent-facing text that ships with the server, so these read the installed
server's ``_skills/`` (``server_src``). The prompts' own tracking, and the same rules applied to
the prompts, are in ``tests/test_prompt_tracking.py``.
"""

from __future__ import annotations

import pytest

from gemot import prompts as P
from test_prompt_tracking import _archaeology_offences


def test_skills_are_versioned_separately_from_the_prompts():
    """A skill edit changes what the agent is TOLD, so the skills are collected and versioned like
    the prompts (``tests/test_prompt_tracking.py`` pins where that version lands); this pins that
    the version is the server's own, over every skill the prompts point at."""
    pytest.importorskip("langchain_core", reason="agent stack not installed")
    from gemot.experiments import collect_skills, skills_version
    from petab_mcp.skills import skills_version as server_skills_version

    skills = collect_skills()
    assert set(skills) == {
        "petab-skill-petab-model-building",
        "petab-skill-antimony-authoring",
        "petab-skill-independence-audit",
    }
    assert all(text.strip() for text in skills.values())

    v = skills_version()
    assert v == server_skills_version() and len(v) == 12

    # it is SEPARATE from prompt_version, so a bump says WHICH surface moved
    assert v != P.prompt_version()


def test_the_petab_skill_is_inlined_from_the_installed_server(server_src):
    """`_inline_skill` returns "" for a file it cannot find, so a server package that moved would
    silently drop the skill from every analyst prompt, and move prompt_version with it."""
    skill = server_src / "_skills" / "petab-model-building" / "SKILL.md"
    assert skill.read_text(encoding="utf-8") in P._PETAB_SKILL_INLINE


def test_skills_version_moves_when_a_skill_changes(tmp_path, server_src):
    """A skill edit must be visible in the version, or the tracking is decorative."""
    from petab_mcp.skills import skills_version

    src = server_src / "_skills"
    before = skills_version(src)

    fake = tmp_path / "_skills" / "petab-model-building"
    fake.mkdir(parents=True)
    (fake / "SKILL.md").write_text(
        (src / "petab-model-building" / "SKILL.md").read_text() + "\nAn extra rule.\n"
    )
    assert skills_version(tmp_path / "_skills") != before


def test_the_shipped_skills_contain_no_archaeology(server_src):
    """The rule ``tests/test_prompt_tracking.py`` applies to the prompts, applied to the skills the
    agent reads through ``get_skill``."""
    skills = {
        f"skill:{path.parent.name}": path.read_text(encoding="utf-8")
        for path in sorted((server_src / "_skills").rglob("SKILL.md"))
    }
    assert skills, "no shipped skills found"
    offences = _archaeology_offences(skills)
    assert not offences, (
        "a shipped skill cites this project's own history instead of stating a rule:\n    "
        + "\n    ".join(offences)
    )
