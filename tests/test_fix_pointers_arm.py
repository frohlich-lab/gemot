"""The `fix_pointers` arm: off, neither the server nor the harness tells the agent what to change.

Prompts read the arm when `gemot.prompts` is imported, so each side is read in a fresh interpreter.
"""

from __future__ import annotations

import json
import subprocess
import sys
import types

import pytest

from gemot import ablations
from gemot.runner.server import _SERVER_BOOTSTRAP

#: What this arm cuts from the harness's own text; each is written in exactly one prompt fragment
#: or template, and a run that still carries one is half-ablated.
_POINTERS = (
    "A priced merge is an OPTION",  # _MERGE_RULE, which only makes sense beside merge_hints
    "Test a basin with more starts",  # _REFIT_TESTS_NO_STRUCTURE
    "STRUCTURE BEFORE POLISH",
    "Simplify the model FORM and refit",  # the parsimony refusal's fix menu
    "change approach: edit a different part of the model",  # the circuit breaker's escalation
)
_FACTS = (
    "A refit that only changes BOUNDS tests nothing about the structure",
    "THE RESIDUAL TESTS DO NOT DECIDE WHEN YOU ARE DONE",
    "matrix at the optimum is RANK-DEFICIENT",
)
_READ = """
import json
from gemot import prompts as P
from gemot.agent import circuit_breaker as C
texts = [v for v in vars(P).values() if isinstance(v, str)]
texts += [json.dumps(P.PROMPT_GROUPS, ensure_ascii=False), C._REJECTION_ESCALATION]
print(json.dumps({"text": "\\n".join(texts), "version": P.prompt_version()}, ensure_ascii=False))
"""


def _prompts(monkeypatch, value: str | None) -> dict:
    if value is None:
        monkeypatch.delenv("PETAB_FIX_POINTERS", raising=False)
    else:
        monkeypatch.setenv("PETAB_FIX_POINTERS", value)
    out = subprocess.run([sys.executable, "-c", _READ], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def test_the_arm_is_registered_on_and_moves_the_prompt():
    arm = next(a for a in ablations.ABLATIONS if a.name == "fix_pointers")
    assert arm.env == "PETAB_FIX_POINTERS"
    assert arm.default_on is True and arm.changes_prompt is True


def test_off_the_prompts_keep_the_facts_and_drop_the_fixes(monkeypatch):
    on, off = _prompts(monkeypatch, None), _prompts(monkeypatch, "0")
    assert on["version"] != off["version"]
    for side in (on, off):
        assert not any(b in side["text"] for b in "⟦⟧⟪⟫")
        assert all(fact in side["text"] for fact in _FACTS)
    assert all(pointer in on["text"] for pointer in _POINTERS)
    assert not [pointer for pointer in _POINTERS if pointer in off["text"]]


@pytest.mark.parametrize("flag,expected", [("1", True), ("0", False)])
def test_the_server_is_launched_with_the_arm(monkeypatch, flag, expected):
    """The twelfth argument of the fixed-arity bootstrap; its one mechanism is this arm's env var."""
    seen: dict = {}
    server = types.ModuleType("petab_mcp.server")
    server.create_server = lambda *a, **kw: (
        seen.update(kw) or types.SimpleNamespace(run=lambda **_: None)
    )
    uvicorn = types.ModuleType("uvicorn")
    uvicorn.Config = object
    monkeypatch.setitem(sys.modules, "petab_mcp.server", server)
    monkeypatch.setitem(sys.modules, "uvicorn", uvicorn)
    argv = ["-c", "p.yaml", "build", "off", "127.0.0.1", "0", "5", "", "0", "", "", flag]
    monkeypatch.setattr(sys, "argv", argv)
    exec(_SERVER_BOOTSTRAP, {})
    assert seen["fix_pointers"] is expected


def test_the_inlined_skill_is_rendered_as_the_server_renders_it(monkeypatch):
    """`_inline_skill` reads the server's SKILL.md from disk, brackets and all, so the harness
    renders it with its own copy of the grammar; the copy must agree with the server's."""
    from petab_mcp.messages import fix, fix_or, render

    from gemot import prompts

    text = "fact." + fix(" Send X" + fix(" now") + ".") + " A " + fix_or("is B", "differs") + "."
    for on in (True, False):
        monkeypatch.setattr(prompts, "FIX_POINTERS_ENABLED", on)
        assert prompts.render_fix_pointers(text) == render(text, on)
