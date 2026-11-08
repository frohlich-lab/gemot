"""The `fix_pointers` arm: off, the server withholds the numbers that point at what to change, and
no prompt tells the agent to read them.

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

#: Prompt text that reads a number the server withholds on this arm; a run that still carries one
#: is half-ablated.
_POINTERS = (
    "`merge_hints` on your last fit",
    "a NEGATIVE `dscore` is the win",
    "A priced merge is an OPTION",
    "get_fit_diagnosis's per-observable bias + worst-fit points",
    "flat_directions and ranked candidates",
)
_FACTS = (
    "THE RESIDUAL TESTS DO NOT DECIDE WHEN YOU ARE DONE",
    "A refit that only changes BOUNDS tests nothing about the structure",
    "Finish on the LOWEST-bic such model",
)
_READ = """
import json
from gemot import prompts as P
texts = [v for v in vars(P).values() if isinstance(v, str)]
texts += [json.dumps(P.PROMPT_GROUPS, ensure_ascii=False)]
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


def test_off_the_prompts_read_none_of_the_withheld_numbers(monkeypatch):
    on, off = _prompts(monkeypatch, None), _prompts(monkeypatch, "0")
    assert on["version"] != off["version"]
    for side in (on, off):
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
