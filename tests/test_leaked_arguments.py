"""Arguments a model leaks into another argument's string are moved back out before dispatch.

Claude Haiku 4.5, on the indra_statements arm, closed `justification` and carried on in
Anthropic's XML tool-call syntax inside the JSON, so `make_changes` arrived with no `statements`:
141 "nothing to do" rejections in one sweep, and two runs ended without a model. The case below is
one of those calls, trimmed.
"""

from __future__ import annotations

import types

import pytest

from gemot.agent import guards

_LEAKED = {
    "justification": 'Initial model for HIV-1 viral dynamics under protease inhibitor treatment.",\n'
    '<parameter name="statements">[\n'
    '  "prod: infected_cells() IncreaseAmount virions() | indra_default kf=\'k_prod\'",\n'
    '  "clear: virions() DecreaseAmount void() | indra_default kf=\'k_clear\'"\n'
    "]",
    "starts": [{"agent": "virions", "amount": "V_0"}],
    "readouts": {"readout_viral_load": "amount(virions())"},
}


def test_the_leaked_rows_are_moved_back_to_their_argument():
    args, moved = guards._recover_leaked_arguments(_LEAKED)
    assert moved == ["statements"]
    assert args["justification"] == (
        "Initial model for HIV-1 viral dynamics under protease inhibitor treatment."
    )
    assert args["statements"][0].startswith("prod: infected_cells()")
    assert len(args["statements"]) == 2
    assert args["starts"] == _LEAKED["starts"] and args["readouts"] == _LEAKED["readouts"]


def test_a_closed_tag_and_a_second_leak_are_read_too():
    args, moved = guards._recover_leaked_arguments(
        {
            "justification": 'why",\n<parameter name="statements">["a: A becomes B | mass_action"]'
            '</parameter>\n<parameter name="n_starts">20</parameter>'
        }
    )
    assert moved == ["statements", "n_starts"]
    assert args == {
        "justification": "why",
        "statements": ["a: A becomes B | mass_action"],
        "n_starts": 20,
    }


def test_an_argument_the_call_carries_is_never_overwritten():
    args, moved = guards._recover_leaked_arguments(
        {"justification": 'why",\n<parameter name="n_starts">20', "n_starts": 5}
    )
    assert moved == [] and args["n_starts"] == 5 and args["justification"] == "why"


def test_keys_written_as_json_inside_the_string_are_moved_back():
    """The other leak: the rest of the call written as JSON inside the justification (Haiku on
    blasi, its `base_version` silently dropped)."""
    args, moved = guards._recover_leaked_arguments(
        {
            "justification": 'Return to version 2 and merge.", "base_version": 2}',
            "statements": ["a: A becomes B | mass_action"],
        }
    )
    assert moved == ["base_version"]
    assert args["justification"] == "Return to version 2 and merge."
    assert args["base_version"] == 2 and args["statements"] == ["a: A becomes B | mass_action"]


def test_a_call_without_a_leak_is_untouched():
    clean = {"justification": "why", "statements": ["a: A becomes B | mass_action"]}
    assert guards._recover_leaked_arguments(clean) == (clean, [])
    assert guards._recover_leaked_arguments(None) == (None, [])
    as_string = {"justification": "why", "amounts": '{"amt_a": "A", "amt_b": "B"}'}
    assert guards._recover_leaked_arguments(as_string) == (as_string, [])


def test_an_xml_leak_whose_value_is_an_object_moves_as_one_argument():
    args, moved = guards._recover_leaked_arguments(
        {"justification": 'Test.",\n<parameter name="amounts">{"a_amt": "A", "b_amt": "B"}'}
    )
    assert moved == ["amounts"] and args["amounts"] == {"a_amt": "A", "b_amt": "B"}


def test_the_middleware_dispatches_the_repaired_call():
    pytest.importorskip("langchain.agents.middleware.types", reason="agent stack not installed")
    stats: dict = {}
    repair = guards._leaked_argument_repair(stats)
    seen = {}

    class _Request(types.SimpleNamespace):
        def override(self, **kw):
            return _Request(**{**self.__dict__, **kw})

    def handler(request):
        seen.update(request.tool_call["args"])
        return "dispatched"

    request = _Request(tool_call={"name": "make_changes", "args": dict(_LEAKED), "id": "c1"})
    assert repair.wrap_tool_call(request, handler) == "dispatched"
    assert len(seen["statements"]) == 2
    assert stats["leaked_arguments_recovered"] == 1
