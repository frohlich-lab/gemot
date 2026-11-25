"""The two compounding failures the August sweep exposed in the tool-call path.

Both were silent: the runs "completed" and simply scored nothing.

1. The breaker disabled `make_changes`, which is the run's ONLY editor — so the run could not
   author a model or move a bound ever again, and ended `no_model_built` by construction.
2. What tripped it was a JSON-encoded STRING passed where the schema declares an array. That
   fails validation on the CLIENT (the pydantic error is titled `call[make_changes]`, raised
   against the published tool schema), so the request never reached the server.
"""

import asyncio

import pytest

from gemot.agent.circuit_breaker import (
    _TOOL_ERROR_DISABLE_THRESHOLD,
    _coerce_int_args,
    _coerce_json_string_args,
    _container_arg_names,
    _is_transport_error,
    _new_breaker_state,
    _record_error,
    _string_arg_names,
    _wrap_tools_with_circuit_breaker,
)


class _FakeTool:
    """Enough of a langchain tool for the wrapper: a name, an args schema, a coroutine."""

    def __init__(self, name, args=None, fail_with=None, record=None):
        self.name = name
        self.args = args or {}
        self._fail_with = fail_with
        self._record = record if record is not None else []

        async def _run(**arguments):
            self._record.append(arguments)
            if self._fail_with is not None:
                raise self._fail_with
            return ([{"type": "text", "text": "{}"}], None)

        self.coroutine = _run

    @property
    def calls(self):
        return self._record


# ------------------------------------------------------- the sole editor is never disabled


def test_make_changes_is_never_disabled_however_often_it_fails():
    state = _new_breaker_state()
    for _ in range(_TOOL_ERROR_DISABLE_THRESHOLD + 5):
        tripped = _record_error(state, "make_changes", "ValidationError: same every time")
        assert tripped is False
    assert "make_changes" not in state["disabled"], (
        "disabling the only editor does not degrade a run, it guarantees no_model_built"
    )


def test_an_ordinary_tool_is_still_disabled():
    """The guard is scoped — the breaker must keep working for everything else."""
    state = _new_breaker_state()
    tripped = [
        _record_error(state, "get_uniprot_protein_info", "same error")
        for _ in range(_TOOL_ERROR_DISABLE_THRESHOLD)
    ]
    assert tripped[-1] is True
    assert "get_uniprot_protein_info" in state["disabled"]


def test_a_repeated_sole_editor_failure_escalates_the_message():
    """It keeps the tool, so the message has to do the work a disable used to."""
    state = _new_breaker_state()
    tool = _FakeTool("make_changes", fail_with=RuntimeError("bad shape"))
    _wrap_tools_with_circuit_breaker([tool], state)

    async def _call():
        return await tool.coroutine(justification="j")

    for i in range(_TOOL_ERROR_DISABLE_THRESHOLD - 1):
        content, _ = asyncio.run(_call())
        assert "bad shape" in content, "the original error must still reach the model"
        assert "identical failure" not in content, f"escalated too early, on call {i + 1}"

    content, _ = asyncio.run(_call())
    assert "identical failure" in content and "SHAPE" in content
    assert "TOOL DISABLED" not in content


def test_a_sole_editor_failure_is_returned_to_the_model_never_raised():
    """The failure that cost 3 of minimax-m3's 5 sneyd runs (0 fits each).

    The tool bridge (`langchain.mcp`, as langchain-mcp-adapters before it) installs
    `_handle_mcp_tool_error` as these tools' `handle_tool_error`,
    and it converts an error into a ToolMessage for `_MCPToolExecutionError` ONLY — every other
    exception hits its `raise error` branch and aborts the run. So raising anything here (the old
    escalation raised a plain ToolException) killed the run instead of teaching the model. Client
    -side pydantic validation errors are worse still: not ToolException subclasses at all, so they
    never reach that callback. `make_changes` is the sole editor; losing the run loses everything.
    """

    class _NotAToolException(Exception):
        pass

    for exc in (_NotAToolException("client-side validation error"), RuntimeError("boom")):
        state = _new_breaker_state()
        tool = _FakeTool("make_changes", fail_with=exc)
        _wrap_tools_with_circuit_breaker([tool], state)
        for _ in range(_TOOL_ERROR_DISABLE_THRESHOLD + 2):
            result = asyncio.run(tool.coroutine(justification="j"))
            assert isinstance(result, tuple) and len(result) == 2, (
                "must be (content, artifact) — the shape these tools already declare"
            )
            assert str(exc) in result[0]


# ------------------------------------------------------- client-side JSON-string coercion

_ARGS_SCHEMA = {
    "justification": {"type": "string"},
    "antimony": {"anyOf": [{"type": "string"}, {"type": "null"}]},
    "parameters": {"anyOf": [{"type": "array", "items": {"type": "object"}}, {"type": "null"}]},
    "references": {"anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "null"}]},
}


def test_container_args_are_read_off_the_published_schema():
    """The DECLARED container type is kept, not just the argument name — an object string where
    an array is required is not a repairable quoting artefact."""
    tool = _FakeTool("make_changes", args=_ARGS_SCHEMA)
    assert _container_arg_names(tool) == {"parameters": "array", "references": "array"}


def test_a_schema_without_properties_does_not_abort_run_setup():
    """`BaseTool.args` is a property that raises KeyError('properties') for a zero-argument MCP
    tool published as `{"type": "object"}` — and getattr's default only swallows AttributeError.
    This runs for EVERY tool at wrap time, including hosted bio tools whose schemas are not ours,
    so an unguarded read would kill the run before the agent starts."""

    class _Exploding:
        name = "zeroarg"

        @property
        def args(self):
            raise KeyError("properties")

    assert _container_arg_names(_Exploding()) == {}


def test_a_json_string_where_an_array_is_required_is_parsed():
    """Verbatim shape from xiaomi/mimo-v2.5 in the sweep."""
    out = _coerce_json_string_args(
        {
            "justification": "j",
            "parameters": '[{"parameter_id": "V_init", "lower_bound": 1, "scale": "log10"}]',
        },
        {"parameters": "array"},
    )
    assert out["parameters"] == [{"parameter_id": "V_init", "lower_bound": 1, "scale": "log10"}]


def test_string_arguments_are_never_touched():
    """An Antimony model or justification is a string BY SCHEMA — coercion must not reach it,
    even if it happens to start with a bracket."""
    args = {"justification": "[not json", "antimony": "[R: A -> B; k*A]"}
    assert _coerce_json_string_args(args, {"parameters": "array"}) == args


def test_unparseable_or_wrongly_shaped_strings_pass_through_untouched():
    """This widens what is accepted; it must never change what a valid call means, and anything
    it cannot repair is left for normal validation to reject.

    `{"parameter_id": ...}` is the notable one: it PARSES, but to the wrong container. Forwarding
    it would only trade a client-side failure for a server round trip that fails anyway, and add
    a streak increment on the tool that is never allowed to be disabled.
    """
    for value in ("[not valid json", '"just a string"', "[1, 2", "42", '{"parameter_id": "k1"}'):
        args = {"parameters": value}
        assert _coerce_json_string_args(args, {"parameters": "array"}) == args


def test_an_int_becomes_a_string_only_where_the_schema_allows_no_number():
    """Run 63782292_21 sent `species=9606` to a string-or-list `species`. An argument that also
    allows a number keeps its int, and a bool is not an int to stringify."""
    tool = _FakeTool(
        "get_reactome_info_by_identifier",
        args={
            "species": {
                "anyOf": [
                    {"type": "string"},
                    {"type": "array", "items": {"type": "string"}},
                    {"type": "null"},
                ]
            },
            "taxon": {"type": ["string", "integer"]},
            "page_size": {"type": "integer"},
            "resource": {"type": "string"},
        },
    )
    strings = _string_arg_names(tool)

    assert strings == {"species", "resource"}
    assert _coerce_int_args(
        {"species": 9606, "taxon": 9606, "page_size": 20, "resource": True}, strings
    ) == {"species": "9606", "taxon": 9606, "page_size": 20, "resource": True}


def test_the_wrapper_hands_the_coerced_value_to_the_tool():
    state = _new_breaker_state()
    tool = _FakeTool("make_changes", args=_ARGS_SCHEMA)
    _wrap_tools_with_circuit_breaker([tool], state)
    asyncio.run(tool.coroutine(justification="j", parameters='[{"parameter_id": "k1"}]'))
    assert tool.calls[-1]["parameters"] == [{"parameter_id": "k1"}]


def test_escalation_survives_an_exception_that_is_not_string_constructible():
    """The real exception here is `langchain.mcp`'s `_MCPToolExecutionError` (the name it kept
    from langchain-mcp-adapters), whose
    __init__ takes a list of content blocks. Rebuilding `type(exc)(str)` yielded an object whose
    message was "string indices must be integers" — losing the escalation AND the original."""

    class _BlockCarrying(Exception):
        def __init__(self, blocks):
            self.blocks = blocks
            super().__init__("".join(b["text"] for b in blocks))  # explodes on a str

    state = _new_breaker_state()
    tool = _FakeTool(
        "make_changes", fail_with=_BlockCarrying([{"type": "text", "text": "server said no"}])
    )
    _wrap_tools_with_circuit_breaker([tool], state)

    async def _call():
        return await tool.coroutine(justification="j")

    for _ in range(_TOOL_ERROR_DISABLE_THRESHOLD - 1):
        asyncio.run(_call())
    text, _ = asyncio.run(_call())
    assert "server said no" in text, "the original error must survive the escalation"
    assert "identical failure" in text
    assert "string indices" not in text


def test_a_never_disabled_tool_still_records_the_streak_for_triage():
    """Not disabling it must not mean not REPORTING it — a run that fails this call 200 times
    used to end with `disabled_tools: []` and nothing in the log to explain a zero score."""
    state = _new_breaker_state()
    for _ in range(_TOOL_ERROR_DISABLE_THRESHOLD):
        _record_error(state, "make_changes", "same error every time")
    assert state["events"], "the streak must leave a trace even when nothing is disabled"
    assert "not disabled" in state["events"][0].lower()


# ------------------------------------------- repeated operation rejections (applied: false)


def _rejection(msg="make_changes: entry 0 is missing a string 'parameter_id'."):
    import json as _json

    return ([{"type": "text", "text": _json.dumps({"applied": False, "error": msg})}], None)


def test_a_repeated_identical_rejection_eventually_escalates():
    """The case the sole-editor escalation could NOT reach: a rejection comes back as a normal
    result (`applied: false`), which is deliberately not a breaker error — so gemini sent 106
    identical ones and luna 246 of 265 with nothing ever telling them to change approach."""
    from gemot.agent.circuit_breaker import _REJECTION_ESCALATE_AT

    state = _new_breaker_state()
    tool = _FakeTool("make_changes")
    tool.coroutine = lambda *_a, **_k: None  # replaced below

    async def _run(**arguments):
        return _rejection()

    tool.coroutine = _run
    _wrap_tools_with_circuit_breaker([tool], state)

    texts = []
    for _ in range(_REJECTION_ESCALATE_AT):
        content, _ = asyncio.run(tool.coroutine(justification="j"))
        texts.append("".join(b["text"] for b in content))

    assert all("th time this call was rejected" not in x for x in texts[:-1])
    assert "th time this call was rejected" in texts[-1]
    assert state["n_rejection_escalations"] == 1


def test_a_different_rejection_resets_the_rejection_streak():
    """Only an IDENTICAL repeat is a loop; a new rejection is ordinary edit/retry progress."""
    from gemot.agent.circuit_breaker import _note_repeated_rejection

    state = _new_breaker_state()
    for i in range(10):
        content, artifact = _rejection(f"distinct error {i}")
        assert _note_repeated_rejection(state, "make_changes", content, artifact) is None
    assert state["n_rejection_escalations"] == 0


def test_a_success_clears_the_rejection_streak():
    from gemot.agent.circuit_breaker import _note_repeated_rejection

    state = _new_breaker_state()
    for _ in range(3):
        c, a = _rejection()
        _note_repeated_rejection(state, "make_changes", c, a)
    ok = ([{"type": "text", "text": '{"applied": true, "committed": true}'}], None)
    assert _note_repeated_rejection(state, "make_changes", *ok) is None
    assert "make_changes" not in state["reject_streaks"]


# ------------------------------------------- call-level outcomes for result.json


def test_call_outcomes_counts_what_edit_rejects_cannot_see():
    """`edit_rejects` (runner/clock.py) tallies only calls that REACH `session.call_tool` and
    return a parseable payload. A mimo-v2.5 run that made 134 failing make_changes calls therefore
    wrote `edit_rejects: {attempted: 0, rejected: 0, reject_rate: null}` — the one metric whose
    job is edit failure, blind to a run that failed every edit. This counts at the client wrapper,
    so the three server-less failure modes are visible.
    """
    from gemot.agent.circuit_breaker import (
        _CALL_CLIENT_INVALID,
        _CALL_DISABLED,
        _CALL_OK,
        _CALL_SERVER_REJECTED,
        _CALL_TOOL_ERROR,
        _note_call,
        call_outcomes,
    )

    state = _new_breaker_state()
    for outcome, n in (
        (_CALL_OK, 2),
        (_CALL_CLIENT_INVALID, 5),
        (_CALL_SERVER_REJECTED, 2),
        (_CALL_TOOL_ERROR, 1),
        (_CALL_DISABLED, 10),
    ):
        for _ in range(n):
            _note_call(state, "make_changes", outcome)

    out = call_outcomes(state, "make_changes")
    assert out["attempted"] == 20
    assert out["failed"] == 18  # everything that is not `ok`
    assert out["fail_rate"] == 0.9
    assert out["by_outcome"][_CALL_CLIENT_INVALID] == 5, (
        "argument-shape failures never reach the server and must still be counted"
    )


def test_call_outcomes_is_empty_not_broken_for_a_tool_never_called():
    from gemot.agent.circuit_breaker import call_outcomes

    out = call_outcomes(_new_breaker_state(), "make_changes")
    assert out == {"attempted": 0, "failed": 0, "fail_rate": None, "by_outcome": {}}


def test_the_wrapper_records_each_outcome_class():
    from gemot.agent.circuit_breaker import call_outcomes

    state = _new_breaker_state()
    ok = _FakeTool("make_changes", args=_ARGS_SCHEMA)
    _wrap_tools_with_circuit_breaker([ok], state)
    asyncio.run(ok.coroutine(justification="j"))
    assert call_outcomes(state, "make_changes")["by_outcome"] == {"ok": 1}

    state2 = _new_breaker_state()
    bad = _FakeTool("make_changes", args=_ARGS_SCHEMA, fail_with=RuntimeError("boom"))
    _wrap_tools_with_circuit_breaker([bad], state2)
    # Returns the error as tool content rather than raising (see the sole-editor note in
    # circuit_breaker) — the outcome must still be RECORDED as a failure for triage.
    content, _ = asyncio.run(bad.coroutine(justification="j"))
    assert "boom" in content
    out = call_outcomes(state2, "make_changes")
    assert out["attempted"] == 1 and out["fail_rate"] == 1.0


# -- a transport failure must still propagate, even for the sole editor ----------------------
def test_a_transport_error_on_the_sole_editor_is_not_swallowed():
    """The never-disable return exists so the MODEL can see and fix a bad call. A closed session or
    an unreachable server is not something it can fix by changing arguments: swallowed as tool
    content, every later call returns the same error, the run burns to the recursion/token limit,
    and it is recorded as a genuine modelling failure with 0 fits — reclassifying infrastructure
    noise as model incapability and biasing every model comparison. It must PROPAGATE, so the
    driver's `_TRANSIENT_ERRORS` resume path can act on it, or failing that so the error class
    lands in `stop_reason` where `experiments.is_noop_infra_abort` excludes the run.
    """
    import anyio

    for exc in (anyio.ClosedResourceError(), anyio.BrokenResourceError()):
        assert _is_transport_error(exc), "predicate must recognise the transport class"
        state = _new_breaker_state()
        tool = _FakeTool("make_changes", fail_with=exc)
        _wrap_tools_with_circuit_breaker([tool], state)
        with pytest.raises(type(exc)):
            asyncio.run(tool.coroutine(justification="j"))
        # Never disabled either — the sole editor must survive a transport blip for a resumed run.
        assert "make_changes" not in state["disabled"]


def test_a_non_transport_failure_on_the_sole_editor_is_still_returned():
    """The exemption above must not re-break the case the return exists for: an argument-shape or
    validation error is still handed back as (content, artifact) so the model can correct it."""
    state = _new_breaker_state()
    tool = _FakeTool("make_changes", fail_with=RuntimeError("bad argument shape"))
    _wrap_tools_with_circuit_breaker([tool], state)
    result = asyncio.run(tool.coroutine(justification="j"))
    assert isinstance(result, tuple) and "bad argument shape" in result[0]


# ------------------------------------------------------------------ verdict-shape protection
def test_a_verdict_payload_is_not_a_breaker_error():
    """Fix-me feedback shaped `{"ok": false, "issues": [...]}` with NO `error` key (the
    draft-era lint shape) must not count as a tool failure: classifying it as one collapsed
    every dirty verdict to the constant signature "tool error" — so the
    different-errors-reset-the-streak protection could never fire, and three consecutive
    verdicts disabled the tool (7 of 18 draft-arm runs, sneyd/fujita A/B). No current tool
    emits the shape; the carve-out is pinned so any future verdict-style payload reads right."""
    from gemot.agent.circuit_breaker import _result_error

    verdict = {"ok": False, "issues": ["Unknown parameter id 'kdim_EGFR'"], "warnings": []}
    assert _result_error(None, {"structured_content": verdict}) is None
    # ...but a REAL failure (ok:false WITH an error) still counts, with its own signature.
    failure = {"ok": False, "error": "server exploded"}
    assert _result_error(None, {"structured_content": failure}) == "server exploded"
    # The text-content fallback path applies the same rule.
    import json as _j

    assert _result_error([{"type": "text", "text": _j.dumps(verdict)}], None) is None
    assert _result_error([{"type": "text", "text": _j.dumps(failure)}], None) == "server exploded"


def test_a_verdict_is_identified_by_issues_not_by_a_missing_error_key():
    """The lint exemption keys POSITIVELY on `issues`. Keying on "no `error` key" instead would
    exempt two shapes that are real failures: an empty `error` (which signs the streak with "",
    handing the model an escalation carrying no error text) and a bare `{"ok": false}` — the
    shape a HOSTED bio server may use, and those are the flaky ones the breaker exists for."""
    from gemot.agent.circuit_breaker import _result_error

    assert _result_error(None, {"structured_content": {"ok": False, "error": ""}}) == "tool error"
    assert _result_error(None, {"structured_content": {"ok": False}}) == "tool error"
    assert _result_error(None, {"structured_content": {"ok": False, "issues": []}}) is None


def test_a_never_disabled_tool_still_escalates_on_the_RESULT_path():
    """A never-disabled tool that reports errors as RESULTS (`{"ok": false, "error": ...}`)
    never raises — so the exception-path escalation cannot fire. Without a result-path
    escalation, an agent stuck on one bad call shape loops on byte-identical feedback to the
    turn limit: no disable, no escalation, nothing it can see."""
    import json as _j

    payload = _j.dumps({"ok": False, "error": "make_changes: `old_string` not found"})
    tool = _FakeTool("make_changes", args=_ARGS_SCHEMA, record=None, fail_with=None)

    async def _always_error(**kw):
        return ([{"type": "text", "text": payload}], {"structured_content": _j.loads(payload)})

    tool.coroutine = _always_error
    state = _new_breaker_state()
    _wrap_tools_with_circuit_breaker([tool], state)
    outs = [
        asyncio.run(tool.coroutine(justification="j")) for _ in range(_TOOL_ERROR_DISABLE_THRESHOLD)
    ]
    assert "make_changes" not in state["disabled"]  # never cut...
    text = "".join(b["text"] for b in outs[-1][0] if isinstance(b, dict))
    assert "REPEATING THIS CALL WILL FAIL AGAIN" in text  # ...but the wording escalates
    assert "old_string` not found" in text  # original error still reaches it
