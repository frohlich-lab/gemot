"""The `make_changes(parameters=...)` contract, and provider content-filter classification.

Both pin behaviour that is invisible without a test. The schema one guards the tool CONTRACT --
the JSON schema is what a model is constrained by, and when this argument was `list[dict[str,
Any]]` it named none of its keys, so 527 of 889 `make_changes` calls in the August benchmark runs
were rejected for guessing them. Nothing failed loudly; the runs just scored badly.
"""
import asyncio

import pytest
from fastmcp import FastMCP

from petab_harness.agent.driver import _content_filter_stop_reason
from petab_mcp.server import ParameterChange, _parameter_entries


def _schema():
    """The schema a model actually receives for a `list[ParameterChange]` argument."""
    mcp = FastMCP("schema-probe")

    @mcp.tool()
    def probe(parameters: list[ParameterChange] | None = None) -> dict:
        """probe"""
        return {}

    tool = asyncio.run(mcp.list_tools())[0]
    s = getattr(tool, "inputSchema", None) or getattr(tool, "parameters")
    entry = s["properties"]["parameters"]
    for branch in entry.get("anyOf", [entry]):
        if branch.get("type") == "array":
            return branch["items"]
    raise AssertionError(f"no array branch in {entry}")


# ------------------------------------------------------------------ the tool contract

def test_schema_names_parameter_id_as_required():
    """The failure that cost 484 rejections: `parameter_id` existed only in prose."""
    items = _schema()
    assert items["required"] == ["parameter_id"], (
        "parameter_id must be REQUIRED in the schema, not only described in the docstring"
    )


def test_schema_names_every_allowed_key():
    props = _schema()["properties"]
    assert set(props) == {"parameter_id", "lower_bound", "upper_bound", "scale"}


def test_schema_forbids_extra_keys():
    """`estimate`/`nominalValue`/`lowerBound` were invented by models and must stay rejected.

    Also the reason `extra="forbid"` is set: pydantic's default would silently DROP them, turning
    a loud rejection into a change that quietly did less than the model asked for.
    """
    assert _schema()["additionalProperties"] is False
    with pytest.raises(Exception):
        ParameterChange(parameter_id="k1", nominalValue=3.0)


def test_schema_constrains_scale_to_petab_values():
    """`parameterScale: "linear"` was a live rejection; the enum makes it unrepresentable."""
    scale = _schema()["properties"]["scale"]
    enums = [b["enum"] for b in scale.get("anyOf", [scale]) if "enum" in b]
    assert enums and set(enums[0]) == {"lin", "log", "log10"}
    with pytest.raises(Exception):
        ParameterChange(parameter_id="k1", scale="linear")


def test_model_facing_description_stays_short():
    """FastMCP copies the class docstring into the schema, so it is shipped to the model on every
    call — maintainer rationale belongs in a comment, not there."""
    assert len(_schema().get("description", "")) < 200


# ------------------------------------------------------------------ boundary conversion

def test_entries_convert_to_plain_dicts_for_operations():
    out = _parameter_entries([ParameterChange(parameter_id="k1", lower_bound=1e-3, scale="log10")])
    assert out == [{"parameter_id": "k1", "lower_bound": 1e-3, "scale": "log10"}]


def test_entries_pass_dicts_through_and_none_stays_none():
    assert _parameter_entries([{"parameter_id": "k2"}]) == [{"parameter_id": "k2"}]
    assert _parameter_entries(None) is None


# ------------------------------------------------------------------ content-filter classification

def test_classifies_the_refusal_seen_in_the_sweep():
    """Verbatim from gpt-5.6-luna-pro on a PEtab immunology model."""
    exc = ValueError(
        "{'message': 'This content was flagged for possible biological risk. If this seems "
        "wrong, try rephrasing your request.', 'code': 400}"
    )
    reason = _content_filter_stop_reason(exc)
    assert reason is not None and reason.startswith("content_filtered: ")
    assert "biological risk" in reason
    assert "'code'" not in reason, "should surface the provider sentence, not the JSON envelope"


def test_the_closing_quote_matches_the_opening_one():
    """Taking whichever of `',` / `",` comes first cuts at a quote inside the message body."""
    exc = ValueError(
        "{'message': 'This content was flagged: the phrase \"biological risk\", as written, "
        "applies here.', 'code': 400}"
    )
    reason = _content_filter_stop_reason(exc)
    assert reason is not None
    assert reason.rstrip().endswith("applies here.")
    assert "'code'" not in reason


def test_a_payload_with_no_trailing_comma_does_not_leak_braces():
    exc = ValueError("{'message': 'content policy: refused'}")
    reason = _content_filter_stop_reason(exc)
    assert reason == "content_filtered: content policy: refused"


def test_does_not_swallow_unrelated_400s():
    """Narrow on purpose: a plain bad request must keep its own exception text."""
    for exc in (ValueError("{'message': 'invalid model id', 'code': 400}"),
                RuntimeError("Recursion limit of 250 reached"),
                # a bare mention of moderation must NOT be relabelled -- the marker list is
                # documented as narrow, so it matches explicit refusal phrases only
                ValueError("{'message': 'moderation service timed out', 'code': 400}"),
                TimeoutError("stalled")):
        assert _content_filter_stop_reason(exc) is None


# ------------------------------------------------- unclassified provider 5xx (resume, not die)

def test_the_observed_504_is_classified_as_transient():
    """Verbatim from a mimo-v2.5 run that died 25 minutes in with 5 fits and a committed model.

    OpenRouter returns HTTP 200 with an `{"error": ...}` body for some upstream failures, so the
    client raises a bare ValueError — invisible to the OpenAI client's own 5xx retry AND to
    `_TRANSIENT_ERRORS`. It reached the terminal handler and ended a healthy run.
    """
    from petab_harness.agent.driver import _transient_provider_reason

    reason = _transient_provider_reason(
        ValueError("{'message': 'The operation was aborted', 'code': 504}"))
    assert reason is not None and reason.startswith("provider 504")


@pytest.mark.parametrize("code", [500, 502, 503, 504, 529, 599])
def test_any_5xx_resumes(code):
    from petab_harness.agent.driver import _transient_provider_reason

    assert _transient_provider_reason(
        ValueError(f"{{'message': 'upstream', 'code': {code}}}")) is not None
    assert _transient_provider_reason(
        ValueError(f'{{"message": "upstream", "code": {code}}}')) is not None, "double-quoted too"


@pytest.mark.parametrize("code", [400, 401, 403, 404, 422, 429])
def test_4xx_is_never_treated_as_transient(code):
    """A 4xx is deterministic: retrying re-sends the same request and fails the same way. The
    content refusal is the case that matters — retrying it would burn three attempts to reach
    the same terminal answer."""
    from petab_harness.agent.driver import _transient_provider_reason

    assert _transient_provider_reason(
        ValueError(f"{{'message': 'nope', 'code': {code}}}")) is None


def test_a_content_refusal_stays_terminal_even_though_it_carries_a_code():
    """Both classifiers see this exception; the content one must win, and it must not be
    resumed."""
    from petab_harness.agent.driver import _content_filter_stop_reason, _transient_provider_reason

    exc = ValueError(
        "{'message': 'This content was flagged for possible biological risk.', 'code': 400}")
    assert _content_filter_stop_reason(exc) is not None
    assert _transient_provider_reason(exc) is None


def test_an_error_with_no_code_is_left_alone():
    from petab_harness.agent.driver import _transient_provider_reason

    for exc in (RuntimeError("Recursion limit of 250 reached"),
                TimeoutError("stalled"),
                ValueError("something went wrong")):
        assert _transient_provider_reason(exc) is None
