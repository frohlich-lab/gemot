"""Provider content-filter and transient-error classification of a failed model call.

This pins behaviour that is invisible without a test: a refusal must stop the run with a reason
the sweep can count, and a transient 5xx must resume it. The ``make_changes(parameters=...)``
schema contract this file used to share is pinned in PEtab-MCP.
"""

import pytest

from gemot.agent.driver import _content_filter_stop_reason


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
    for exc in (
        ValueError("{'message': 'invalid model id', 'code': 400}"),
        RuntimeError("Recursion limit of 250 reached"),
        # a bare mention of moderation must NOT be relabelled -- the marker list is
        # documented as narrow, so it matches explicit refusal phrases only
        ValueError("{'message': 'moderation service timed out', 'code': 400}"),
        TimeoutError("stalled"),
    ):
        assert _content_filter_stop_reason(exc) is None


# ------------------------------------------------- unclassified provider 5xx (resume, not die)


def test_the_observed_504_is_classified_as_transient():
    """Verbatim from a mimo-v2.5 run that died 25 minutes in with 5 fits and a committed model.

    OpenRouter returns HTTP 200 with an `{"error": ...}` body for some upstream failures, so the
    client raises a bare ValueError — invisible to the OpenAI client's own 5xx retry AND to
    `_TRANSIENT_ERRORS`. It reached the terminal handler and ended a healthy run.
    """
    from gemot.agent.driver import _transient_provider_reason

    reason = _transient_provider_reason(
        ValueError("{'message': 'The operation was aborted', 'code': 504}")
    )
    assert reason is not None and reason.startswith("provider 504")


@pytest.mark.parametrize("code", [500, 502, 503, 504, 529, 599])
def test_any_5xx_resumes(code):
    from gemot.agent.driver import _transient_provider_reason

    assert (
        _transient_provider_reason(ValueError(f"{{'message': 'upstream', 'code': {code}}}"))
        is not None
    )
    assert (
        _transient_provider_reason(ValueError(f'{{"message": "upstream", "code": {code}}}'))
        is not None
    ), "double-quoted too"


@pytest.mark.parametrize("code", [400, 401, 403, 404, 422, 429])
def test_4xx_is_never_treated_as_transient(code):
    """A 4xx is deterministic: retrying re-sends the same request and fails the same way. The
    content refusal is the case that matters — retrying it would burn three attempts to reach
    the same terminal answer."""
    from gemot.agent.driver import _transient_provider_reason

    assert _transient_provider_reason(ValueError(f"{{'message': 'nope', 'code': {code}}}")) is None


def test_a_content_refusal_stays_terminal_even_though_it_carries_a_code():
    """Both classifiers see this exception; the content one must win, and it must not be
    resumed."""
    from gemot.agent.driver import _content_filter_stop_reason, _transient_provider_reason

    exc = ValueError(
        "{'message': 'This content was flagged for possible biological risk.', 'code': 400}"
    )
    assert _content_filter_stop_reason(exc) is not None
    assert _transient_provider_reason(exc) is None


def test_an_error_with_no_code_is_left_alone():
    from gemot.agent.driver import _transient_provider_reason

    for exc in (
        RuntimeError("Recursion limit of 250 reached"),
        TimeoutError("stalled"),
        ValueError("something went wrong"),
    ):
        assert _transient_provider_reason(exc) is None
