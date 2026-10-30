"""OpenRouter's 402 ends a run as ``provider_credit``, not as raw exception text (gemot#76).

OpenRouter reserves a request's whole ``max_tokens`` against the key's credit, so on perica_egf
most GLM runs ended on "can only afford 91509" with credit still left, each stop_reason a unique
string. The run-loop half of this is in test_run_loop_no_model.py (panel) and test_barebones.py
(null arm); these pin the classifier on each shape the client raises it in.
"""

from __future__ import annotations

import pytest

from gemot import experiments
from gemot.agent import driver

openai = pytest.importorskip("openai", reason="agent stack not installed")
httpx = pytest.importorskip("httpx", reason="agent stack not installed")

_AFFORD = "You requested up to 131072 tokens, but can only afford 91509."
_REQUEST = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")


def _status_error(code: int, message: str = _AFFORD):
    """What the OpenAI client raises for an HTTP ``code``, built by the client's own mapping."""
    response = httpx.Response(
        code, request=_REQUEST, json={"error": {"message": message, "code": code}}
    )
    return openai.OpenAI(api_key="x", base_url="http://x")._make_status_error_from_response(
        response
    )


def test_an_http_402_is_provider_credit():
    exc = _status_error(402)
    assert type(exc) is openai.APIStatusError  # no subclass of its own, unlike 400/429/5xx
    assert driver._is_provider_credit(exc)


def test_a_402_mid_stream_is_provider_credit_not_an_outage():
    """A mid-stream error event raises the bare APIError, which the outage path would wait out
    eight times over; the credit check has to see it first."""
    exc = openai.APIError(_AFFORD, _REQUEST, body={"code": 402, "message": _AFFORD})
    assert driver._is_provider_credit(exc)
    assert driver._transient_provider_reason(exc) is not None  # why the order matters


def test_a_402_in_an_http_200_error_body_is_provider_credit():
    assert driver._is_provider_credit(ValueError(f"{{'message': '{_AFFORD}', 'code': 402}}"))


@pytest.mark.parametrize(
    "exc",
    [
        _status_error(429, "rate limited"),
        _status_error(400, "invalid model id"),
        ValueError("{'message': 'upstream', 'code': 504}"),
        # gemot#44: a 402 inside a token count is not the code
        RuntimeError("token budget: 20,402,551 tokens over 20,000,000"),
    ],
)
def test_other_errors_are_not_provider_credit(exc):
    assert not driver._is_provider_credit(exc)


def test_a_provider_credit_stop_is_still_an_infrastructure_abort():
    """The raw 402 text was one; the bare label must stay one, so a no-work run is still dropped
    from the aggregation and a late one kept as a partial result."""
    assert experiments._is_infra_abort(driver.PROVIDER_CREDIT)
    assert experiments.is_noop_infra_abort(
        {"stop_reason": driver.PROVIDER_CREDIT, "n_ops": 0, "n_fits": 0}
    )
    assert not experiments.is_noop_infra_abort(
        {"stop_reason": driver.PROVIDER_CREDIT, "n_ops": 4, "n_fits": 3}
    )
