"""gemot's copies of the server's constants are the server's values.

A threshold both sides apply cannot be shared as one object: gemot must import without the server
(:mod:`gemot.server_api`), so it keeps a copy beside the code that uses it, and nothing but these
tests stops the copy drifting. A drift is a real defect, not a style one: the server's adequacy
gate and the judge's success gate would disagree on what an implausible noise sigma is, and a run
the server let finish would be failed afterwards (or the reverse). The fit budget's copies are
pinned the same way, in ``test_fit_budget.py``.
"""

from __future__ import annotations

import pytest

from gemot.runner import scoring

from petab_mcp import constants


@pytest.mark.parametrize(
    "name",
    [
        "NOISE_SIGMA_COLLAPSE_FRACTION",
        "NOISE_SIGMA_REFERENCE_FACTOR",
        "NOISE_SIGMA_BAND_ALPHA",
        "NOISE_SIGMA_BAND_MIN_DF",
    ],
)
def test_the_judges_noise_sigma_thresholds_are_the_servers(name):
    assert getattr(scoring, f"_{name}") == getattr(constants, name)
