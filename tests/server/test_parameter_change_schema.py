"""The `make_changes(parameters=...)` contract.

It pins behaviour that is invisible without a test: the schema guards the tool CONTRACT --
the JSON schema is what a model is constrained by, and when this argument was `list[dict[str,
Any]]` it named none of its keys, so 527 of 889 `make_changes` calls in the August benchmark runs
were rejected for guessing them. Nothing failed loudly; the runs just scored badly.
"""

import asyncio

import pytest
from fastmcp import FastMCP

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


# -- the provider list-wrapper artefact (#158 follow-up) --------------------------------------
# Real payloads from the Aug 8-10 sweep. Each killed a whole run: `make_changes` is the only tool
# that can change the model, so it stays available after a tool error, and the model re-emitted the
# same wrapper until the run died (3 runs at 0 fits, 1 recursion-limited).
def test_unwrap_listish_recovers_the_item_wrapper():
    from petab_mcp.server import _unwrap_listish

    payload = {
        "item": [{"parameter_id": "k1", "lower_bound": 1e-6, "upper_bound": 1e3, "scale": "log10"}]
    }
    assert _unwrap_listish(payload) == payload["item"]
    # references saw the same shape
    refs = {"item": ["PMID:1547331 (Fujita 2010)", "Cell 166:461 (reduction)"]}
    assert _unwrap_listish(refs) == refs["item"]


def test_unwrap_listish_recovers_a_json_encoded_array():
    from petab_mcp.server import _unwrap_listish

    assert _unwrap_listish('[{"parameter_id": "k1"}]') == [{"parameter_id": "k1"}]
    assert _unwrap_listish("[]") == []


def test_unwrap_listish_passes_everything_else_through_untouched():
    from petab_mcp.server import _unwrap_listish

    # A genuine list is already correct.
    good = [{"parameter_id": "k1"}]
    assert _unwrap_listish(good) is good
    # A real single entry sent as a bare object is NOT a wrapper -- it must still fail validation
    # rather than being silently reinterpreted, or a typo'd argument becomes a silent no-op.
    bare = {"parameter_id": "k1", "scale": "log10"}
    assert _unwrap_listish(bare) is bare
    # Multi-key objects, wrapper-like keys holding non-lists, and other strings are untouched.
    assert _unwrap_listish({"item": [1], "extra": 2}) == {"item": [1], "extra": 2}
    assert _unwrap_listish({"item": "k1"}) == {"item": "k1"}
    assert _unwrap_listish("not a list") == "not a list"
    assert _unwrap_listish(None) is None
    assert _unwrap_listish("[unparseable") == "[unparseable"


def test_make_changes_still_advertises_an_array_schema():
    """The unwrap must not leak into the CONTRACT: a model is still told `array`, because the
    wrapper is a malformed encoding we tolerate, not a second valid way to call the tool."""
    import asyncio

    from fastmcp import FastMCP
    from petab_mcp.server import ParameterChange, _unwrap_listish
    from typing import Annotated
    from pydantic import BeforeValidator

    mcp = FastMCP("wrapper-probe")

    @mcp.tool()
    def probe(
        parameters: Annotated[
            list[ParameterChange] | None, BeforeValidator(_unwrap_listish)
        ] = None,
    ) -> dict:
        """probe"""
        return {}

    tool = asyncio.run(mcp.list_tools())[0]
    s = getattr(tool, "inputSchema", None) or getattr(tool, "parameters")
    entry = s["properties"]["parameters"]
    branches = entry.get("anyOf", [entry])
    array_branch = next(b for b in branches if b.get("type") == "array")
    assert array_branch["type"] == "array"  # still an array, not anyOf[array, object]
    assert not any(b.get("type") == "object" for b in branches)


def test_default_bounds_never_dip_below_zero_for_a_zero_nominal():
    """A zero nominal is an initial CONCENTRATION in practice (reconcile minting <sid>_init for
    an empty pool). The old bracket rule (`nominal - 1.0`) gave every such row a -1.0 lower
    bound — fixed rows, so never sampled, but visible in every parameter read and draft TSV,
    where it reads as a live negative-concentration box (and derailed a segfault investigation).
    Zero is in [0, ub]; the bracket contract holds without dipping."""
    from petab_mcp.parameters import _default_bounds

    lb, ub = _default_bounds(0)
    assert lb == 0.0 and ub > 0
    lb, ub = _default_bounds(0.0)
    assert lb == 0.0
    # a genuinely negative nominal keeps the dip — correct for log-space/difference parameters
    lb, ub = _default_bounds(-2.5)
    assert lb <= -3.5
    # positive nominals bracket as before
    lb, ub = _default_bounds(5.0)
    assert lb <= 5.0 <= ub
