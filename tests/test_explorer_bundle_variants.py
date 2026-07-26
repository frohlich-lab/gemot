"""The run explorer must see a version produced by a BUNDLED call (issue #114).

Both halves of the explorer's variant trajectory used to key on the ``fit`` tool span. Now that a
coherent change is normally one ``set_model(..., fit=true)`` call, a run can emit **zero** ``fit``
spans — a fit-only filter would rebuild an empty trajectory and flag every variant ``no_turn``.
These tests pin the widened matching (``examples/_harness/explorer/bundle.py``).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

_BUNDLE_PY = (Path(__file__).resolve().parents[1]
              / "examples" / "_harness" / "explorer" / "bundle.py")


def _load_bundle_module():
    """Load bundle.py by file path — it only touches stdlib at import time, so this avoids the
    heavy `_harness` package __init__ (phoenix, petab_mcp)."""
    spec = importlib.util.spec_from_file_location("_explorer_bundle_under_test", _BUNDLE_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


B = _load_bundle_module()


def _span_output(result: dict) -> str:
    """A Phoenix TOOL span's `attributes.output.value` wrapping `result`."""
    return json.dumps({"data": {"content": [{"text": json.dumps(result)}]}})


def _bundled_result(version, antimony_committed=True, **fit):
    """What `set_model(..., fit=true)` returns: edit fields + the fit result nested under `fit`.

    The two `event_timestamp`s differ deliberately: the edit event is stamped when the edit
    applied, the fit event when the version was snapshotted."""
    return {
        "applied": True, "operation": "set_model",
        "committed": antimony_committed, "version": version,
        "justification": "bundled change", "event_timestamp": f"edit-{version}",
        "fit": {"committed": antimony_committed, "version": version,
                "event_timestamp": f"fit-{version}",
                "neg_log_likelihood": fit.get("nll", -8.0), "chi2": 0.5,
                "bic": fit.get("bic", 3.0), "n_estimated": 2, "n_data_points": 6,
                "improvement": {"is_new_best": fit.get("is_new_best", True)}},
    }


def _spans(rows):
    return pd.DataFrame(rows)


_ANT = "model m\n  species A in c;\nend\n"


def test_variants_are_rebuilt_from_bundled_set_model_spans():
    df = _spans([
        {"attributes.tool.name": "set_model", "start_time": 1,
         "attributes.input.value": json.dumps({"antimony": _ANT, "fit": True}),
         "attributes.output.value": _span_output(_bundled_result(1, nll=-8.0, bic=3.0))},
        {"attributes.tool.name": "set_model", "start_time": 2,
         "attributes.input.value": json.dumps({"antimony": _ANT + "\n", "fit": True}),
         "attributes.output.value": _span_output(_bundled_result(2, nll=-9.0, bic=1.0))},
    ])
    variants = B.load_variants_from_spans(df, n_default=6)

    assert [v["version"] for v in variants] == [1, 2]
    # the metrics come from the NESTED fit block, not the top level
    assert variants[0]["nll"] == -8.0 and variants[1]["bic"] == 1.0
    assert variants[0]["k"] == 2 and variants[0]["n"] == 6
    assert variants[1]["antimony"] == _ANT + "\n"   # each bundled span carries its own model
    assert variants[0]["is_new_best"] is True
    # a variant IS a fit snapshot, so it is stamped with the FIT event — not the earlier edit
    # event the bundled result also carries (the version store is keyed on the fit)
    assert [v["timestamp"] for v in variants] == ["fit-1", "fit-2"]


def test_unbundled_edit_span_produces_no_variant():
    """A plain fit-free edit reports no `committed`, so it must not become a variant — only the
    later `fit` span does, and it inherits the preceding edit's Antimony."""
    df = _spans([
        {"attributes.tool.name": "set_model", "start_time": 1,
         "attributes.input.value": json.dumps({"antimony": _ANT}),
         "attributes.output.value": _span_output(
             {"applied": True, "version": None, "pending_fit": True})},
        {"attributes.tool.name": "fit", "start_time": 2,
         "attributes.input.value": json.dumps({}),
         "attributes.output.value": _span_output(
             {"committed": True, "version": 1, "event_timestamp": "fit-1",
              "neg_log_likelihood": -8.0, "bic": 3.0,
              "n_estimated": 2, "n_data_points": 6, "improvement": {"is_new_best": True}})},
    ])
    variants = B.load_variants_from_spans(df, n_default=6)

    assert len(variants) == 1
    assert variants[0]["version"] == 1 and variants[0]["antimony"] == _ANT
    # a standalone fit keeps its scalars (and its timestamp) at the top level
    assert variants[0]["nll"] == -8.0 and variants[0]["timestamp"] == "fit-1"


def test_bundled_change_parameters_span_produces_a_variant():
    df = _spans([
        {"attributes.tool.name": "set_model", "start_time": 1,
         "attributes.input.value": json.dumps({"antimony": _ANT, "fit": True}),
         "attributes.output.value": _span_output(_bundled_result(1))},
        {"attributes.tool.name": "change_parameters", "start_time": 2,
         "attributes.input.value": json.dumps({"changes": [], "fit": True}),
         "attributes.output.value": _span_output(_bundled_result(2, nll=-9.5))},
    ])
    variants = B.load_variants_from_spans(df, n_default=6)

    assert [v["version"] for v in variants] == [1, 2]
    # a bounds-only retuning doesn't change the model, so it keeps the last authored Antimony
    assert variants[1]["antimony"] == _ANT


def test_a_no_op_bundled_fit_produces_no_variant():
    df = _spans([
        {"attributes.tool.name": "set_model", "start_time": 1,
         "attributes.input.value": json.dumps({"antimony": _ANT, "fit": True}),
         "attributes.output.value": _span_output(
             {"applied": True, "committed": False, "version": 1,
              "fit": {"committed": False, "no_change": True, "duplicate_of_version": 1}})},
    ])
    assert B.load_variants_from_spans(df, n_default=6) == []


def test_turns_are_matched_to_variants_committed_by_a_bundled_call():
    variants = [
        {"index": 0, "version": 1, "nll": -8.0},
        {"index": 1, "version": 2, "nll": -9.0},
    ]
    turns = [
        {"i": 0, "tools": [{"name": "set_model",
                            "result": _span_output(_bundled_result(1))}]},
        {"i": 1, "tools": [{"name": "get_fit_diagnosis", "result": _span_output({"ok": True})}]},
        {"i": 2, "tools": [{"name": "change_parameters",
                            "result": _span_output(_bundled_result(2))}]},
    ]
    B._map_variants_to_turns(turns, variants)

    assert turns[0]["variant_index"] == 0
    assert "variant_index" not in turns[1]
    assert turns[2]["variant_index"] == 1
    assert [v["no_turn"] for v in variants] == [False, False]
