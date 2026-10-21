"""The run explorer must see a version produced by a `make_changes` call — and still read the
pre-consolidation traces.

Both halves of the explorer's variant trajectory once keyed on the ``fit`` tool span. A
coherent change is now one ``make_changes`` call, which always fits, so a run emits **zero**
``fit`` spans and a fit-only filter would rebuild an empty trajectory and flag every variant
``no_turn``. The reverse matters too: the explorer's job is reading traces that already exist,
and every sweep recorded before the collapse is a ``set_model``/``change_parameters``/``fit``
trace. These tests pin both directions (``gemot.explorer.bundle``).
"""

from __future__ import annotations

import json


import pandas as pd

from gemot.explorer import bundle as B  # noqa: E402


def _span_output(result: dict) -> str:
    """A Phoenix TOOL span's `attributes.output.value` wrapping `result`."""
    return json.dumps({"data": {"content": [{"text": json.dumps(result)}]}})


def _result(version, committed=True, **fit):
    """What `make_changes` returns: edit fields + the fit result nested under `fit`.

    The two `event_timestamp`s differ deliberately: the edit event is stamped when the edit
    applied, the fit event when the version was snapshotted."""
    return {
        "applied": True,
        "operation": "make_changes",
        "committed": committed,
        "version": version,
        "justification": "one coherent change",
        "event_timestamp": f"edit-{version}",
        "fit": {
            "committed": committed,
            "version": version,
            "event_timestamp": f"fit-{version}",
            "neg_log_likelihood": fit.get("nll", -8.0),
            "chi2": 0.5,
            "bic": fit.get("bic", 3.0),
            "n_estimated": 2,
            "n_data_points": 6,
            "improvement": {"is_new_best": fit.get("is_new_best", True)},
        },
    }


def _spans(rows):
    return pd.DataFrame(rows)


_ANT = "model m\n  species A in c;\nend\n"


def test_variants_are_rebuilt_from_make_changes_spans():
    df = _spans(
        [
            {
                "attributes.tool.name": "make_changes",
                "start_time": 1,
                "attributes.input.value": json.dumps({"antimony": _ANT}),
                "attributes.output.value": _span_output(_result(1, nll=-8.0, bic=3.0)),
            },
            {
                "attributes.tool.name": "make_changes",
                "start_time": 2,
                "attributes.input.value": json.dumps({"antimony": _ANT + "\n"}),
                "attributes.output.value": _span_output(_result(2, nll=-9.0, bic=1.0)),
            },
        ]
    )
    variants = B.load_variants_from_spans(df, n_default=6)

    assert [v["version"] for v in variants] == [1, 2]
    # the metrics come from the NESTED fit block, not the top level
    assert variants[0]["nll"] == -8.0 and variants[1]["bic"] == 1.0
    assert variants[0]["k"] == 2 and variants[0]["n"] == 6
    assert variants[1]["antimony"] == _ANT + "\n"  # each span carries its own model
    assert variants[0]["is_new_best"] is True
    # a variant IS a fit snapshot, so it is stamped with the FIT event — not the earlier edit
    # event the result also carries (the version store is keyed on the fit)
    assert [v["timestamp"] for v in variants] == ["fit-1", "fit-2"]


def test_a_pre_consolidation_trace_still_rebuilds():
    """A sweep recorded before the collapse: a fit-free `set_model` edit (no `committed`, so no
    variant) followed by the standalone `fit` that scored it — which inherits the edit's
    Antimony, since its own input carries none."""
    df = _spans(
        [
            {
                "attributes.tool.name": "set_model",
                "start_time": 1,
                "attributes.input.value": json.dumps({"antimony": _ANT}),
                "attributes.output.value": _span_output(
                    {"applied": True, "version": None, "pending_fit": True}
                ),
            },
            {
                "attributes.tool.name": "fit",
                "start_time": 2,
                "attributes.input.value": json.dumps({}),
                "attributes.output.value": _span_output(
                    {
                        "committed": True,
                        "version": 1,
                        "event_timestamp": "fit-1",
                        "neg_log_likelihood": -8.0,
                        "bic": 3.0,
                        "n_estimated": 2,
                        "n_data_points": 6,
                        "improvement": {"is_new_best": True},
                    }
                ),
            },
        ]
    )
    variants = B.load_variants_from_spans(df, n_default=6)

    assert len(variants) == 1
    assert variants[0]["version"] == 1 and variants[0]["antimony"] == _ANT
    # a standalone fit keeps its scalars (and its timestamp) at the top level
    assert variants[0]["nll"] == -8.0 and variants[0]["timestamp"] == "fit-1"


def test_a_parameters_only_span_produces_a_variant():
    """A bounds-only retuning fits like any other change, so it is a variant — and since its
    input carries no `antimony`, it keeps the last authored model."""
    df = _spans(
        [
            {
                "attributes.tool.name": "make_changes",
                "start_time": 1,
                "attributes.input.value": json.dumps({"antimony": _ANT}),
                "attributes.output.value": _span_output(_result(1)),
            },
            {
                "attributes.tool.name": "make_changes",
                "start_time": 2,
                "attributes.input.value": json.dumps({"parameters": []}),
                "attributes.output.value": _span_output(_result(2, nll=-9.5)),
            },
        ]
    )
    variants = B.load_variants_from_spans(df, n_default=6)

    assert [v["version"] for v in variants] == [1, 2]
    assert variants[1]["antimony"] == _ANT


def test_a_no_op_fit_produces_no_variant():
    df = _spans(
        [
            {
                "attributes.tool.name": "make_changes",
                "start_time": 1,
                "attributes.input.value": json.dumps({"antimony": _ANT}),
                "attributes.output.value": _span_output(
                    {
                        "applied": True,
                        "committed": False,
                        "version": 1,
                        "fit": {"committed": False, "no_change": True, "duplicate_of_version": 1},
                    }
                ),
            },
        ]
    )
    assert B.load_variants_from_spans(df, n_default=6) == []


def test_turns_are_matched_to_the_variants_they_committed():
    variants = [
        {"index": 0, "version": 1, "nll": -8.0},
        {"index": 1, "version": 2, "nll": -9.0},
    ]
    turns = [
        {"i": 0, "tools": [{"name": "make_changes", "result": _span_output(_result(1))}]},
        {"i": 1, "tools": [{"name": "get_fit_diagnosis", "result": _span_output({"ok": True})}]},
        {"i": 2, "tools": [{"name": "make_changes", "result": _span_output(_result(2))}]},
    ]
    B._map_variants_to_turns(turns, variants)

    assert turns[0]["variant_index"] == 0
    assert "variant_index" not in turns[1]
    assert turns[2]["variant_index"] == 1
    assert [v["no_turn"] for v in variants] == [False, False]


def _version_dir(store, version, meta, estimated, n):
    vdir = store / "versions" / f"v{version}"
    vdir.mkdir(parents=True)
    (vdir / "meta.json").write_text(json.dumps({"version": version, **meta}))
    rows = "".join(f"p{i}\t{int(i < estimated)}\n" for i in range(3))
    (vdir / "parameters.tsv").write_text("parameterId\testimate\n" + rows)
    (vdir / "simulations.json").write_text(json.dumps({"n_data_points": n}))


def test_archived_variants_carry_the_stored_bic_and_the_k_it_charged(tmp_path):
    """The archive path recomputed BIC from the estimated rows of `parameters.tsv`, so a model
    with a hardcoded constant scored one ln(n) better than its own fit result said and than the
    span path showed it (#10). v1 has 2 estimated rows and one literal: the server charged k=3.
    v2 is a version the server could not score: it gets no BIC, not a recomputed one."""
    import math

    n, nll = 20, 5.0
    store = tmp_path / ".petab_mcp"
    _version_dir(store, 1, {"neg_log_likelihood": nll, "bic": 2 * nll + 3 * math.log(n)}, 2, n)
    _version_dir(store, 2, {"neg_log_likelihood": nll, "bic": None}, 2, n)

    v1, v2 = B.load_variants(tmp_path, n_data_points=None)

    assert v1["bic"] == 2 * nll + 3 * math.log(n) and v1["k"] == 3 and v1["n"] == n
    assert v2["bic"] is None and v2["k"] == 2
    # A BIC formed with another n gives no whole k: show the rows rather than a wrong count.
    assert B._charged_k(2 * nll + 3 * math.log(n), nll, n + 1) is None
