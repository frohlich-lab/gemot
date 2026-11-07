"""Assemble one run's view model from the on-disk version store + the Phoenix trace.

``build_bundle`` returns a JSON-able dict:

    {
      "run_id", "meta": {...result.json...},
      "variants": [ {index, version, nll, bic, chi2, is_new_best, timestamp,
                     antimony, params:[...], k, justification,
                     starts:{n_starts, n_at_best, exitflag_counts}, statements} ],
      "turns":    [ {i, kind:"lead", items:[...], tokens:{...}, model, variant_index} ],
      "totals":   {prompt, completion, cache_read, total, lead_turns, analyst_turns, tool_calls},
      "phoenix":  {ok, trace_id, note, complete}
    }

The conversation (turns/tokens/analyst sub-agents) comes from Phoenix — queried by the indexed
``trace_id`` (fast; unfiltered pulls time out). The model-variant + score trajectory comes from the
run's ``.petab_mcp/`` version store, each version's BIC as its ``meta.json`` stored it. The two are
stitched by aligning each fitting tool span (see ``_FITTING_TOOLS``) with the version it produced.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from ..analysis import bic_of, read_version_json, span_error

# ---------------------------------------------------------------- Phoenix message parsing

# CHAIN/middleware span names that are langgraph plumbing, not conversation.
_PLUMBING = {"tools", "model", "LangGraph", "model_build_run"}


def _msg_text(msg: dict) -> str:
    """Text of one OpenInference message dict (``message.content`` or ``message.contents[]``)."""
    if not isinstance(msg, dict):
        return ""
    c = msg.get("message.content")
    if isinstance(c, str) and c.strip():
        return c
    parts = msg.get("message.contents") or []
    out = []
    for p in parts:
        if isinstance(p, dict) and p.get("message_content.type") == "text":
            out.append(str(p.get("message_content.text", "")))
    return "\n".join(out)


def _assistant_from_output(output_messages: Any) -> dict:
    """Extract {text, tool_calls[]} from an LLM span's ``llm.output_messages``."""
    text, calls = "", []
    for m in output_messages or []:
        if not isinstance(m, dict):
            continue
        text = text or _msg_text(m)
        for tc in m.get("message.tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            args = tc.get("tool_call.function.arguments")
            try:
                args = json.loads(args) if isinstance(args, str) else args
            except Exception:
                pass
            calls.append(
                {
                    "id": tc.get("tool_call.id"),
                    "name": tc.get("tool_call.function.name"),
                    "args": args,
                }
            )
    return {"text": text, "tool_calls": calls}


def _agent_name(metadata) -> str:
    """Which agent made an LLM call, from the span's LangChain ``metadata.lc_agent_name``:
    ``lead``, or the analyst's name, which is also the ``subagent_type`` it was delegated under.

    Not from the system prompt in ``llm.input_messages``, as it used to be: the tracer keeps only
    each call's new messages (``gemot.tracing``), so most LLM spans no longer carry one."""
    metadata = _jload(metadata)
    return str(metadata.get("lc_agent_name") or "") if isinstance(metadata, dict) else ""


def _emits_task(out_msgs) -> bool:
    """Whether an LLM span's output contains a ``task`` tool call — only the lead delegates,
    so this identifies lead spans even when the trace tree is incomplete (a live run)."""
    return any(tc.get("name") == "task" for tc in _assistant_from_output(out_msgs)["tool_calls"])


def _match_pending_subagent(agent: str, emitted: list[dict]) -> tuple[str | None, str | None]:
    """(subagent_type, description) for an in-progress analyst: its agent name, and the
    description of the lead's latest ``task`` call for that subagent_type, if there is one.

    The latest, as the lead waits on its delegations: matching the first gave a running analyst
    the description of the first delegation of its type the trace held (#125)."""
    for a in reversed(emitted):
        if a.get("subagent_type") == agent:
            return agent, a.get("description")
    return agent or None, None


def _num(v) -> float | None:
    try:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _token_total(
    total: float | None, prompt: float | None, completion: float | None
) -> float | None:
    """An LLM span's total tokens, or its prompt plus completion where it recorded none. A turn the
    thinking hand-off put together from two calls carried no total until #125, so the traces before
    it have spans like that, and summing each count apart left a run's total short of the two."""
    if total is not None or (prompt is None and completion is None):
        return total
    return (prompt or 0) + (completion or 0)


def _jload(v):
    """Best-effort: a JSON string -> obj; already-obj passes through; else the raw value."""
    if isinstance(v, str):
        try:
            return json.loads(v)
        except Exception:
            return v
    return v


class _Trace:
    """The span tree for one trace, with helpers to walk lead vs analyst subtrees."""

    def __init__(self, df):
        self.df = df
        self.by_id: dict[str, dict] = {}
        self.children: dict[str, list[str]] = {}
        kind_c = "attributes.openinference.span.kind"
        for span_id, row in df.iterrows():
            r = row.to_dict()
            prompt = _num(r.get("attributes.llm.token_count.prompt"))
            completion = _num(r.get("attributes.llm.token_count.completion"))
            self.by_id[span_id] = {
                "id": span_id,
                "parent": (None if _isnull(r.get("parent_id")) else r.get("parent_id")),
                "name": r.get("name"),
                "kind": r.get(kind_c),
                "start": r.get("start_time"),
                "in": r.get("attributes.input.value"),
                "out": r.get("attributes.output.value"),
                "tool_name": r.get("attributes.tool.name"),
                "agent": _agent_name(r.get("attributes.metadata")),
                "out_msgs": r.get("attributes.llm.output_messages"),
                "model": r.get("attributes.llm.model_name"),
                # The status message of a span that ended in ERROR: a delegation that died on a
                # dropped connection and was run again otherwise reads as a duplicate with no
                # result (#125).
                "error": None if _isnull(r.get("status_code")) else span_error(r),
                "tok_prompt": prompt,
                "tok_completion": completion,
                "tok_total": _token_total(
                    _num(r.get("attributes.llm.token_count.total")), prompt, completion
                ),
                "tok_cache": _num(r.get("attributes.llm.token_count.prompt_details.cache_read")),
            }
        for sid, s in self.by_id.items():
            self.children.setdefault(s["parent"], []).append(sid)

    def _sorted(self, ids: list[str]) -> list[dict]:
        return sorted((self.by_id[i] for i in ids), key=lambda s: (s["start"] is None, s["start"]))

    def subtree_ids(self, root_id: str) -> set[str]:
        seen, frontier = set(), [root_id]
        while frontier:
            nxt = []
            for s in frontier:
                for k in self.children.get(s, []):
                    if k not in seen:
                        seen.add(k)
                        nxt.append(k)
            frontier = nxt
        return seen

    def timeline(self, scope_ids: list[str], exclude: set[str]) -> list[dict]:
        """Ordered LLM + meaningful-TOOL spans within ``scope_ids``, skipping ``exclude`` and
        plumbing. Each LLM span becomes an 'assistant' item; each TOOL a 'tool' item."""
        items = []
        for s in self._sorted([i for i in scope_ids if i not in exclude]):
            if s["kind"] == "LLM":
                a = _assistant_from_output(s["out_msgs"])
                items.append(
                    {
                        "type": "assistant",
                        "text": a["text"],
                        "tool_calls": a["tool_calls"],
                        "model": s["model"],
                        "tokens": {
                            "prompt": s["tok_prompt"],
                            "completion": s["tok_completion"],
                            "cache_read": s["tok_cache"],
                            "total": s["tok_total"],
                        },
                        "error": s["error"],
                        "_id": s["id"],
                        "start": s["start"],
                    }
                )
            elif s["kind"] == "TOOL":
                items.append(
                    {
                        "type": "tool",
                        "name": s["tool_name"] or s["name"],
                        "args": _jload(s["in"]),
                        "result": _jload(s["out"]),
                        "error": s["error"],
                        "_id": s["id"],
                        "start": s["start"],
                    }
                )
        return items


def _isnull(v) -> bool:
    try:
        return v is None or (isinstance(v, float) and math.isnan(v))
    except Exception:
        return v is None


#: Spans asked of Phoenix per request. Phoenix returns a query's spans newest first and stops at
#: its ``limit``, which the client defaults to 1000: fontaine runs 5 and 8 of sweep 63372750 came
#: back with exactly 1000 spans and without their earliest, the root among them, and their orphaned
#: analyst turns were drawn as delegations still running in turn 0 (#125). So
#: :func:`_fetch_trace_df` pages back through the trace, and this number only sets how many
#: requests that takes.
_SPAN_PAGE = 1000


def _fetch_trace_df(trace_id: str, base_url: str, project: str):
    """Every span of one trace, in the dataframe shape Phoenix returns.

    Each page asks for the spans that started before the earliest one read so far, plus that
    instant itself: spans that share a start time can straddle a page's end, so the boundary is
    read again and the spans already held are dropped. If a whole page shares one start time it
    adds nothing, and the same page is asked for again at twice the size. The trace has been read
    to its first span once a page comes back short of its limit."""
    import pandas as pd
    from phoenix.client import Client
    from phoenix.client.types.spans import SpanQuery

    spans = Client(base_url=base_url).spans
    q = SpanQuery().where(f"trace_id == '{trace_id}'")
    pages, seen, before, limit = [], set(), None, _SPAN_PAGE
    while True:
        page = spans.get_spans_dataframe(
            query=q, project_name=project, end_time=before, limit=limit, timeout=120
        )
        new = page[~page.index.isin(seen)]
        if len(new):
            pages.append(new)
            seen.update(new.index)
        if len(page) < limit:
            return pd.concat(pages) if pages else page
        if new.empty:
            limit *= 2
            continue
        # Phoenix keeps start times to the microsecond and ``end_time`` is exclusive.
        before = (page["start_time"].min() + pd.Timedelta(microseconds=1)).to_pydatetime()


def load_turns(
    trace_id: str, base_url: str, project: str, df=None, *, finished: bool = False
) -> tuple[list[dict], dict, dict]:
    """Return (turns, totals, phoenix_status) for a trace. Degrades to empty on any failure.

    ``df`` may be a pre-fetched spans dataframe (so a caller can reuse it for variant
    reconstruction); when None it is fetched here. ``finished`` says the run has ended, so none
    of its delegations can still be running: an analyst span outside every ``task`` span then
    means Phoenix lacks part of the trace, and ``phoenix_status`` says so with ``complete``
    false and a note.
    """
    status = {"ok": False, "trace_id": trace_id, "note": "", "complete": True}
    if df is None:
        try:
            df = _fetch_trace_df(trace_id, base_url, project)
        except Exception as exc:  # Phoenix down / trace not ingested / timeout
            status["note"] = f"could not read Phoenix trace: {type(exc).__name__}: {exc}"
            return [], _empty_totals(), status
    if df is None or len(df) == 0:
        status["note"] = "no spans found for this trace_id in Phoenix"
        return [], _empty_totals(), status

    tr = _Trace(df)
    # analyst delegations: each `task` TOOL span owns a subtree = one analyst trajectory
    task_ids = [sid for sid, s in tr.by_id.items() if s["tool_name"] == "task"]
    analyst_subtrees = {tid: tr.subtree_ids(tid) for tid in task_ids}
    all_analyst_ids = set().union(*analyst_subtrees.values()) if analyst_subtrees else set()

    # In-progress delegations (LIVE): while an analyst subagent is still running, its `task`
    # TOOL span hasn't ended, so it isn't in the trace yet and its analyst LLM spans have no
    # task ancestor — the tree alone would drop them into the lead timeline. The lead is the
    # only agent that emits `task` tool calls, so its agent name is a reliable signature:
    # any LLM span NOT under a completed task whose agent isn't the lead is an in-progress
    # analyst. (Empty lead set ⇒ no delegation has started ⇒ no orphans.)
    lead_agents = {
        s["agent"]
        for sid, s in tr.by_id.items()
        if s["kind"] == "LLM" and _emits_task(s["out_msgs"])
    }
    orphan_analyst_ids = []
    if lead_agents:
        for sid, s in tr.by_id.items():
            if s["kind"] == "LLM" and sid not in all_analyst_ids:
                if s["agent"] and s["agent"] not in lead_agents:
                    orphan_analyst_ids.append(sid)
    # A finished run's trace has its root, and every analyst span sits under a `task` span. When
    # one does not, the spans that would hold it are missing (#125: a read cut at 1000 spans lost
    # the earliest, root and all), and so is the lead turn that delegated it, which started
    # before. Its turns are then left out rather than drawn as a delegation still running.
    rooted = any(s["parent"] is None for s in tr.by_id.values())
    if finished and (orphan_analyst_ids or not rooted):
        n = len(orphan_analyst_ids)
        status["complete"] = False
        status["note"] = (
            f"Phoenix holds only part of this trace ({len(tr.by_id)} spans"
            + ("" if rooted else ", not its root")
            + "): the turns whose spans are missing are not shown"
            + (
                f", nor {n} analyst turn{'' if n == 1 else 's'} whose delegation is among them"
                if n
                else ""
            )
            + "."
        )

    # Lead timeline = every span EXCEPT those inside a completed task subtree, the in-progress
    # analyst spans just identified (task spans themselves stay, to render the delegation),
    # minus plumbing.
    exclude = (
        all_analyst_ids
        | set(orphan_analyst_ids)
        | {
            sid
            for sid, s in tr.by_id.items()
            if s["kind"] == "CHAIN"
            or (s["name"] in _PLUMBING)
            or (s["kind"] == "AGENT")  # middleware "*.before_agent" etc.
        }
    )
    lead_items = tr.timeline(list(tr.by_id.keys()), exclude)
    inner_ex = {
        s
        for s, sp in tr.by_id.items()
        if sp["kind"] in ("CHAIN", "AGENT") or sp["name"] in _PLUMBING
    }

    analyst_turn_count = 0
    analyst_tokens = _zero()
    for it in lead_items:
        if it["type"] == "tool" and it["name"] == "task":
            tid = _match_task_span(tr, task_ids, it)
            if tid is not None:
                sub = analyst_subtrees[tid]
                sub_items = tr.timeline(list(sub), inner_ex)
                args = it.get("args") or {}
                it["delegation"] = {
                    "subagent_type": (
                        args.get("subagent_type") if isinstance(args, dict) else None
                    ),
                    "description": (args.get("description") if isinstance(args, dict) else None),
                    "items": sub_items,
                }
                analyst_turn_count += sum(1 for x in sub_items if x["type"] == "assistant")
                _accumulate(analyst_tokens, sub_items)

    # Render each in-progress analyst as its own delegation block (grouped by persona, since
    # the panel spawns the three analysts concurrently) attached to the lead's spawning turn,
    # rather than letting its turns show up as lead turns.
    if orphan_analyst_ids and not finished:
        emitted = [
            tc.get("args")
            for it in lead_items
            if it["type"] == "assistant"
            for tc in (it.get("tool_calls") or [])
            if tc.get("name") == "task" and isinstance(tc.get("args"), dict)
        ]
        by_agent: dict[str, list[str]] = {}
        for sid in orphan_analyst_ids:
            by_agent.setdefault(tr.by_id[sid]["agent"], []).append(sid)
        for name, ids in by_agent.items():
            sub_items = tr.timeline(ids, inner_ex)
            st, desc = _match_pending_subagent(name, emitted)
            lead_items.append(
                {
                    "type": "tool",
                    "name": "task",
                    "_id": None,
                    "result": None,
                    "args": {"subagent_type": st, "description": desc},
                    "start": min((tr.by_id[i]["start"] for i in ids), default=None),
                    "delegation": {
                        "subagent_type": st,
                        "description": desc,
                        "items": sub_items,
                        "in_progress": True,
                    },
                }
            )
            analyst_turn_count += sum(1 for x in sub_items if x["type"] == "assistant")
            _accumulate(analyst_tokens, sub_items)
        # Re-sort by start so each in-progress delegation lands right after the lead turn that
        # spawned it, not merely at the end: _group_turns attaches a tool to the PRECEDING
        # assistant, so chronological order decides which turn owns the delegation.
        lead_items.sort(key=lambda it: (it.get("start") is None, str(it.get("start"))))

    turns = _group_turns(lead_items)
    totals = _tally(lead_items, analyst_tokens, analyst_turn_count)
    status["ok"] = True
    return turns, totals, status


def _match_task_span(tr: "_Trace", task_ids: list[str], tool_item: dict) -> str | None:
    """Map a lead 'task' tool item to its TOOL span (same subagent_type, nearest start time)."""
    args = tool_item.get("args") or {}
    want = args.get("subagent_type") if isinstance(args, dict) else None
    cand = []
    for tid in task_ids:
        s = tr.by_id[tid]
        sargs = _jload(s["in"]) or {}
        st = sargs.get("subagent_type") if isinstance(sargs, dict) else None
        if want is None or st == want:
            cand.append(tid)
    if not cand:
        cand = task_ids
    # nearest by start time
    t0 = tool_item.get("start")
    cand.sort(key=lambda tid: abs(_delta(tr.by_id[tid]["start"], t0)))
    return cand[0] if cand else None


def _delta(a, b) -> float:
    try:
        return abs((a - b).total_seconds())
    except Exception:
        return 0.0


def _group_turns(items: list[dict]) -> list[dict]:
    """Group the flat lead timeline into turns: each assistant + the tools it triggered."""
    turns, cur = [], None
    for it in items:
        if it["type"] == "assistant":
            if cur is not None:
                turns.append(cur)
            cur = {
                "kind": "lead",
                "text": it["text"],
                "model": it["model"],
                "tokens": it["tokens"],
                "error": it["error"],
                "tools": [],
                "start": it["start"],
            }
        else:  # tool
            if cur is None:  # a tool before any assistant (rare) -> standalone turn
                cur = {
                    "kind": "lead",
                    "text": "",
                    "model": None,
                    "tokens": _empty_tokens(),
                    "tools": [],
                    "start": it["start"],
                }
            cur["tools"].append(
                {
                    k: it[k]
                    for k in ("name", "args", "result", "error", "delegation", "start")
                    if k in it
                }
            )
    if cur is not None:
        turns.append(cur)
    for i, t in enumerate(turns):
        t["i"] = i
    return turns


def _empty_tokens():
    return {"prompt": None, "completion": None, "cache_read": None, "total": None}


def _zero():
    return {"prompt": 0, "completion": 0, "cache_read": 0, "total": 0}


def _accumulate(acc: dict, items: list[dict]) -> None:
    for it in items:
        if it.get("type") == "assistant":
            for k in ("prompt", "completion", "cache_read", "total"):
                v = it["tokens"].get(k)
                if v:
                    acc[k] += int(v)


def _empty_totals():
    return {
        "lead": _zero(),
        "analyst": _zero(),
        "total": _zero(),
        "lead_turns": 0,
        "analyst_turns": 0,
        "tool_calls": 0,
    }


def _tally(lead_items, analyst_tokens: dict, analyst_turns: int) -> dict:
    tot = _empty_totals()
    _accumulate(tot["lead"], lead_items)
    tot["analyst"] = analyst_tokens
    tot["total"] = {k: tot["lead"][k] + analyst_tokens.get(k, 0) for k in tot["lead"]}
    tot["tool_calls"] = sum(1 for it in lead_items if it.get("type") == "tool")
    tot["lead_turns"] = sum(1 for it in lead_items if it.get("type") == "assistant")
    tot["analyst_turns"] = analyst_turns
    return tot


# ---------------------------------------------------------------- on-disk variants


def _read_tsv(path: Path) -> list[dict]:
    import csv

    with path.open() as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def _k_from_params(rows: list[dict]) -> int:
    n = 0
    for r in rows:
        est = str(r.get("estimate", "")).strip()
        if est in ("1", "1.0", "true", "True"):
            n += 1
    return n


def _charged_k(bic: float | None, nll, n: int | None) -> int | None:
    """The ``k`` a stored BIC charged, recovered from ``bic = 2·nll + k·ln(n)``.

    A version's ``meta.json`` records the server's BIC but not the ``k`` behind it, and the
    estimated rows of ``parameters.tsv`` are only part of that ``k``: the model's hardcoded
    constants are charged too. ``None`` when an input is missing, or when ``n`` is not the count
    the BIC was formed with (the quotient is not a whole number)."""
    if bic is None or nll is None or not n or n < 2:
        return None
    k = (bic - 2.0 * float(nll)) / math.log(n)
    return round(k) if math.isfinite(k) and abs(k - round(k)) < 1e-6 * max(1.0, abs(k)) else None


#: How close to the best fval a start must end to count as having reached it: the margin of the
#: server's own ``converged~N`` in its fit log and ``starts_at_best`` on its fit result
#: (``petab_mcp.fitting.AT_BEST_NLL_TOL``, frohlich-lab/PEtab-MCP#215), so the counts agree.
#: Retyped rather than imported: this module stays light enough to import without the server.
_AT_BEST_TOL = 1.0


def _multistart(opt: dict | None) -> dict | None:
    """What the card shows of a version's multistart (its ``optimization.json``): how many starts
    ran, how many reached the best fval, and how they ended. A best that one start of fifty found
    is not the same evidence as one that thirty did (#62). ``None`` for a version with no fit."""
    if not opt:
        return None
    starts = [s for s in opt.get("starts") or [] if isinstance(s, dict)]
    fvals = [f for f in (_num(s.get("fval")) for s in starts) if f is not None and math.isfinite(f)]
    best = min(fvals, default=None)
    return {
        "n_starts": opt.get("n_starts") or len(starts),
        "n_at_best": None if best is None else sum(f <= best + _AT_BEST_TOL for f in fvals),
        "exitflag_counts": opt.get("exitflag_counts"),
    }


def load_variants(run_dir: Path, n_data_points: int | None) -> list[dict]:
    """Build the ordered model-variant trajectory from ``<run_dir>/.petab_mcp``."""
    store = run_dir / ".petab_mcp"
    vdir = store / "versions"
    if not vdir.is_dir():
        return []
    # justification per version from provenance.jsonl
    justif: dict[int, str] = {}
    prov = store / "provenance.jsonl"
    if prov.is_file():
        for line in prov.read_text().splitlines():
            try:
                ev = json.loads(line)
            except Exception:
                continue
            if "version" in ev and ev.get("justification"):
                justif[int(ev["version"])] = ev["justification"]

    variants = []
    for d in sorted(vdir.glob("v*"), key=lambda p: _vnum(p.name)):
        meta = _safe_json(d / "meta.json") or {}
        params = _read_tsv(d / "parameters.tsv") if (d / "parameters.tsv").is_file() else []
        sim = _safe_json(d / "simulations.json") or {}
        n = sim.get("n_data_points") or n_data_points
        nll = meta.get("neg_log_likelihood")
        # The BIC the server stored, as the span path (`load_variants_from_spans`) reads it.
        # Recomputing it from the estimated rows dropped the hardcoded constants, so a model with
        # a literal scored better here than in its own fit result (#10).
        bic = bic_of(meta)
        k = _charged_k(bic, nll, n)
        if k is None:
            k = _k_from_params(params) if params else None
        variants.append(
            {
                "index": len(variants),
                "version": meta.get("version", _vnum(d.name)),
                "label": meta.get("label"),
                "kind": meta.get("kind"),
                "nll": nll,
                "chi2": meta.get("chi2"),
                "bic": bic,
                "k": k,
                "n": n,
                "is_new_best": meta.get("is_new_best"),
                "timestamp": meta.get("timestamp"),
                "antimony": _safe_read(d / "model.ant"),
                "params": params,
                # Both read through `read_version_json`, which takes the archive's gzipped
                # optimization.json under its plain name: an archived run shows the same card
                # as the live one did.
                "starts": _multistart(read_version_json(d, "optimization.json")),
                "statements": read_version_json(d, "statements.json"),
                "justification": justif.get(
                    int(meta.get("version", _vnum(d.name)))
                    if str(meta.get("version", "")).strip() != ""
                    else _vnum(d.name)
                ),
            }
        )
    return variants


def _tool_result(output_value) -> dict | None:
    """Unwrap a TOOL span's output (``{type,data:{content:[{text:<json>}]}}``) to the
    result dict, or None."""
    try:
        outer = output_value if isinstance(output_value, dict) else json.loads(output_value)
        txt = outer["data"]["content"][0]["text"]
        return json.loads(txt) if isinstance(txt, str) else txt
    except Exception:
        return None


# Tool spans whose result can carry a committed version. `make_changes` always fits, so a variant
# is one span. The other three are the pre-consolidation surface (a fit-free `set_model` /
# `change_parameters` plus a standalone `fit`); they stay listed because the explorer's job is
# reading traces that already exist, and a sweep recorded before the collapse would otherwise
# rebuild an empty trajectory and flag every variant `no_turn`.
_FITTING_TOOLS = ("make_changes", "set_model", "change_parameters", "fit")

# Spans whose INPUT carries the model text, so a later fit-only span inherits it.
_AUTHORING_TOOLS = ("make_changes", "set_model")


def _fit_block(res: dict) -> dict:
    """The part of a tool result carrying the fit scalars.

    ``make_changes`` carries them at the top level (as the standalone ``fit`` tool did before it);
    the intermediate `fit=true` bundle nested them under ``fit``, which some recorded traces still
    have."""
    nested = res.get("fit")
    return nested if isinstance(nested, dict) else res


def load_variants_from_spans(df, n_default: int | None) -> list[dict]:
    """Reconstruct the model-variant trajectory from the trace's fitting tool spans — the
    fallback when the on-disk version store has been wiped. A variant is any span in
    :data:`_FITTING_TOOLS` whose result reports ``committed``; its Antimony is taken from that
    call's own input when it re-authored, else from the most recent preceding one that did (which
    is what makes a pre-consolidation standalone ``fit`` span resolve to a model)."""
    tool_c = "attributes.tool.name"
    if tool_c not in df.columns:
        return []
    spans = df[df[tool_c].isin(_FITTING_TOOLS)].sort_values("start_time")
    variants: list[dict] = []
    last_ant: str | None = None
    last_justif: str | None = None
    for _, r in spans.iterrows():
        if r.get(tool_c) in _AUTHORING_TOOLS:
            try:
                args = json.loads(r.get("attributes.input.value")) or {}
            except Exception:
                args = {}
            ant = args.get("antimony")
            # The justification is read from the call's INPUT, not its result: the result no
            # longer echoes it back (a `make_changes` result is the fit and nothing else).
            last_justif = args.get("justification") or last_justif
            # Only a call that actually re-authored replaces the running model. `antimony` is
            # optional on `make_changes`, so a bounds-only call carries none — assigning it
            # unconditionally would blank the model out for that variant and every later one.
            if ant:
                last_ant = ant
        res = _tool_result(r.get("attributes.output.value"))
        # No version unless something actually fitted: an unbundled edit, and a rejected or
        # skipped fit, all report committed falsy.
        if not res or not res.get("committed"):
            continue
        fit = _fit_block(res)
        improvement = fit.get("improvement") or {}
        variants.append(
            {
                "index": len(variants),
                "version": res.get("version"),
                "label": "fit",
                "kind": "fit",
                "nll": fit.get("neg_log_likelihood"),
                "chi2": fit.get("chi2"),
                "bic": fit.get("bic"),
                # `k` (n_estimated + hardcoded constants) where the payload carries it; older
                # traces only have n_estimated, which was k then.
                "k": fit.get("k") if fit.get("k") is not None else fit.get("n_estimated"),
                "n": fit.get("n_data_points") or n_default,
                "is_new_best": improvement.get("is_new_best"),
                # The FIT's timestamp — a variant IS a fit snapshot, and that is what the on-disk
                # version store is keyed on. In the nested (pre-flattening) trace shape
                # `res.event_timestamp` is the earlier EDIT event, so preferring it would skew
                # time-based ordering against the version store.
                "timestamp": fit.get("event_timestamp") or res.get("event_timestamp"),
                "antimony": last_ant,
                "params": [],
                "justification": res.get("justification") or last_justif,
            }
        )
    return variants


def _vnum(name: str) -> int:
    try:
        return int(name.lstrip("v"))
    except ValueError:
        return 0


def _safe_json(p: Path):
    try:
        return json.loads(p.read_text()) if p.is_file() else None
    except Exception:
        return None


def _safe_read(p: Path) -> str | None:
    try:
        return p.read_text() if p.is_file() else None
    except Exception:
        return None


# ---------------------------------------------------------------- assembly + variant<->turn


def _vnum_of(x) -> int | None:
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def _map_variants_to_turns(turns: list[dict], variants: list[dict]) -> None:
    """Tag each turn with the model variant it produced, matched by the **committed version
    number** in that turn's fitting result — any span in :data:`_FITTING_TOOLS` — not by position.

    Positional pairing (j-th fit turn ↔ j-th fitted variant) silently mis-labels every
    variant whenever the counts diverge, which they routinely do: *rejected* / no-op ``fit`` calls
    carry no new version, and a committed variant's span can be dropped from the trace. Matching on
    the version number is immune to both. Turns whose fit was rejected get no variant; a fitted
    variant that no turn produced (dropped span) is flagged ``no_turn`` so the chart can show the
    gap instead of shifting later labels onto it. Also forward-fills ``active_variant`` (the variant
    in effect while viewing each turn)."""
    by_version = {vn: v for v in variants if (vn := _vnum_of(v.get("version"))) is not None}

    produced_by_turn: dict[int, dict] = {}  # turn index -> the variant that turn committed
    matched: set[int] = set()  # variant indices reached by some turn
    for t in turns:
        last_v = None
        for tc in t.get("tools", []):
            if tc.get("name") not in _FITTING_TOOLS:
                continue
            res = _tool_result(tc.get("result"))
            if not (isinstance(res, dict) and res.get("committed")):
                continue  # unbundled edit, or a rejected / no-op fit — no new version
            v = by_version.get(_vnum_of(res.get("version")))
            if v is not None:
                last_v = v
                matched.add(v["index"])
        if last_v is not None:
            produced_by_turn[t["i"]] = last_v
            t["variant_index"] = last_v["index"]

    # A fitted variant no turn produced (e.g. its make_changes span was dropped from the trace):
    # flag it so the chart marks the gap rather than pretending it never happened.
    for v in variants:
        if v.get("nll") is not None:
            v["no_turn"] = v["index"] not in matched

    # forward-fill the active variant while viewing each turn, starting at the initial v0.
    v0 = by_version.get(0)
    active = v0["index"] if v0 else None
    for t in turns:
        if t["i"] in produced_by_turn:
            active = produced_by_turn[t["i"]]["index"]
        t["active_variant"] = active


def _jsonsafe(o):
    """Recursively coerce a value to something ``json.dumps`` (strict) accepts: NaN/inf floats
    and pandas NaT -> None, numpy scalars -> python, datetimes/Timestamps -> ISO strings. Phoenix
    dataframe cells and arbitrary tool payloads carry all of these."""
    if o is None or isinstance(o, (bool, int, str)):
        return o
    if isinstance(o, float):
        return None if (math.isnan(o) or math.isinf(o)) else o
    if isinstance(o, dict):
        return {str(k): _jsonsafe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [_jsonsafe(x) for x in o]
    if hasattr(o, "isoformat"):  # datetime / pandas Timestamp
        try:
            return o.isoformat()
        except Exception:
            return str(o)
    item = getattr(o, "item", None)  # numpy scalar
    if callable(item):
        try:
            return _jsonsafe(item())
        except Exception:
            pass
    try:
        f = float(o)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return str(o)


def build_bundle(
    run_id: str, *, run_dir: Path, meta: dict, phoenix_base_url: str, project: str | None
) -> dict:
    """Assemble the full view model for one run (JSON-safe)."""
    project = project or meta.get("phoenix_project") or "default"
    trace_id = meta.get("phoenix_trace_id") or run_id
    variants = load_variants(run_dir, meta.get("n"))
    turns, totals, phoenix = (
        [],
        _empty_totals(),
        {"ok": False, "trace_id": trace_id, "note": "no phoenix_trace_id"},
    )
    if meta.get("phoenix_trace_id"):
        df = None
        try:
            df = _fetch_trace_df(trace_id, phoenix_base_url, project)
        except Exception as exc:
            phoenix = {
                "ok": False,
                "trace_id": trace_id,
                "note": f"could not read Phoenix trace: {type(exc).__name__}: {exc}",
            }
        if df is not None:
            # `result.json`, written at run end, carries `run_ended_at`; the live marker, written
            # at its start, does not.
            turns, totals, phoenix = load_turns(
                trace_id, phoenix_base_url, project, df=df, finished="run_ended_at" in meta
            )
            # If the on-disk store was wiped, rebuild the variant trajectory from the trace's
            # fitting spans so the score panel still works.
            if not variants:
                variants = load_variants_from_spans(df, meta.get("n"))
                for v in variants:
                    v["from_trace"] = True
    _map_variants_to_turns(turns, variants)
    return _jsonsafe(
        {
            "run_id": run_id,
            "meta": meta,
            "variants": variants,
            "turns": turns,
            "totals": totals,
            "phoenix": phoenix,
        }
    )
