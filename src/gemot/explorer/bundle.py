"""Assemble one run's view model from the on-disk version store + the Phoenix trace.

``build_bundle`` returns a JSON-able dict:

    {
      "run_id", "meta": {...result.json...},
      "variants": [ {index, version, nll, bic, chi2, is_new_best, timestamp,
                     antimony, params:[...], k, justification} ],
      "turns":    [ {i, kind:"lead", items:[...], tokens:{...}, model, variant_index} ],
      "totals":   {prompt, completion, cache_read, total, lead_turns, analyst_turns, tool_calls},
      "phoenix":  {ok, trace_id, note}
    }

The conversation (turns/tokens/analyst sub-agents) comes from Phoenix — queried by the indexed
``trace_id`` (fast; unfiltered pulls time out). The model-variant + score trajectory comes from the
run's ``.petab_mcp/`` version store. BIC-per-variant is recomputed (``2·nll + k·ln(n)``) since only
the final BIC is persisted. The two are stitched by aligning each fitting tool span (see
``_FITTING_TOOLS``) with the version it produced.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

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
            calls.append({
                "id": tc.get("tool_call.id"),
                "name": tc.get("tool_call.function.name"),
                "args": args,
            })
    return {"text": text, "tool_calls": calls}


def _system_prompt(msg_list) -> str:
    """The first system message's text from an LLM span's ``llm.input_messages`` (the persona:
    the lead and each analyst subagent run under a distinct, stable system prompt)."""
    for m in (msg_list if msg_list is not None else []):
        if isinstance(m, dict) and m.get("message.role") == "system":
            return _msg_text(m).strip()
    return ""


def _emits_task(out_msgs) -> bool:
    """Whether an LLM span's output contains a ``task`` tool call — only the lead delegates,
    so this identifies lead spans even when the trace tree is incomplete (a live run)."""
    return any(tc.get("name") == "task" for tc in _assistant_from_output(out_msgs)["tool_calls"])


def _match_pending_subagent(system_prompt: str, emitted: list[dict]) -> tuple[str | None, str | None]:
    """Best-effort (subagent_type, description) for an in-progress analyst, by matching the
    lead's emitted ``task`` calls against the analyst's persona prompt (e.g. subagent_type
    ``mathematician`` ↔ a prompt beginning "You are a mathematician…").

    Only the persona intro (the "You are a …" clause) is matched, not the whole prompt: the
    shared report contract mentions every analyst, so a full-text search would mislabel."""
    intro = (system_prompt or "").lower()[:80]
    for a in emitted:
        st = a.get("subagent_type")
        toks = str(st or "").lower().replace("_", " ").replace("-", " ").split()
        if toks and toks[0] in intro:
            return st, a.get("description")
    return None, None


def _num(v) -> float | None:
    try:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


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
            self.by_id[span_id] = {
                "id": span_id,
                "parent": (None if _isnull(r.get("parent_id")) else r.get("parent_id")),
                "name": r.get("name"),
                "kind": r.get(kind_c),
                "start": r.get("start_time"),
                "in": r.get("attributes.input.value"),
                "out": r.get("attributes.output.value"),
                "tool_name": r.get("attributes.tool.name"),
                "in_msgs": r.get("attributes.llm.input_messages"),
                "out_msgs": r.get("attributes.llm.output_messages"),
                "model": r.get("attributes.llm.model_name"),
                "tok_prompt": _num(r.get("attributes.llm.token_count.prompt")),
                "tok_completion": _num(r.get("attributes.llm.token_count.completion")),
                "tok_total": _num(r.get("attributes.llm.token_count.total")),
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
                items.append({"type": "assistant", "text": a["text"],
                              "tool_calls": a["tool_calls"], "model": s["model"],
                              "tokens": {"prompt": s["tok_prompt"], "completion": s["tok_completion"],
                                         "cache_read": s["tok_cache"], "total": s["tok_total"]},
                              "_id": s["id"], "start": s["start"]})
            elif s["kind"] == "TOOL":
                items.append({"type": "tool", "name": s["tool_name"] or s["name"],
                              "args": _jload(s["in"]), "result": _jload(s["out"]),
                              "_id": s["id"], "start": s["start"]})
        return items


def _isnull(v) -> bool:
    try:
        return v is None or (isinstance(v, float) and math.isnan(v))
    except Exception:
        return v is None


def _fetch_trace_df(trace_id: str, base_url: str, project: str):
    from phoenix.client import Client
    from phoenix.client.types.spans import SpanQuery
    client = Client(base_url=base_url)
    q = SpanQuery().where(f"trace_id == '{trace_id}'")
    return client.spans.get_spans_dataframe(query=q, project_name=project, timeout=120)


def load_turns(trace_id: str, base_url: str, project: str, df=None) -> tuple[list[dict], dict, dict]:
    """Return (turns, totals, phoenix_status) for a trace. Degrades to empty on any failure.

    ``df`` may be a pre-fetched spans dataframe (so a caller can reuse it for variant
    reconstruction); when None it is fetched here.
    """
    status = {"ok": False, "trace_id": trace_id, "note": ""}
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
    # only agent that emits `task` tool calls, so its system prompt is a reliable signature:
    # any LLM span NOT under a completed task whose system prompt isn't the lead's is an
    # in-progress analyst. (Empty lead-prompt set ⇒ no delegation has started ⇒ no orphans.)
    lead_prompts = {_system_prompt(s["in_msgs"]) for sid, s in tr.by_id.items()
                    if s["kind"] == "LLM" and _emits_task(s["out_msgs"])}
    orphan_analyst_ids = []
    if lead_prompts:
        for sid, s in tr.by_id.items():
            if s["kind"] == "LLM" and sid not in all_analyst_ids:
                sp = _system_prompt(s["in_msgs"])
                if sp and sp not in lead_prompts:
                    orphan_analyst_ids.append(sid)

    # Lead timeline = every span EXCEPT those inside a completed task subtree, the in-progress
    # analyst spans just identified (task spans themselves stay, to render the delegation),
    # minus plumbing.
    exclude = all_analyst_ids | set(orphan_analyst_ids) | {
        sid for sid, s in tr.by_id.items()
        if s["kind"] == "CHAIN" or (s["name"] in _PLUMBING) or
        (s["kind"] == "AGENT")  # middleware "*.before_agent" etc.
    }
    lead_items = tr.timeline(list(tr.by_id.keys()), exclude)
    inner_ex = {s for s, sp in tr.by_id.items()
                if sp["kind"] in ("CHAIN", "AGENT") or sp["name"] in _PLUMBING}

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
                    "subagent_type": (args.get("subagent_type") if isinstance(args, dict) else None),
                    "description": (args.get("description") if isinstance(args, dict) else None),
                    "items": sub_items,
                }
                analyst_turn_count += sum(1 for x in sub_items if x["type"] == "assistant")
                _accumulate(analyst_tokens, sub_items)

    # Render each in-progress analyst as its own delegation block (grouped by persona, since
    # the panel spawns the three analysts concurrently) attached to the lead's spawning turn,
    # rather than letting its turns show up as lead turns.
    if orphan_analyst_ids:
        emitted = [tc.get("args") for it in lead_items if it["type"] == "assistant"
                   for tc in (it.get("tool_calls") or [])
                   if tc.get("name") == "task" and isinstance(tc.get("args"), dict)]
        by_prompt: dict[str, list[str]] = {}
        for sid in orphan_analyst_ids:
            by_prompt.setdefault(_system_prompt(tr.by_id[sid]["in_msgs"]), []).append(sid)
        for sp, ids in by_prompt.items():
            sub_items = tr.timeline(ids, inner_ex)
            st, desc = _match_pending_subagent(sp, emitted)
            lead_items.append({
                "type": "tool", "name": "task", "_id": None, "result": None,
                "args": {"subagent_type": st, "description": desc},
                "start": min((tr.by_id[i]["start"] for i in ids), default=None),
                "delegation": {"subagent_type": st, "description": desc,
                               "items": sub_items, "in_progress": True},
            })
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
            cur = {"kind": "lead", "text": it["text"], "model": it["model"],
                   "tokens": it["tokens"], "tools": [], "start": it["start"]}
        else:  # tool
            if cur is None:  # a tool before any assistant (rare) -> standalone turn
                cur = {"kind": "lead", "text": "", "model": None,
                       "tokens": _empty_tokens(), "tools": [], "start": it["start"]}
            cur["tools"].append({k: it[k] for k in ("name", "args", "result", "delegation", "start")
                                 if k in it})
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
    return {"lead": _zero(), "analyst": _zero(), "total": _zero(),
            "lead_turns": 0, "analyst_turns": 0, "tool_calls": 0}


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
        k = _k_from_params(params) if params else (
            len(meta.get("fitted_parameters") or {}) or None)
        # ONE BIC definition, in `petab_mcp.criteria`. It returns None for any missing or
        # impossible input, so no separate guard is needed here.
        from petab_mcp.criteria import CANONICAL

        bic = CANONICAL["bic"].value(nll, k, n)
        variants.append({
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
            "justification": justif.get(int(meta.get("version", _vnum(d.name))) if
                                        str(meta.get("version", "")).strip() != "" else _vnum(d.name)),
        })
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
        variants.append({
            "index": len(variants),
            "version": res.get("version"),
            "label": "fit", "kind": "fit",
            "nll": fit.get("neg_log_likelihood"),
            "chi2": fit.get("chi2"),
            "bic": fit.get("bic"),
            "k": fit.get("n_estimated"),
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
        })
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

    produced_by_turn: dict[int, dict] = {}   # turn index -> the variant that turn committed
    matched: set[int] = set()                # variant indices reached by some turn
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


def build_bundle(run_id: str, *, run_dir: Path, meta: dict,
                 phoenix_base_url: str, project: str | None) -> dict:
    """Assemble the full view model for one run (JSON-safe)."""
    project = project or meta.get("phoenix_project") or "default"
    trace_id = meta.get("phoenix_trace_id") or run_id
    variants = load_variants(run_dir, meta.get("n"))
    turns, totals, phoenix = ([], _empty_totals(), {"ok": False, "trace_id": trace_id,
                                                     "note": "no phoenix_trace_id"})
    if meta.get("phoenix_trace_id"):
        df = None
        try:
            df = _fetch_trace_df(trace_id, phoenix_base_url, project)
        except Exception as exc:
            phoenix = {"ok": False, "trace_id": trace_id,
                       "note": f"could not read Phoenix trace: {type(exc).__name__}: {exc}"}
        if df is not None:
            turns, totals, phoenix = load_turns(trace_id, phoenix_base_url, project, df=df)
            # If the on-disk store was wiped, rebuild the variant trajectory from the trace's
            # fitting spans so the score panel still works.
            if not variants:
                variants = load_variants_from_spans(df, meta.get("n"))
                for v in variants:
                    v["from_trace"] = True
    _map_variants_to_turns(turns, variants)
    return _jsonsafe({
        "run_id": run_id,
        "meta": meta,
        "variants": variants,
        "turns": turns,
        "totals": totals,
        "phoenix": phoenix,
    })
