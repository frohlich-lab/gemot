"""The Conversation tab: a run's agents, turn by turn, as its Phoenix trace recorded them.

petab-mcp's viewer (``petab_mcp.viz``) shows what the server stored, version by version; this is
what only the harness has, the conversation that produced the versions (:func:`.bundle.build_bundle`).
It follows the viewer's version: by default it shows the turns that produced the version picked,
from the turn after the previous version's to the one that committed it, and a switch shows the
whole run, a page at a time. Each turn is the lead's text, its tokens, and its tool calls collapsed,
an analyst's delegation nested inside the call that made it.

A live run's trace is fetched again at most every :data:`LIVE_TTL` seconds; Phoenix's trace query
is the heaviest thing the explorer does, so fetches never overlap (:data:`_FETCH`).
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import panel as pn

from . import bundle as B

#: Seconds a live run's conversation is reused before Phoenix is asked again.
LIVE_TTL = 15.0

#: Turns a page of the whole run shows.
PAGE = 12

#: Characters of a tool call's arguments or result shown before it is cut.
CLIP = 4000

_FETCH = threading.Lock()


def _code(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, indent=1, default=str)
    if len(text) > CLIP:
        text = text[:CLIP] + f"\n... ({len(text) - CLIP} more characters)"
    return f"```\n{text}\n```"


def _tokens(tokens: dict | None) -> str:
    total = (tokens or {}).get("total")
    return f"{int(total):,} tokens" if isinstance(total, (int, float)) and total else ""


def item_view(item: dict) -> pn.viewable.Viewable:
    """An analyst's step: its text, or a tool call it made."""
    if item.get("type") == "assistant":
        return pn.pane.Markdown(item.get("text") or "_(no text)_", sizing_mode="stretch_width")
    return tool_view(item)


def tool_view(call: dict) -> pn.Card:
    """A tool call, collapsed: its arguments and result, or the delegation it made."""
    delegation = call.get("delegation")
    if delegation:
        title = f"analyst {delegation.get('subagent_type') or '?'}"
        if delegation.get("in_progress"):
            title += " (running)"
        body = [pn.pane.Markdown(delegation.get("description") or "", sizing_mode="stretch_width")]
        body += [item_view(i) for i in delegation.get("items") or []]
        return pn.Card(*body, title=title, collapsed=True, sizing_mode="stretch_width")
    title = call.get("name") or "tool"
    if call.get("error"):
        title += " · error"
    body = [
        pn.pane.Markdown("**arguments**\n" + _code(call.get("args")), sizing_mode="stretch_width")
    ]
    if call.get("result") is not None or call.get("error"):
        body.append(
            pn.pane.Markdown(
                "**result**\n" + _code(call.get("error") or call.get("result")),
                sizing_mode="stretch_width",
            )
        )
    return pn.Card(*body, title=title, collapsed=True, sizing_mode="stretch_width")


def turn_view(turn: dict, committed: int | None) -> pn.Card:
    title = " · ".join(
        filter(
            None,
            [
                f"turn {turn.get('i')}",
                turn.get("model"),
                _tokens(turn.get("tokens")),
                f"committed v{committed}" if committed is not None else None,
                "error" if turn.get("error") else None,
            ],
        )
    )
    # A turn that only calls tools has no text to show: most of a real run's turns.
    text = turn.get("text") or ("" if turn.get("tools") else "_(no text)_")
    body = [pn.pane.Markdown(text, sizing_mode="stretch_width")] if text else []
    body += [tool_view(t) for t in turn.get("tools") or []]
    return pn.Card(*body, title=title, sizing_mode="stretch_width")


def committed_versions(data: dict) -> dict[int, int]:
    """Each turn that committed a version, by its index: that version's number."""
    variants = {v.get("index"): B._vnum_of(v.get("version")) for v in data.get("variants") or []}
    return {
        t["i"]: variants[t["variant_index"]]
        for t in data.get("turns") or []
        if t.get("variant_index") in variants and variants[t["variant_index"]] is not None
    }


def turns_for(data: dict, version: int | None) -> list[dict]:
    """The turns that produced ``version``: from the turn after the one that committed the
    version before it, to the one that committed it. All turns where no turn committed it."""
    turns = data.get("turns") or []
    committed = committed_versions(data)
    by_version = {v: i for i, v in committed.items()}
    if version not in by_version:
        return turns
    end = by_version[version]
    earlier = [i for i, v in committed.items() if i < end]
    start = max(earlier) + 1 if earlier else 0
    return [t for t in turns if start <= t["i"] <= end]


def header(data: dict) -> str:
    meta, totals, phoenix = (
        data.get("meta") or {},
        data.get("totals") or {},
        data.get("phoenix") or {},
    )
    run = " · ".join(
        str(x)
        for x in (
            meta.get("scenario"),
            meta.get("model"),
            meta.get("driver"),
            meta.get("stop_reason"),
            {True: "succeeded", False: "failed"}.get(meta.get("success")),
        )
        if x
    )
    # Token totals are per role, each {prompt, completion, cache_read, total} (bundle._tally).
    counts = {**totals, "tokens": (totals.get("total") or {}).get("total")}
    used = " · ".join(
        f"{label} {int(counts[key]):,}"
        for key, label in (
            ("lead_turns", "lead turns"),
            ("analyst_turns", "analyst turns"),
            ("tool_calls", "tool calls"),
            ("tokens", "tokens"),
        )
        if isinstance(counts.get(key), (int, float))
    )
    lines = [f"**{run}**" if run else "", used]
    if not phoenix.get("ok"):
        lines.append(f"No conversation: {phoenix.get('note') or 'the Phoenix trace was not read'}.")
    return "  \n".join(line for line in lines if line)


class Conversation:
    """One run's conversation, fetched once (an archived run) or kept fresh (a live one)."""

    def __init__(
        self, run_id: str, run_dir: Path, meta: dict, phoenix_base_url: str, live: bool
    ) -> None:
        self.run_id, self.run_dir, self.meta = run_id, Path(run_dir), meta
        self.phoenix_base_url, self.live = phoenix_base_url, live
        self._data: dict | None = None
        self._fetched = 0.0
        self.whole = pn.widgets.Checkbox(name="the whole run", value=False)
        self.page = pn.widgets.IntSlider(name="page", start=1, end=1, value=1, width=300)

    def data(self) -> dict:
        stale = self.live and time.monotonic() - self._fetched > LIVE_TTL
        if self._data is None or stale:
            with _FETCH:
                self._data = B.build_bundle(
                    self.run_id,
                    run_dir=self.run_dir,
                    meta=self.meta,
                    phoenix_base_url=self.phoenix_base_url,
                    project=self.meta.get("phoenix_project"),
                )
            self._fetched = time.monotonic()
        return self._data

    def view(self, viz) -> pn.viewable.Viewable:
        """The tab, for ``petab_mcp.viz``'s ``extra_tabs``: it follows ``viz``'s version, and its
        ``tick`` for a live run."""
        return pn.Column(
            pn.bind(self._render, viz.param.version, viz.param.tick, self.whole, self.page),
            sizing_mode="stretch_width",
        )

    def _render(self, version, _tick, whole, page):
        data = self.data()
        if not data.get("turns"):  # no trace, or none read yet: nothing to page or pick
            return pn.pane.Markdown(header(data), sizing_mode="stretch_width")
        committed = committed_versions(data)
        if whole:
            turns = data.get("turns") or []
            pages = max(1, -(-len(turns) // PAGE))
            self.page.end = pages
            shown = turns[(min(page, pages) - 1) * PAGE : min(page, pages) * PAGE]
            note = f"All {len(turns)} turns, {PAGE} a page."
            controls = pn.Row(self.whole, self.page)
        else:
            shown = turns_for(data, version)
            note = (
                f"The {len(shown)} turns that produced v{version}."
                if version in committed.values()
                else "No turn committed this version (the run's start, or its span is not in the "
                "trace): every turn is shown."
            )
            controls = pn.Row(self.whole)
        return pn.Column(
            pn.pane.Markdown(header(data), sizing_mode="stretch_width"),
            controls,
            pn.pane.Markdown(note, styles={"color": "#8c8c8c"}),
            *[turn_view(t, committed.get(t["i"])) for t in shown],
            sizing_mode="stretch_width",
        )
