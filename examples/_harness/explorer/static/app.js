"use strict";
// Run explorer viewer. Fetches the run bundle, renders the lead turn stream (analyst
// delegations collapsed by default), and keeps a sticky score chart + model-variant card that
// track the turn scrolled into view.

const $ = (sel, root = document) => root.querySelector(sel);
const fmt = (x, d = 2) => (x === null || x === undefined || Number.isNaN(x)) ? "—" : (+x).toFixed(d);
const int = (x) => (x === null || x === undefined) ? "—" : Math.round(x).toLocaleString();

function el(tag, attrs = {}, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === false || v === null || v === undefined) continue;
    if (k === "class") n.className = v;
    else if (k === "html") n.innerHTML = v;
    else if (k === "text") n.textContent = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else if (v === true) n.setAttribute(k, "");
    else n.setAttribute(k, v);
  }
  for (const kid of kids.flat()) if (kid != null) n.append(kid.nodeType ? kid : document.createTextNode(kid));
  return n;
}
const esc = (s) => String(s ?? "").replace(/[&<>]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
const pretty = (v) => typeof v === "string" ? v : JSON.stringify(v, null, 2);

// ---------------------------------------------------------------- Antimony highlighting
const ANT_KEYWORDS = new Set(("model end function const var species compartment formula substanceOnly " +
  "unit import is has at after delete ext in encodes convert reaction default").split(" "));
function highlightAntimony(src) {
  if (!src) return "";
  let out = "", i = 0;
  const s = src;
  const push = (cls, txt) => out += `<span class="${cls}">${esc(txt)}</span>`;
  while (i < s.length) {
    const c = s[i];
    if (c === "/" && s[i + 1] === "/") { let j = s.indexOf("\n", i); if (j < 0) j = s.length; push("ant-com", s.slice(i, j)); i = j; continue; }
    if (c === "/" && s[i + 1] === "*") { let j = s.indexOf("*/", i); j = j < 0 ? s.length : j + 2; push("ant-com", s.slice(i, j)); i = j; continue; }
    if (c === "#") { let j = s.indexOf("\n", i); if (j < 0) j = s.length; push("ant-com", s.slice(i, j)); i = j; continue; }
    if (c === '"') { let j = i + 1; while (j < s.length && s[j] !== '"') j++; push("ant-str", s.slice(i, j + 1)); i = j + 1; continue; }
    const arrow = s.startsWith("->", i) || s.startsWith("=>", i) || s.startsWith(":=", i);
    if (arrow) { push("ant-op", s.slice(i, i + 2)); i += 2; continue; }
    if (/[;=+\-*/^()<>,{}]/.test(c)) { push("ant-op", c); i++; continue; }
    let m = /^[0-9]+\.?[0-9]*(?:[eE][+-]?[0-9]+)?/.exec(s.slice(i));
    if (m && /[0-9]/.test(c)) { push("ant-num", m[0]); i += m[0].length; continue; }
    m = /^[A-Za-z_]\w*/.exec(s.slice(i));
    if (m) {
      const w = m[0];
      // a reaction/rule label:  name:
      let k = i + w.length; while (s[k] === " ") k++;
      if (s[k] === ":" && s[k + 1] !== "=") push("ant-rxn", w);
      else if (ANT_KEYWORDS.has(w)) push("ant-key", w);
      else out += esc(w);
      i += w.length; continue;
    }
    out += esc(c); i++;
  }
  return out;
}

// ---------------------------------------------------------------- chart
// tokens spent in one turn = the lead call + any analyst turns inside its delegations
function turnTokens(t) {
  let n = (t.tokens && t.tokens.total) || 0;
  for (const tc of (t.tools || [])) {
    const d = tc.delegation;
    if (d) for (const it of d.items) if (it.type === "assistant" && it.tokens) n += it.tokens.total || 0;
  }
  return n;
}
const NS = "http://www.w3.org/2000/svg";
const mk = (t, a) => { const e = document.createElementNS(NS, t); for (const k in a) e.setAttribute(k, a[k]); return e; };
const kfmt = (n) => n >= 1000 ? (n / 1000).toFixed(n >= 10000 ? 0 : 1) + "k" : String(Math.round(n));

// Dual-axis chart over the TURN timeline: left = score (ΔBIC/nll) as a step line marked at each
// set_model, right = cumulative token sum, plus a vertical cursor at the scrolled turn.
function renderChart(container, turns, variants, metric, bicRef, activeTurn, onPickTurn) {
  container.innerHTML = "";
  const byIdx = Object.fromEntries(variants.map(v => [v.index, v]));
  const useRef = metric === "bic";
  const scoreOf = (v) => {
    if (v == null) return null;
    if (metric === "nll") return v.nll;
    return (v.bic != null && bicRef != null) ? v.bic - bicRef : v.bic;  // ΔBIC
  };
  const N = turns.length;
  if (N === 0) {  // no conversation (e.g. scripted run) — plot score vs variant index
    const fv = variants.filter(v => scoreOf(v) != null);
    if (!fv.length) { container.append(el("div", { class: "muted", text: "no turns or fitted variants" })); return; }
    const W = 372, H = 150, PL = 40, PR = 12, PT = 12, PB = 22;
    const xs = (i) => PL + (fv.length <= 1 ? (W - PL - PR) / 2 : (i / (fv.length - 1)) * (W - PL - PR));
    const vals = fv.map(scoreOf); let lo = Math.min(...vals), hi = Math.max(...vals);
    if (lo === hi) { lo -= 1; hi += 1; } const pad = (hi - lo) * 0.12; lo -= pad; hi += pad;
    const ly = (v) => PT + (1 - (v - lo) / (hi - lo)) * (H - PT - PB);
    const svg = mk("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" });
    svg.append(mk("line", { x1: PL, y1: H - PB, x2: W - PR, y2: H - PB, stroke: "var(--line)" }));
    if (fv.length > 1) svg.append(mk("polyline", { points: fv.map((v, k) => `${xs(k)},${ly(scoreOf(v))}`).join(" "), fill: "none", stroke: "var(--accent)", "stroke-width": "2" }));
    fv.forEach((v, k) => {
      svg.append(mk("circle", { cx: xs(k), cy: ly(scoreOf(v)), r: 4, fill: v.is_new_best ? "var(--best)" : "var(--accent)", stroke: "var(--bg)", "stroke-width": "1.5" }));
      const t = mk("text", { x: xs(k), y: H - PB + 12, "text-anchor": "middle", fill: "var(--muted)", "font-size": "9" }); t.textContent = "v" + v.version; svg.append(t);
    });
    container.append(svg); return;
  }
  const score = turns.map(t => scoreOf(byIdx[t.active_variant]));
  let run = 0; const toks = turns.map(t => (run += turnTokens(t)));
  const setmodelTurns = turns.filter(t => t.variant_index != null).map(t => t.i);
  // committed variants that no turn produced (a dropped set_model span) — shown as gap markers
  // rather than silently dropped. Prefer the server's `no_turn` flag; fall back to deriving it.
  const mappedIdx = new Set(turns.filter(t => t.variant_index != null).map(t => t.variant_index));
  const orphans = variants.filter(v => v.nll != null
    && (v.no_turn ?? !mappedIdx.has(v.index)) && scoreOf(v) != null);

  const W = 372, H = 158, PL = 40, PR = 42, PT = 12, PB = 24;
  const xs = (i) => PL + (N <= 1 ? (W - PL - PR) / 2 : (i / (N - 1)) * (W - PL - PR));
  const sVals = score.filter(v => v != null).concat(orphans.map(scoreOf));
  const svg = mk("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" });

  // right axis: cumulative tokens (0..max)
  const tMax = Math.max(1, toks[toks.length - 1] || 1);
  const ry = (v) => PT + (1 - v / tMax) * (H - PT - PB);
  // token area/line
  const tokPts = toks.map((v, i) => `${xs(i)},${ry(v)}`).join(" ");
  svg.append(mk("polyline", { points: tokPts, fill: "none", stroke: "var(--muted)",
    "stroke-width": "1.5", "stroke-dasharray": "3 3", opacity: "0.9" }));
  // right ticks
  for (const v of [tMax]) {
    const tx = mk("text", { x: W - PR + 4, y: ry(v) + 3, fill: "var(--muted)", "font-size": "9" });
    tx.textContent = kfmt(v); svg.append(tx);
  }
  const rlbl = mk("text", { x: W - PR + 4, y: H - PB + 12, fill: "var(--muted)", "font-size": "9" });
  rlbl.textContent = "tok"; svg.append(rlbl);

  // left axis: score
  if (sVals.length) {
    let lo = Math.min(...sVals), hi = Math.max(...sVals);
    if (lo === hi) { lo -= 1; hi += 1; }
    const pad = (hi - lo) * 0.12; lo -= pad; hi += pad;
    const ly = (v) => PT + (1 - (v - lo) / (hi - lo)) * (H - PT - PB);
    // step line across turns where score defined
    let d = "", started = false;
    score.forEach((v, i) => { if (v == null) return; d += `${started ? "L" : "M"}${xs(i)} ${ly(v)} `; started = true; });
    svg.append(mk("path", { d, fill: "none", stroke: "var(--accent)", "stroke-width": "2" }));
    // dots at set_model turns
    for (const ti of setmodelTurns) {
      const v = byIdx[turns[ti].variant_index];
      const sv = scoreOf(v); if (sv == null) continue;
      const cx = xs(ti), cy = ly(sv);
      const dot = mk("circle", { cx, cy, r: 4, style: "cursor:pointer",
        fill: v.is_new_best ? "var(--best)" : "var(--accent)", stroke: "var(--bg)", "stroke-width": "1.5" });
      dot.addEventListener("click", () => onPickTurn && onPickTurn(ti));
      const tl = mk("title", {});
      tl.textContent = `turn ${ti} · v${v.version}: ${useRef ? "ΔBIC" : "nll"}=${fmt(sv)}`; dot.append(tl);
      svg.append(dot);
    }
    // orphan commits (no turn): hollow diamonds at an x interpolated between the version's
    // neighbours, so a dropped span reads as "a commit happened here" rather than vanishing.
    if (orphans.length) {
      const mapped = turns.filter(t => t.variant_index != null)
        .map(t => ({ ver: byIdx[t.variant_index].version, ti: t.i }))
        .sort((a, b) => a.ver - b.ver);
      for (const v of orphans) {
        let prev = null, next = null;
        for (const m of mapped) { if (m.ver < v.version) prev = m; else if (m.ver > v.version && !next) next = m; }
        let ox = (PL + W - PR) / 2;
        if (prev && next) ox = (xs(prev.ti) + xs(next.ti)) / 2;
        else if (prev) ox = Math.min(W - PR, xs(prev.ti) + 14);
        else if (next) ox = Math.max(PL, xs(next.ti) - 14);
        const oy = ly(scoreOf(v));
        const dm = mk("path", { d: `M${ox} ${oy - 4.5} L${ox + 4.5} ${oy} L${ox} ${oy + 4.5} L${ox - 4.5} ${oy} Z`,
          fill: "var(--bg)", stroke: "var(--warn)", "stroke-width": "1.5", "stroke-dasharray": "2 1.4" });
        const tl = mk("title", {});
        tl.textContent = `v${v.version}: ${useRef ? "ΔBIC" : "nll"}=${fmt(scoreOf(v))} — commit has no conversation turn (span missing from trace)`;
        dm.append(tl); svg.append(dm);
      }
    }
    for (const v of [lo + pad, hi - pad]) {
      const tx = mk("text", { x: PL - 4, y: ly(v) + 3, "text-anchor": "end", fill: "var(--muted)", "font-size": "9" });
      tx.textContent = Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(1); svg.append(tx);
    }
  }
  const llbl = mk("text", { x: PL - 4, y: H - PB + 12, "text-anchor": "end", fill: "var(--accent)", "font-size": "9" });
  llbl.textContent = useRef ? "ΔBIC" : "nll"; svg.append(llbl);

  // baseline axis
  svg.append(mk("line", { x1: PL, y1: H - PB, x2: W - PR, y2: H - PB, stroke: "var(--line)" }));

  // vertical scroll cursor
  if (activeTurn != null) {
    const cx = xs(activeTurn);
    svg.append(mk("line", { x1: cx, y1: PT, x2: cx, y2: H - PB, stroke: "var(--ink)",
      "stroke-width": "1", opacity: "0.55" }));
    const cl = mk("text", { x: cx, y: PT - 2, "text-anchor": "middle", fill: "var(--ink)", "font-size": "9" });
    cl.textContent = "turn " + activeTurn; svg.append(cl);
  }
  container.append(svg);
}

// ---------------------------------------------------------------- variant card
function renderVariantCard(card, v, meta, metric) {
  card.innerHTML = "";
  if (!v) { card.append(el("div", { class: "vc-just muted", text: "no model variant yet" })); return; }
  const head = el("div", { class: "vc-head" },
    el("span", { class: "vn", text: `v${v.version}` }),
    el("span", { class: "muted", text: v.label || v.kind || "" }),
    v.is_new_best ? el("span", { class: "badge-best", text: "new best" }) : null);
  const dbic = (v.bic != null && meta.bic_reference != null) ? v.bic - meta.bic_reference : null;
  const metrics = el("div", { class: "vc-metrics" },
    el("span", { class: "m", html: `<span>nll</span> ${fmt(v.nll, 3)}` }),
    el("span", { class: "m", html: `<span>bic</span> ${fmt(v.bic, 2)}` }),
    el("span", { class: "m", html: `<span>ΔBIC</span> ${dbic == null ? "—" : (dbic >= 0 ? "+" : "") + fmt(dbic, 2)}` }),
    el("span", { class: "m", html: `<span>chi2</span> ${fmt(v.chi2, 1)}` }),
    el("span", { class: "m", html: `<span>k</span> ${v.k ?? "—"}` }));
  card.append(head, metrics);
  if (v.justification) card.append(el("div", { class: "vc-just" },
    el("span", { class: "lbl", text: "justification" }), v.justification));
  card.append(el("pre", { class: "vc-ant", html: highlightAntimony(v.antimony) }));
}

// ---------------------------------------------------------------- turn stream
function toolCallEl(tc) {
  if (tc.delegation) {
    const d = tc.delegation;
    const nsub = d.items.filter(x => x.type === "assistant").length;
    const det = el("details", { class: "delegation" + (d.in_progress ? " running" : ""), open: d.in_progress || false },
      el("summary", {},
        el("span", { class: "deleg-badge", text: (d.subagent_type || "analyst") }),
        el("span", { class: "deleg-desc", text: d.description || "" }),
        d.in_progress ? el("span", { class: "deleg-live", html: `<span class="live-dot"></span>running` }) : null,
        el("span", { class: "deleg-meta", text: `${nsub} turn${nsub === 1 ? "" : "s"}${d.in_progress ? " so far" : ""}` })));
    const box = el("div", { class: "subturns" });
    let cur = null;
    for (const it of d.items) {
      if (it.type === "assistant") {
        cur = el("div", { class: "subturn" },
          el("div", { class: "turn-role", text: d.subagent_type || "analyst" }),
          el("div", { class: "assistant-text" + (it.text ? "" : " empty"), text: it.text || "(tool calls only)" }));
        box.append(cur);
      } else {
        (cur || box).append(kvTool(it));
      }
    }
    det.append(box);
    return det;
  }
  return kvTool(tc);
}
function kvTool(tc) {
  const isSet = tc.name === "set_model";
  const det = el("details", { class: "toolcall" + (isSet ? " setmodel" : "") },
    el("summary", {}, el("span", { class: "tool-name", text: tc.name || "tool" })));
  const body = el("div", { class: "kv" });
  if (tc.args !== undefined && tc.args !== null && tc.args !== "")
    body.append(el("div", { class: "lbl", text: "args" }), el("pre", { text: pretty(tc.args) }));
  if (tc.result !== undefined && tc.result !== null && tc.result !== "")
    body.append(el("div", { class: "lbl", text: "result" }), el("pre", { text: pretty(tc.result) }));
  det.append(body);
  return det;
}

function turnEl(t) {
  const toks = t.tokens || {};
  const badge = el("span", { class: "turn-toks", html:
    `<b>${int(toks.total)}</b> tok${toks.cache_read ? ` · ${int(toks.cache_read)} cached` : ""}` });
  const head = el("div", { class: "turn-head" },
    el("span", { class: "turn-role", text: "lead" }),
    el("span", { class: "turn-idx", text: `turn ${t.i}` }),
    (t.active_variant != null ? el("span", { class: "vtag", text: `v${t.active_variant}` }) : null),
    badge);
  const body = el("div", { class: "turn-body" },
    el("div", { class: "assistant-text" + (t.text ? "" : " empty"), text: t.text || "(tool calls only)" }));
  for (const tc of (t.tools || [])) body.append(toolCallEl(tc));
  const node = el("div", { class: "turn", "data-i": t.i }, head, body);
  node._turn = t;
  return node;
}

// ---------------------------------------------------------------- main
// The viewer renders once for an archived run, and re-renders on a poll loop for a LIVE run
// (data.live === true): each poll re-reads the on-disk variant store + re-queries Phoenix, so
// new turns and model variants appear as the agent produces them. Only the growing tail of the
// stream is rebuilt, so earlier turns — and the reader's scroll position — stay put; when the
// run finishes, the server serves the archived bundle (live:false) and polling stops.
(function main() {
  const stream = $("#stream"), chart = $("#chart"), card = $("#variant-card"), legend = $("#chart-legend");
  const badges = $("#run-badges"), note = $("#phoenix-note");

  // mutable view state, refreshed by applyData()
  let nodes = [], turns = [], variants = [], variantByIndex = {}, meta = {}, bicRef = null;
  let metric = "bic", activeTurn = null, live = false, placeholder = null, pollTimer = null;
  const atBottom = () => (window.innerHeight + window.scrollY) >= (document.body.scrollHeight - 8);

  function clearPlaceholder() { if (placeholder) { placeholder.remove(); placeholder = null; } }
  function showPlaceholder() {
    clearPlaceholder();
    placeholder = el("div", { class: "muted", text: live ? "waiting for the first turn…" : "no conversation turns (see note above)." });
    stream.append(placeholder);
  }

  // header badges (rebuilt each render): live → a pulsing "live" pill and no wall/stop badges;
  // finished → the verdict pill. ΔBIC/k fall back to the best/last variant while live (the
  // run-level values only exist once result.json is written).
  function renderHeader(data) {
    const m = data.meta || {}, tot = data.totals || {}, vs = data.variants || [];
    $("#run-title").textContent = `${m.scenario || "run"} · ${m.model || ""}`;
    badges.innerHTML = "";
    const addB = (h) => badges.append(el("span", { class: "badge", html: h }));
    if (data.live) badges.append(el("span", { class: "pill live", html: `<span class="live-dot"></span>live` }));
    else {
      const verdict = m.success ? ["ok", "success"]
        : (String(m.stop_reason || "").includes("aborted") ? ["warn", "aborted"] : ["bad", "not fixed"]);
      badges.append(el("span", { class: `pill ${verdict[0]}`, text: verdict[1] }));
    }
    let dbic = m.delta_bic;
    if (dbic == null && m.bic_reference != null) {
      const fb = vs.filter(v => v.bic != null).map(v => v.bic);
      if (fb.length) dbic = Math.min(...fb) - m.bic_reference;  // best-so-far while live
    }
    let kv = m.k;
    if (kv == null) { const fv = vs.filter(v => v.nll != null); if (fv.length) kv = fv[fv.length - 1].k; }
    addB(`<b>ΔBIC</b> ${dbic == null ? "—" : (dbic >= 0 ? "+" : "") + fmt(dbic)}`
      + (m.bic_threshold != null ? ` <span class="muted">/ ${fmt(m.bic_threshold, 0)}</span>` : ""));
    addB(`<b>k</b> ${kv ?? "—"}<span class="muted"> vs ${m.reference?.k ?? "—"}</span>`);
    const T = tot.total || {};
    addB(`<b>${int(T.total)}</b> tok <span class="muted">(${int(T.prompt)}↓ ${int(T.completion)}↑${T.cache_read ? " · " + int(T.cache_read) + " cached" : ""})</span>`);
    addB(`lead <b>${int((tot.lead || {}).total)}</b> · analyst <b>${int((tot.analyst || {}).total)}</b>`);
    addB(`<b>${tot.lead_turns ?? "—"}</b> lead · <b>${tot.analyst_turns ?? "—"}</b> analyst turns`);
    if (!data.live && m.wall_seconds != null) addB(`<b>${fmt(m.wall_seconds, 0)}s</b> wall`);
    if (!data.live) addB(`stop: ${esc(m.stop_reason || "?")}`);

    const px = data.phoenix || {};
    if (!data.live && !px.ok) {
      note.hidden = false; note.classList.remove("live-note");
      note.textContent = "Conversation unavailable — " + (px.note || "no Phoenix trace") +
        ". Model-variant and score panels below are from the on-disk archive.";
    } else if (data.live && !px.ok && (data.turns || []).length === 0) {
      note.hidden = false; note.classList.add("live-note");
      note.textContent = "Live — waiting for the first turns to stream in from Phoenix; the model-variant panel updates from disk in the meantime.";
    } else { note.hidden = true; note.classList.remove("live-note"); }
  }

  // rebuild only the growing tail: the previously-last turn may have been partial (its tool
  // results still arriving), so refresh from there and append the rest. A shrink/reorder (rare;
  // e.g. the finished archive bundle differs) triggers a full rebuild.
  function syncStream(newTurns) {
    clearPlaceholder();
    if (newTurns.length < nodes.length) { stream.innerHTML = ""; nodes = []; }
    const start = Math.max(0, nodes.length - 1);
    while (nodes.length > start) nodes.pop().remove();
    for (let i = start; i < newTurns.length; i++) { const n = turnEl(newTurns[i]); nodes.push(n); stream.append(n); }
    if (newTurns.length === 0) showPlaceholder();
  }

  function draw() {
    renderChart(chart, turns, variants, metric, bicRef, activeTurn,
      (ti) => nodes[ti]?.scrollIntoView({ behavior: "smooth", block: "center" }));
    let av = activeTurn != null && turns[activeTurn] ? turns[activeTurn].active_variant : null;
    if (av == null && variants.length) {  // no conversation to scroll — show the final variant
      const fitted = variants.filter(v => v.nll != null);
      av = (fitted.length ? fitted[fitted.length - 1] : variants[variants.length - 1]).index;
    }
    renderVariantCard(card, variantByIndex[av], meta, metric);
    const nfit = variants.filter(v => v.nll != null).length;
    const norph = variants.filter(v => v.nll != null && (v.no_turn ?? false)).length;
    legend.textContent = `— ${metric === "bic" ? "ΔBIC" : "nll"} (left) · ⋯ cumulative tokens (right) · ${nfit} variant(s), green = new best`
      + (norph ? ` · ◇ ${norph} commit(s) with no turn` : "");
  }

  // scroll-spy: the turn nearest the top of the viewport is the cursor position — EXCEPT at
  // the bottom of the page, where the selection pins to the last (newest) turn, so while
  // following a live run the chart cursor + variant card track the turn being produced.
  function syncScroll() {
    if (!nodes.length) return;
    let top = null, best = Infinity;
    for (const n of nodes) {
      const r = n.getBoundingClientRect();
      if (r.bottom > 150 && r.top < best) { best = r.top; top = n; }
    }
    const idx = atBottom() ? nodes.length - 1 : (top ? top._turn.i : null);
    if (idx == null) return;
    nodes.forEach((n, i) => n.classList.toggle("active", i === idx));
    if (idx !== activeTurn) { activeTurn = idx; draw(); }
  }

  function applyData(data, initial) {
    live = !!data.live;
    meta = data.meta || {}; variants = data.variants || [];
    variantByIndex = Object.fromEntries(variants.map(v => [v.index, v]));
    bicRef = meta.bic_reference ?? null;
    const newTurns = data.turns || [];
    renderHeader(data);
    // Follow the tail like `tail -f`: a live run opens at its newest turn, and each poll keeps
    // the view (and the selected turn) on the last turn while the reader is at the bottom. Once
    // they scroll up to read history, their position is preserved across the tail rebuild.
    const y = window.scrollY;
    const nearBottom = (window.innerHeight + y) >= (document.body.scrollHeight - 220);
    const following = live && (initial || nearBottom);
    syncStream(newTurns);
    turns = newTurns;
    if (initial) activeTurn = turns.length ? (following ? turns.length - 1 : 0) : null;
    else if (activeTurn != null && activeTurn >= turns.length) activeTurn = turns.length ? turns.length - 1 : null;
    draw();
    if (following && turns.length) window.scrollTo(0, document.body.scrollHeight);
    else if (!initial) window.scrollTo(0, y);
    syncScroll();
  }

  async function fetchData(refresh) {
    const url = `/api/run/${encodeURIComponent(window.RUN_ID)}` + (refresh ? "?refresh=true" : "");
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    return resp.json();
  }
  // Poll cadence: the server caches a live bundle for ~15s (a Phoenix trace fetch is heavy), so
  // polling much faster only burns requests; 8s keeps the view responsive without piling on.
  function schedulePoll() { clearTimeout(pollTimer); pollTimer = setTimeout(poll, 8000); }
  async function poll() {
    let data;
    try { data = await fetchData(true); }
    catch (e) { schedulePoll(); return; }   // transient (trace not ingested / hiccup) — retry
    applyData(data, false);
    if (data.live) schedulePoll();           // finished → archived bundle applied, stop polling
  }

  (async () => {
    let data;
    try { data = await fetchData(false); }
    catch (e) {
      $("#run-title").textContent = "failed to load run";
      note.hidden = false; note.textContent = String(e);
      return;
    }
    applyData(data, true);

    $("#metric-toggle").addEventListener("click", (e) => {
      const b = e.target.closest("button"); if (!b) return;
      metric = b.dataset.metric;
      [...$("#metric-toggle").children].forEach(x => x.classList.toggle("on", x === b));
      draw();
    });
    let ticking = false;
    window.addEventListener("scroll", () => {
      if (ticking) return; ticking = true;
      requestAnimationFrame(() => { ticking = false; syncScroll(); });
    }, { passive: true });

    if (data.live) schedulePoll();
  })();
})();
