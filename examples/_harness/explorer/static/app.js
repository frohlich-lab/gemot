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

  const W = 372, H = 158, PL = 40, PR = 42, PT = 12, PB = 24;
  const xs = (i) => PL + (N <= 1 ? (W - PL - PR) / 2 : (i / (N - 1)) * (W - PL - PR));
  const sVals = score.filter(v => v != null);
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
    const det = el("details", { class: "delegation" },
      el("summary", {},
        el("span", { class: "deleg-badge", text: (d.subagent_type || "analyst") }),
        el("span", { class: "deleg-desc", text: d.description || "" }),
        el("span", { class: "deleg-meta", text: `${nsub} turn${nsub === 1 ? "" : "s"}` })));
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
(async function main() {
  let data;
  try {
    const resp = await fetch(`/api/run/${encodeURIComponent(window.RUN_ID)}`);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    data = await resp.json();
  } catch (e) {
    $("#run-title").textContent = "failed to load run";
    $("#phoenix-note").hidden = false; $("#phoenix-note").textContent = String(e);
    return;
  }
  const meta = data.meta || {}, variants = data.variants || [], turns = data.turns || [];
  const tot = data.totals || {};

  // header
  $("#run-title").textContent = `${meta.scenario || "run"} · ${meta.model || ""}`;
  const verdict = meta.success ? ["ok", "success"]
    : (String(meta.stop_reason || "").includes("aborted") ? ["warn", "aborted"] : ["bad", "not fixed"]);
  const badges = $("#run-badges");
  const addB = (h) => badges.append(el("span", { class: "badge", html: h }));
  badges.append(el("span", { class: `pill ${verdict[0]}`, text: verdict[1] }));
  addB(`<b>ΔBIC</b> ${meta.delta_bic == null ? "—" : (meta.delta_bic >= 0 ? "+" : "") + fmt(meta.delta_bic)}`
    + (meta.bic_threshold != null ? ` <span class="muted">/ ${fmt(meta.bic_threshold, 0)}</span>` : ""));
  addB(`<b>k</b> ${meta.k ?? "—"}<span class="muted"> vs ${meta.reference?.k ?? "—"}</span>`);
  const T = tot.total || {};
  addB(`<b>${int(T.total)}</b> tok <span class="muted">(${int(T.prompt)}↓ ${int(T.completion)}↑${T.cache_read ? " · " + int(T.cache_read) + " cached" : ""})</span>`);
  addB(`lead <b>${int((tot.lead || {}).total)}</b> · analyst <b>${int((tot.analyst || {}).total)}</b>`);
  addB(`<b>${tot.lead_turns ?? "—"}</b> lead · <b>${tot.analyst_turns ?? "—"}</b> analyst turns`);
  if (meta.wall_seconds != null) addB(`<b>${fmt(meta.wall_seconds, 0)}s</b> wall`);
  addB(`stop: ${esc(meta.stop_reason || "?")}`);

  if (data.phoenix && !data.phoenix.ok) {
    const n = $("#phoenix-note"); n.hidden = false;
    n.textContent = "Conversation unavailable — " + (data.phoenix.note || "no Phoenix trace") +
      ". Model-variant and score panels below are from the on-disk archive.";
  }

  // stream
  const stream = $("#stream");
  if (turns.length === 0) stream.append(el("div", { class: "muted", text: "no conversation turns (see note above)." }));
  const nodes = turns.map(turnEl);
  nodes.forEach(n => stream.append(n));

  // chart + variant card, synced to scroll
  let metric = "bic";
  const bicRef = meta.bic_reference ?? null;
  const chart = $("#chart"), card = $("#variant-card"), legend = $("#chart-legend");
  const variantByIndex = Object.fromEntries(variants.map(v => [v.index, v]));
  let activeTurn = turns.length ? 0 : null;

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
    legend.textContent = `— ${metric === "bic" ? "ΔBIC" : "nll"} (left) · ⋯ cumulative tokens (right) · ${nfit} variant(s), green = new best`;
  }
  draw();

  // metric toggle
  $("#metric-toggle").addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b) return;
    metric = b.dataset.metric;
    [...$("#metric-toggle").children].forEach(x => x.classList.toggle("on", x === b));
    draw();
  });

  // scroll-spy: the turn nearest the top of the viewport is the cursor position
  function syncScroll() {
    let top = null, best = Infinity;
    for (const n of nodes) {
      const r = n.getBoundingClientRect();
      if (r.bottom > 150 && r.top < best) { best = r.top; top = n; }
    }
    if (!top) return;
    nodes.forEach(n => n.classList.toggle("active", n === top));
    if (top._turn.i !== activeTurn) { activeTurn = top._turn.i; draw(); }
  }
  let ticking = false;
  window.addEventListener("scroll", () => {
    if (ticking) return; ticking = true;
    requestAnimationFrame(() => { ticking = false; syncScroll(); });
  }, { passive: true });
  syncScroll();
})();
