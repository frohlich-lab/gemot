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
// ---------------------------------------------------------------- tool payloads as YAML
// Tool args and results are READ, not copied, and JSON spends most of its width on punctuation:
// braces, quotes around every key, and — the real cost here — `\n` escapes that flatten a model
// listing or a multi-line advice string into one unreadable line. YAML block style drops all of it.
// Dumped, never parsed: this only has to be readable and unambiguous, not round-trippable by
// another tool. Both call sites put the result in a <pre> via textContent, so nothing here is
// HTML-escaped, and nothing here may emit markup.
const Y_BARE = /^[A-Za-z_][\w.\-/]*$/;                        // an id, key or path: safe unquoted
const Y_WORD = /^(?:true|false|null|yes|no|on|off|~)$/i;      // would re-read as a non-string
// Single quotes, doubling any inside: valid YAML for every string, with no backslash escapes to
// reason about. Plain style only where the value cannot be mistaken for a number or a keyword.
const yScalar = (v) => {
  if (v === null || v === undefined) return "null";
  if (typeof v === "number") {
    if (!Number.isFinite(v)) return `'${v}'`;                 // NaN / Infinity are not YAML numbers
    const s = String(v);
    // 1e-12 is a float to YAML 1.2 but a STRING to the 1.1 parsers still in wide use (PyYAML), so
    // give the exponent a mantissa: this is a display format, and it costs two characters to stay
    // readable as a number wherever the reader pastes it.
    return /e/i.test(s) && !s.includes(".") ? s.replace(/e/i, ".0e") : s;
  }
  if (typeof v === "boolean") return String(v);
  const s = String(v);
  return (Y_BARE.test(s) && !Y_WORD.test(s)) ? s : `'${s.replace(/'/g, "''")}'`;
};
// A row of scalars this short reads better on one line than as a column of dashes
const Y_INLINE_MAX = 72;
// `ind` is the column of the parent's key line; this node's own lines sit two further in, so the
// top level is called at -2 and lands at column 0.
function yDump(v, ind) {
  const pad = " ".repeat(Math.max(0, ind + 2));
  if (v === null || typeof v !== "object") {
    const s = String(v);
    if (typeof v === "string" && s.includes("\n")) {   // a block scalar: real lines, not \n escapes
      // The `2` is an explicit indentation indicator, and it is load-bearing rather than tidy: a
      // block scalar whose FIRST line is itself indented (an Antimony listing, a wrapped advice
      // line) is otherwise ambiguous, and the document fails to parse. Body lines always sit two
      // columns beyond their key or dash, so the indicator is always 2.
      // Chomping: `-` strips the final newline, so a string ending in exactly one keeps it under
      // plain `|` (clip). Trailing whitespace is the one thing not reproduced exactly — two or more
      // trailing newlines are dropped, and a clipped one is lost when the block is the document's
      // last line. Invisible in a <pre>, and this is a view, not a serialisation format.
      const chomp = /(^|[^\n])\n$/.test(s) ? "" : "-";
      const body = s.replace(/\n+$/, "").split("\n").map(l => `${pad}${l}`).join("\n");
      return `|2${chomp}\n${body}`;
    }
    return yScalar(v);
  }
  if (Array.isArray(v)) {
    if (!v.length) return "[]";
    if (v.every(x => x === null || typeof x !== "object")) {
      const one = `[${v.map(yScalar).join(", ")}]`;
      if (one.length <= Y_INLINE_MAX && !one.includes("\n")) return one;
    }
    // A container item hoists its first line onto the dash, the conventional form. It costs
    // nothing: this node's children already indent to ind+4, which is exactly where `- ` puts them.
    return "\n" + v.map(x => {
      const inner = yDump(x, ind + 2);
      if (!inner.startsWith("\n")) return `${pad}- ${inner}`;
      const [first, ...rest] = inner.slice(1).split("\n");
      return `${pad}- ${first.trimStart()}${rest.length ? "\n" + rest.join("\n") : ""}`;
    }).join("\n");
  }
  const keys = Object.keys(v);
  if (!keys.length) return "{}";
  return "\n" + keys.map(k => {
    const key = (Y_BARE.test(k) && !Y_WORD.test(k)) ? k : `'${String(k).replace(/'/g, "''")}'`;
    const val = yDump(v[k], ind + 2);
    // a nested block already carries its own leading newline; everything else — scalars and the
    // `|-` block-scalar indicator alike — needs the space after the colon
    return `${pad}${key}:${val.startsWith("\n") ? "" : " "}${val}`;
  }).join("\n");
}
const pretty = (v) => typeof v === "string" ? v : yDump(v, -2).replace(/^\n/, "");

// ---------------------------------------------------------------- LaTeX math -> MathML
// The models write LaTeX in their prose — 613 \(…\), 141 \[…\], 138 $…$ and 19 $$…$$ across the
// traces this viewer reads — which the markdown pass below used to show as raw source. Safari and
// Chrome render MathML Core natively, so a small converter buys real formulas with no library, no
// font and no build step, and it keeps mdToHtml's escape-first invariant: every leaf goes through
// esc() and every element name comes from the tables below, never from the model.
// This is a viewer, not a TeX engine. One left-to-right scan, a symbol table and six shapes
// (fraction, script, root, accent, brace, aligned table) cover the corpus; anything unknown
// degrades to its literal source marked .md-math-raw, so a formula is never silently dropped and
// never silently wrong — the reader can always see which parts are raw.
const M_MI = { // \cmd -> an identifier (italic when it's one char, upright when it's a word)
  alpha: "α", beta: "β", gamma: "γ", delta: "δ", epsilon: "ϵ", varepsilon: "ε", zeta: "ζ", eta: "η",
  theta: "θ", vartheta: "ϑ", iota: "ι", kappa: "κ", lambda: "λ", mu: "μ", nu: "ν", xi: "ξ", pi: "π",
  rho: "ρ", sigma: "σ", varsigma: "ς", tau: "τ", upsilon: "υ", phi: "ϕ", varphi: "φ", chi: "χ",
  psi: "ψ", omega: "ω", Gamma: "Γ", Delta: "Δ", Theta: "Θ", Lambda: "Λ", Xi: "Ξ", Pi: "Π",
  Sigma: "Σ", Upsilon: "Υ", Phi: "Φ", Psi: "Ψ", Omega: "Ω",
  infty: "∞", partial: "∂", nabla: "∇", varnothing: "∅", emptyset: "∅", ell: "ℓ", hbar: "ℏ",
};
const M_MO = { // \cmd -> an operator, relation, arrow or fence
  times: "×", div: "÷", cdot: "⋅", ast: "∗", star: "⋆", circ: "∘", bullet: "∙", pm: "±", mp: "∓",
  oplus: "⊕", ominus: "⊖", otimes: "⊗", odot: "⊙", sum: "∑", prod: "∏", coprod: "∐", int: "∫",
  iint: "∬", oint: "∮", bigcup: "⋃", bigcap: "⋂", approx: "≈", sim: "∼", simeq: "≃", cong: "≅",
  equiv: "≡", neq: "≠", ne: "≠", leq: "≤", le: "≤", geq: "≥", ge: "≥", ll: "≪", gg: "≫",
  propto: "∝", doteq: "≐", in: "∈", notin: "∉", ni: "∋", subset: "⊂", subseteq: "⊆", supset: "⊃",
  supseteq: "⊇", cup: "∪", cap: "∩", setminus: "∖", to: "→", rightarrow: "→", longrightarrow: "⟶",
  Rightarrow: "⇒", leftarrow: "←", longleftarrow: "⟵", Leftarrow: "⇐", leftrightarrow: "↔",
  Leftrightarrow: "⇔", mapsto: "↦", rightleftharpoons: "⇌", rightleftarrows: "⇄",
  leftrightharpoons: "⇋", uparrow: "↑", downarrow: "↓", dashv: "⊣", vdash: "⊢", neg: "¬",
  lnot: "¬", land: "∧", lor: "∨", forall: "∀", exists: "∃", therefore: "∴", ldots: "…", dots: "…",
  cdots: "⋯", vdots: "⋮", ddots: "⋱", prime: "′", perp: "⊥", angle: "∠", degree: "°",
  langle: "⟨", rangle: "⟩", lceil: "⌈", rceil: "⌉", lfloor: "⌊", rfloor: "⌋", vert: "|", Vert: "‖",
  lbrace: "{", rbrace: "}", "{": "{", "}": "}", "|": "‖",
};
// \, \; \! … are ~28% of all command uses in the corpus, so they are the cheapest win here.
const M_SPACE = { ",": "0.167em", ":": "0.222em", ";": "0.278em", " ": "0.25em", "!": "0em",
  quad: "1em", qquad: "2em", thinspace: "0.167em", enspace: "0.5em", negthinspace: "0em" };
const M_TX = { _: "_", "%": "%", "#": "#", $: "$", "&": "&" };  // escaped punctuation, no operator spacing
const M_ACC = { dot: "˙", ddot: "¨", hat: "ˆ", widehat: "ˆ", tilde: "˜", widetilde: "˜", bar: "‾",
  overline: "‾", vec: "→", check: "ˇ", breve: "˘", acute: "´", grave: "`" };
const M_XARROW = { xrightarrow: "→", xleftarrow: "←", xleftrightarrow: "↔", xmapsto: "↦",
  xrightleftharpoons: "⇌", xrightleftarrows: "⇄" };  // \xrightleftharpoons[below]{above}
const M_FUNC = new Set(("log ln lg exp sin cos tan sec csc cot arcsin arccos arctan sinh cosh " +
  "tanh coth max min sup inf lim limsup liminf det dim ker deg gcd arg Pr").split(" "));
// the thin space a function name puts between itself and its operand; named because mathConv has
// to be able to take it back off again when a script follows instead (see the base pop there)
const M_THINSP = '<mspace width="0.167em"></mspace>';
const M_TEXT = new Set(("text textrm textbf textit mbox mathrm mathbf mathit mathsf mathtt " +
  "mathbb mathcal mathscr operatorname").split(" "));  // MathML Core dropped mathvariant, so these
                                                       // all land as upright <mtext>: font lost, letters kept
const M_FENCE = new Set(("left right middle big Big bigg Bigg bigl Bigl biggl Biggl bigr Bigr " +
  "biggr Biggr").split(" "));  // size/stretch hints — MathML stretches ( [ | { by itself
const M_DROP = new Set("displaystyle textstyle scriptstyle limits nolimits mathstrut".split(" "));
const M_FRAC = new Set("frac dfrac tfrac cfrac".split(" "));
// \sum_a^b and friends put their scripts under/over in display mode; an under/over construct
// (\underbrace, \underset) always keeps stacking, or its label would slide off to the right.
const M_STACK = new Set(["<mo>∑</mo>", "<mo>∏</mo>", "<mo>∐</mo>", "<mo>⋃</mo>", "<mo>⋂</mo>",
  "<mi>lim</mi>", "<mi>max</mi>", "<mi>min</mi>", "<mi>sup</mi>", "<mi>inf</mi>"]);
// \begin{env} -> an <mtable>: [columnalign, fences]. Only `aligned` and `pmatrix` show up in
// the corpus; the neighbours cost a line each and spare the reader a wall of raw source.
const M_ENV = { aligned: ["right left", ""], align: ["right left", ""], "align*": ["right left", ""],
  split: ["right left", ""], gathered: ["center", ""], gather: ["center", ""],
  cases: ["left left", "{"], matrix: ["center", ""], pmatrix: ["center", "()"],
  bmatrix: ["center", "[]"], vmatrix: ["center", "||"] };
// own-property lookup only: `\constructor` must not resolve through Object.prototype
const mathHas = (tab, k) => Object.prototype.hasOwnProperty.call(tab, k);

// index of the `}` closing the `{` at i — \{ and \} are literal, so they never nest
function mathBrace(s, i) {
  let d = 0;
  for (let j = i; j < s.length; j++) {
    if (s[j] === "\\") { j++; continue; }
    if (s[j] === "{") d++;
    else if (s[j] === "}" && --d === 0) return j;
  }
  return -1;
}
// an optional [arg], read ONLY straight after a command name: `[S]` in a rate law and `[0,23]`
// in an interval are plain brackets, and there are 22 of those in the corpus. Spaces and tabs are
// skipped, as TeX does and as both sibling argument readers below already did — `\sqrt [3]{V}`
// otherwise took the bracket as its radicand and set a square root of "[". NOT newlines: a bracket
// opening the next line is a concentration, and M_DIM makes the same distinction for `\\ [S]`.
function mathOpt(s, i) {
  let k = i;
  while (s[k] === " " || s[k] === "\t") k++;
  if (s[k] !== "[") return [null, i];       // nothing here: consume nothing, not even the spaces
  const j = s.indexOf("]", k);
  return j < 0 ? [null, i] : [s.slice(k + 1, j), j + 1];
}
// the literal characters of a {…} argument, for \text{state } — spaces and all
function mathRawArg(s, i) {
  while (s[i] === " ") i++;
  if (i >= s.length) throw new Error("missing argument");  // `\text` with nothing after it
  if (s[i] !== "{") return [s[i], i + 1];
  const j = mathBrace(s, i);
  if (j < 0) throw new Error("unbalanced {");
  return [s.slice(i + 1, j), j + 1];
}
// one converted argument: a {group} or, TeX-style, the single atom that follows (x_\varnothing)
function mathArg(s, i) {
  while (s[i] === " " || s[i] === "\n") i++;
  if (i >= s.length) throw new Error("missing argument");
  // one CHARACTER of a number, because that is what TeX takes: \tfrac12 is one half (it shows up
  // in the corpus as \tfrac12\log(252)), and x^12 is x¹ times 2
  if (s[i] >= "0" && s[i] <= "9") return [`<mn>${s[i]}</mn>`, i + 1];
  return mathTok(s, i, false);
}
// the `[4pt]` a row break may carry. Only a real TeX length is eaten: `\\ [S] = 2` starts its row
// with a concentration bracket, and dropping that would be silent content loss.
const M_DIM = /^\[\s*[-+]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)\s*(?:pt|pc|in|bp|cm|mm|dd|cc|sp|ex|em|mu)\s*\]/;
// split a row/environment body at depth 0: row=true splits on `\\` (dropping its optional [4pt]
// skip), row=false on `&`. Depth keeps a \begin{pmatrix} nested in \boxed{…} whole — and counts
// \begin/\end as well as braces, or a nested environment's own `\\` rows and `&` cells would be
// split as the OUTER table's, shredding \begin{cases} into a row of raw <mtext>. An unbalanced `}`
// takes the depth negative, after which nothing would ever split again and the whole body would
// collapse into one cell; that is malformed input, so throw and let the caller show the source.
function mathSplit(s, row) {
  const parts = []; let d = 0, start = 0;
  for (let i = 0; i < s.length; i++) {
    const c = s[i];
    if (c === "{") d++;
    else if (c === "}") { if (--d < 0) throw new Error("unbalanced }"); }
    else if (c === "\\") {
      const env = /^\\(begin|end)\b/.exec(s.slice(i));
      if (env) { d += env[1] === "begin" ? 1 : -1; i += env[0].length - 1; if (d < 0) throw new Error("stray \\end"); continue; }
      if (row && s[i + 1] === "\\" && d === 0) {
        parts.push(s.slice(start, i));
        const dim = M_DIM.exec(s.slice(i + 2));
        start = i + 2 + (dim ? dim[0].length : 0); i = start - 1;
      } else i++;
    } else if (!row && c === "&" && d === 0) { parts.push(s.slice(start, i)); start = i + 1; }
  }
  parts.push(s.slice(start));
  return parts;
}
function mathTable(body, shape) {
  const rl = shape[0] === "right left";
  const rows = mathSplit(body, true);
  // a trailing `\\` ends the last row, it does not open an empty one
  if (rows.length > 1 && !rows[rows.length - 1].trim()) rows.pop();
  const html = rows.map(r => {
    const cells = mathSplit(r, false);
    // only pull a cell right when the row actually has an alignment point, or an aligned row
    // written without `&` drifts away from the column of `=` signs above it
    return `<mtr>${cells.map((c, k) => rl && k === 0 && cells.length > 1
      ? `<mtd class="md-math-r">${mathConv(c, false)}</mtd>`
      : `<mtd>${mathConv(c, false)}</mtd>`).join("")}</mtr>`;
  }).join("");
  const t = `<mtable columnalign="${shape[0]}"${rl ? ' class="md-math-rl"' : ""}>${html}</mtable>`;
  return shape[1] ? `<mrow><mo>${shape[1][0]}</mo>${t}${shape[1][1] ? `<mo>${shape[1][1]}</mo>` : ""}</mrow>` : t;
}
// one \command and whatever arguments it eats
function mathCmd(s, i, display) {
  const m = /^\\([A-Za-z]+|[\s\S])/.exec(s.slice(i));
  const name = m[1], j = i + m[0].length;
  if (mathHas(M_SPACE, name)) return [`<mspace width="${M_SPACE[name]}"></mspace>`, j];
  if (mathHas(M_MI, name)) return [`<mi>${M_MI[name]}</mi>`, j];
  if (mathHas(M_MO, name)) return [`<mo>${esc(M_MO[name])}</mo>`, j];
  if (mathHas(M_TX, name)) return [`<mtext>${esc(M_TX[name])}</mtext>`, j];
  if (M_FUNC.has(name)) {  // a function name followed by its operand takes a thin space, or
    const gap = /^[ \t]*[A-Za-z0-9\\]/.test(s.slice(j))  // \log x sets as "logx"
      ? M_THINSP : "";
    return [`<mi>${name}</mi>${gap}`, j];
  }
  if (M_DROP.has(name)) return ["", j];
  // Text mode keeps its argument verbatim — but TeX's own escapes are not part of the text: without
  // this, \text{50\% of } shows the reader a literal backslash, and as a plain <mtext> rather than
  // .md-math-raw they cannot even tell it is unprocessed source. M_TX handles the same characters
  // in math mode. A \command inside \text{} is still kept literally, which is visible and rare.
  if (M_TEXT.has(name)) {
    const [t, k] = mathRawArg(s, j);
    return [`<mtext>${esc(t.replace(/\\([%_&#${}])/g, "$1"))}</mtext>`, k];
  }
  if (mathHas(M_ACC, name)) { const [a, k] = mathArg(s, j); return [`<mover accent="true">${a}<mo>${M_ACC[name]}</mo></mover>`, k]; }
  if (M_FRAC.has(name)) { const [a, k] = mathArg(s, j), [b, k2] = mathArg(s, k); return [`<mfrac>${a}${b}</mfrac>`, k2]; }
  if (name === "sqrt") {
    const [n, k] = mathOpt(s, j), [a, k2] = mathArg(s, k);
    return [n === null ? `<msqrt>${a}</msqrt>` : `<mroot>${a}<mrow>${mathConv(n, false)}</mrow></mroot>`, k2];
  }
  if (mathHas(M_XARROW, name)) {
    const [u, k] = mathOpt(s, j), [a, k2] = mathArg(s, k);
    const ar = `<mo stretchy="true">${M_XARROW[name]}</mo>`;
    return [u === null ? `<mover>${ar}${a}</mover>`
      : `<munderover>${ar}<mrow>${mathConv(u, false)}</mrow>${a}</munderover>`, k2];
  }
  if (name === "underset" || name === "overset" || name === "stackrel") {
    const [a, k] = mathArg(s, j), [b, k2] = mathArg(s, k);  // {label}{base}
    return [name === "underset" ? `<munder>${b}${a}</munder>` : `<mover>${b}${a}</mover>`, k2];
  }
  if (name === "underbrace" || name === "overbrace") {
    const [a, k] = mathArg(s, j);
    return [name === "underbrace" ? `<munder>${a}<mo>⏟</mo></munder>` : `<mover>${a}<mo>⏞</mo></mover>`, k];
  }
  if (name === "boxed") { const [a, k] = mathArg(s, j); return [`<mrow class="md-math-box">${a}</mrow>`, k]; }
  if (M_FENCE.has(name)) {
    let k = j; while (s[k] === " " || s[k] === "\n") k++;
    if (s[k] === ".") return ["", k + 1];        // \left. — a null delimiter, nothing to draw
    return mathTok(s, k, display);
  }
  if (name === "begin") {
    const [env, k] = mathRawArg(s, j);
    const close = `\\end{${env}}`, e = s.indexOf(close, k);
    if (e < 0 || !mathHas(M_ENV, env)) {            // unknown environment: show it, don't lose it
      const end = e < 0 ? s.length : e + close.length;
      return [`<mtext class="md-math-raw">${esc(s.slice(i, end))}</mtext>`, end];
    }
    return [mathTable(s.slice(k, e), M_ENV[env]), e + close.length];
  }
  return [`<mtext class="md-math-raw">${esc(m[0])}</mtext>`, j];  // unknown: keep the source
}
// one atom: a command, a {group}, a number, a letter, or a lone character
function mathTok(s, i, display) {
  const c = s[i];
  if (c === "\\") return mathCmd(s, i, display);
  if (c === "{") {
    const j = mathBrace(s, i);
    if (j < 0) throw new Error("unbalanced {");
    return [`<mrow>${mathConv(s.slice(i + 1, j), display)}</mrow>`, j + 1];
  }
  const num = /^[0-9]+(?:\.[0-9]+)?/.exec(s.slice(i));
  if (num) return [`<mn>${num[0]}</mn>`, i + num[0].length];
  if (/[A-Za-zΑ-Ωα-ω]/.test(c)) return [`<mi>${c}</mi>`, i + 1];  // one <mi> per letter, like TeX
  if (c === "-") return ["<mo>−</mo>", i + 1];                    // U+2212, the real minus
  if (c === "'") return ["<mo>′</mo>", i + 1];
  return [`<mo>${esc(c)}</mo>`, i + 1];
}
function mathConv(src, display) {
  const out = [];
  let forced = null;   // \limits / \nolimits, pending until the next script pair consumes it
  for (let i = 0; i < src.length;) {
    const c = src[i];
    if (c === " " || c === "\n" || c === "\t" || c === "\r") { i++; continue; }  // TeX eats space
    // \limits and \nolimits are not atoms — they retarget the scripts that FOLLOW them — so they
    // are read here rather than dropped in mathCmd: dropping them there left "" on the stack for
    // the next _ or ^ to take as its base, emitting an <msubsup> with no base and orphaning the
    // operator. Which is how the one command whose whole purpose is stacking defeated the stacking.
    const lim = c === "\\" && /^\\(limits|nolimits)(?![A-Za-z])/.exec(src.slice(i));
    if (lim) { forced = lim[1] === "limits"; i += lim[0].length; continue; }
    if (c === "_" || c === "^") {
      // scripts bind to the atom just emitted; take a _ ^ pair in one go so x_i^2 stacks as an
      // msubsup instead of staggering the exponent above the subscript
      let base = out.pop() || "<mrow></mrow>";   // || not ??: a dropped command leaves "", not undefined
      // A script takes exactly ONE base element. The only token that is two is a function name that
      // pulled a thin space in for its operand (\log x), and if a script follows instead there is no
      // operand to separate — so drop the space rather than hand <munder> a third child.
      if (base.endsWith(M_THINSP)) base = base.slice(0, -M_THINSP.length);
      let sub = null, sup = null;
      while ((src[i] === "_" && sub === null) || (src[i] === "^" && sup === null)) {
        const which = src[i], [a, n] = mathArg(src, i + 1);
        if (which === "_") sub = a; else sup = a;
        i = n; while (src[i] === " ") i++;
      }
      const st = forced ?? (base.startsWith("<munder>") || base.startsWith("<mover>")
        || (display && M_STACK.has(base)));
      forced = null;
      const [lo, hi] = st ? ["under", "over"] : ["sub", "sup"];
      out.push(sub !== null && sup !== null ? `<m${lo}${hi}>${base}${sub}${sup}</m${lo}${hi}>`
        : sub !== null ? `<m${lo}>${base}${sub}</m${lo}>` : `<m${hi}>${base}${sup}</m${hi}>`);
      continue;
    }
    const [tok, n] = mathTok(src, i, display);
    // a command that renders nothing (\displaystyle, a \left. null delimiter) is not an atom:
    // pushing "" would hand it to a following script as the base
    if (tok) { out.push(tok); forced = null; }
    i = n;
  }
  return out.join("");
}
// A display formula written inside a `>` blockquote carries the quote prefixes into its body,
// because math is extracted before the block loop below sees the quote. Strip them only when
// EVERY continuation line has one: `a\n> b` is a relation at the start of a line, and deleting
// that `>` would turn an inequality into a silent juxtaposition.
function mathDequote(tex) {
  const lines = tex.split("\n");
  if (lines.length < 2 || !lines.slice(1).every(l => !l.trim() || /^[ \t]*>/.test(l))) return tex;
  return lines.map(l => l.replace(/^[ \t]*>[ \t]?/, "")).join("\n");
}
// One formula -> one <math>. Anything that trips the scanner (unbalanced braces, a truncated
// command) falls back to the LaTeX source: a wrong-looking formula is worse than visible source,
// and a missing one is worst of all.
function mathToMathML(tex, display) {
  try {
    const inner = mathConv(mathDequote(tex), display);
    if (!inner) throw new Error("empty");
    return `<math${display ? ' display="block"' : ""}>${inner}</math>`;
  } catch (_e) {
    return `<code class="md-math-raw">${esc(tex.trim())}</code>`;
  }
}
// $$ comes FIRST in this alternation: scanning left to right in one pass is what stops the
// `\\[4pt]` row break inside a $$ block from being read as a \[ display opener. The bodies are
// length-bounded (the longest real formula is 594 chars) so an unclosed delimiter in a long
// message costs one bounded rescan instead of one per character; an over-long body stays source.
const MATH_RE = /\$\$([\s\S]{1,1200}?)\$\$|\\\[([\s\S]{1,1200}?)\\\]|\\\(([\s\S]{1,1200}?)\\\)|\$([^\n$]{1,200}?)\$/g;
// Single $…$ IS real math here (138 occurrences, 3 of the 4 projects — `$ K_p $`, `$\tau$`), but
// it is the one delimiter a lone $ in prose can fake, so it must look like a formula: one
// operator/script/brace, or a single bare token. "$5 to $10" and a backticked Antimony `$` fail.
// AND it must contain no English: two shell/parameter names either side of a sentence
// ("$SLURM_JOB_ID and $HOME", "we sweep $k_deg and $k_syn") satisfy every character test above,
// because `_` is exactly the character this file elsewhere refuses to treat as markup — snake_case
// ids are everywhere in this domain. A word is a run of 3+ letters left after commands and brace
// groups are removed, so `V_{max}S/(K_m + S)` and `n > 2` pass while `x/y ratio for the` does not.
// Checked against the corpus: keeps all 60 real single-$ formulas, rejects every prose case.
const mathProseWord = (b) => {
  let s = b.replace(/\\[A-Za-z]+/g, "");
  for (let i = 0; i < 4; i++) s = s.replace(/\{[^{}]*\}/g, "");   // innermost group out, a few deep
  return /[A-Za-z]{3,}/.test(s);
};
const mathIsFormula = (b) => {
  if (b.includes("`")) return false;
  // a body that opens with a digit is a price far more often than a formula, unless it also
  // carries real notation: "$5-$8", "$50 (list) and $70", "$200/day … $300/day" are all prose
  if (/^\s*[0-9]/.test(b) && !/[\\^_]/.test(b)) return false;
  if (mathProseWord(b)) return false;
  return /[\\^_{}=<>+\-*/|()]/.test(b) || !/\s/.test(b.trim());
};
const mathBlock = (tex) => `<div class="md-math">${mathToMathML(tex, true)}</div>`;
// One left-to-right scan, stashing each formula as ph(i). A match this rejects resumes ONE
// character in, not past the whole match: "costs $5 and $$a=b$$" must not let the rejected `$5
// and $` swallow the real formula's opener. A body holding the fenced-code placeholder is
// rejected too — swallowing one would delete the code block it stands for (the model's own text
// has no `@@` in it, in 4M chars of traces, so this costs nothing).
function mathStash(src, maths, ph) {
  let out = "", last = 0, m;
  MATH_RE.lastIndex = 0;
  while ((m = MATH_RE.exec(src))) {
    const tex = m[1] ?? m[2] ?? m[3] ?? m[4];
    // A body holding either sentinel is rejected: swallowing the fenced-code one would delete the
    // block it stands for, and swallowing an inline-code one would end up restoring a <code>
    // element inside the emitted <math>.
    if (!tex.trim() || tex.includes("@@") || tex.includes("\u0011")
        || (m[4] !== undefined && !mathIsFormula(tex))) {
      MATH_RE.lastIndex = m.index + 1; continue;
    }
    maths.push({ tex, display: m[1] !== undefined || m[2] !== undefined });
    out += src.slice(last, m.index) + ph(maths.length - 1);
    last = MATH_RE.lastIndex;
  }
  return out + src.slice(last);
}

// ---------------------------------------------------------------- GFM pipe tables
// The analysts tabulate constantly (340 table rows across the traces): per-observable ranges,
// parameter listings, fit comparisons. Without this the block loop below joined every row into one
// paragraph, so a 15-row table arrived as a single line of pipes -- the least readable form of the
// most structured thing in the message.
// A table is a header row plus a delimiter row whose cell COUNT MATCHES, which is what keeps a bare
// `---` under a line that happens to contain a pipe from being read as a one-column table.
const T_ROW = /\|/;
const T_CELL = /^\s*:?-+:?\s*$/;
// split on unescaped pipes, dropping the optional leading/trailing one
const tCells = (row) => row.replace(/^\s*\|/, "").replace(/\|\s*$/, "")
  .split(/(?<!\\)\|/).map(c => c.trim().replace(/\\\|/g, "|"));
const tAlign = (c) => /^:-+:$/.test(c) ? "c" : /-+:$/.test(c) ? "r" : /^:-+/.test(c) ? "l" : "";
// alignment comes from the delimiter row this code parsed, never from model text, so the class
// names here are ours and a cell can only ever land inside <th>/<td> as escaped content
const T_CLASS = { l: ' class="md-t-l"', c: ' class="md-t-c"', r: ' class="md-t-r"', "": "" };
function tIsDelim(line, ncols) {
  const cells = tCells(line);
  return cells.length === ncols && cells.every(c => T_CELL.test(c) && c.includes("-"));
}
// The wrapper is the scroll box: a table too wide to fold would otherwise be clipped by .turn's
// overflow:hidden, exactly as a wide formula would.
function tRender(head, align, rows, inline) {
  const th = head.map((c, i) => `<th${T_CLASS[align[i] || ""]}>${inline(c)}</th>`).join("");
  const tb = rows.map(r => `<tr>${head.map((_, i) =>
    `<td${T_CLASS[align[i] || ""]}>${inline(r[i] ?? "")}</td>`).join("")}</tr>`).join("");
  return `<div class="md-table-wrap"><table class="md-table"><thead><tr>${th}</tr></thead>`
    + (tb ? `<tbody>${tb}</tbody>` : "") + "</table></div>";
}
// Minimal, SAFE markdown -> HTML for the agent's assistant text (the lead + analysts write
// markdown: headers, **bold**, *italic*, `code`, fenced blocks, -/1. lists, > quotes, links, and
// \(…\)/\[…\]/$…$/$$…$$ LaTeX math). Everything is HTML-escaped FIRST, so only the tags this
// emits are ever HTML — the model's text can't inject markup. Math becomes MathML Core, which
// Safari and Chrome render natively (no library, no font, no build step) and whose element names
// come from a fixed whitelist, so the same invariant holds inside a formula. Underscores are left
// alone on purpose (snake_case ids like `S_init` are everywhere in this domain and must not
// become italics).
function mdToHtml(src) {
  if (!src) return "";
  const blocks = [];
  src = String(src).replace(/```([^\n]*)\n?([\s\S]*?)```/g, (_m, info, code) => {
    blocks.push(info.trim() === "math" ? mathBlock(code)  // a ```math fence is a formula, not source
      : `<pre class="md-pre"><code>${esc(code.replace(/\n$/, ""))}</code></pre>`);
    return `@@${blocks.length - 1}@@`;
  });
  // Math is stashed before ANY markdown runs, for the same reason fenced code is: `*` and `_`
  // inside a formula are not emphasis, and a multi-line \[…\] must not be shredded into list
  // items and paragraphs by the block loop below. The sentinel carries a per-call random tag, so
  // the model cannot forge one: a literal placeholder in its prose must not resolve to somebody
  // else's formula, or splice MathML into a code block. It survives esc() and every inline rule,
  // and can't be mistaken for a code block by the whole-line test in that loop.
  const maths = [], tag = "\u0011" + Math.random().toString(36).slice(2, 10);
  const ph = (n) => `${tag}M${n}\u0011`;    // the trailing \u0011 fences the index off from
                                            // adjacent prose, and is not whitespace, so a
                                            // trimmed line still matches the placeholder exactly
  const phAll = new RegExp(`${tag}M(\\d+)\u0011`, "g"), phOne = new RegExp(`^${tag}M(\\d+)\u0011$`);
  // the formula index if this text is nothing but one display placeholder, else -1
  const dispAt = (s) => { const m = phOne.exec(s); return m && maths[+m[1]].display ? +m[1] : -1; };
  // An inline `code` span is stashed BEFORE math, for the same reason a fenced block already
  // was: LaTeX the model deliberately quoted AS SOURCE -- `\(x^2\)`, or an Antimony `$K_p$`
  // id -- must stay source rather than become a formula rendered inside <code>. A span holding
  // a blank line is not a code span (markdown agrees) so it is left to the backtick rule below,
  // and the length bound stops an unpaired backtick from scanning to the end of a long message.
  const codes = [], cph = (n) => `${tag}C${n}\u0011`;
  const cphAll = new RegExp(`${tag}C(\\d+)\u0011`, "g");
  src = src.replace(/`([^`]{1,500}?)`/g, (m, c) =>
    /\n\s*\n/.test(c) ? m : (codes.push(c), cph(codes.length - 1)));
  src = mathStash(src, maths, ph);
  const inline = (s) => esc(s)
    .replace(/`([^`]+)`/g, (_m, c) => `<code>${c}</code>`)
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\*([^*\n]+)\*/g, "<em>$1</em>")
    // A URL may not contain a math placeholder: math is restored AFTER this pass, so a
    // placeholder left inside the href would splice unescaped MathML — quotes and all — into
    // an attribute value, which is how model text could name attributes on the <a>.
    .replace(/\[([^\]]+)\]\((https?:[^)\s\u0011]+)\)/g,
      (_m, t, u) => `<a href="${u.replace(/"/g, "&quot;")}" target="_blank" rel="noopener">${t}</a>`);
  const out = [];
  let list = null, para = [];
  const flushPara = () => { if (para.length) { out.push(`<p>${inline(para.join(" "))}</p>`); para = []; } };
  const closeList = () => { if (list) { out.push(`</${list}>`); list = null; } };
  const lines = src.split("\n");
  for (let li = 0; li < lines.length; li++) {
    const line = lines[li];
    const t = line.trim(); let m;
    if ((m = /^@@(\d+)@@$/.exec(line))) { flushPara(); closeList(); out.push(blocks[+m[1]]); continue; }
    // display math alone on its line is block-level, like a code block — the scroll box that
    // keeps a wide formula from clipping only exists on the block form
    let d = dispAt(t);
    if (d >= 0) {
      flushPara();
      const b = mathBlock(maths[d].tex);
      // Under an OPEN list the formula belongs to the item above it, indented or not, and is
      // spliced into that <li>: closing and reopening the list would restart an <ol> at 1. Reviewed
      // and deliberately NOT keyed on indentation — a line straight after an item with no blank
      // line between is that item's lazy continuation (CommonMark agrees), and requiring indentation
      // measurably splits `1. a / \[x\] / 2. b` into two <ol>s. A formula that really is top-level
      // is separated by a blank line, which closed the list before this branch is ever reached.
      if (list && out.length && out[out.length - 1].endsWith("</li>"))
        out[out.length - 1] = out[out.length - 1].slice(0, -5) + b + "</li>";
      else { closeList(); out.push(b); }
      continue;
    }
    // a table: this row plus a delimiter row under it with the same number of cells
    if (T_ROW.test(t) && li + 1 < lines.length && tIsDelim(lines[li + 1].trim(), tCells(t).length)) {
      flushPara(); closeList();
      const head = tCells(t), align = tCells(lines[li + 1].trim()).map(tAlign);
      const rows = [];
      li++;                                        // the delimiter row is consumed, never rendered
      while (li + 1 < lines.length && T_ROW.test(lines[li + 1].trim()) && lines[li + 1].trim())
        rows.push(tCells(lines[++li].trim()));
      out.push(tRender(head, align, rows, inline));
      continue;
    }
    if (t === "") { flushPara(); closeList(); continue; }
    if ((m = /^(#{1,6})\s+(.*)$/.exec(t))) { flushPara(); closeList(); const n = m[1].length; out.push(`<h${n} class="md-h">${inline(m[2])}</h${n}>`); continue; }
    if (/^(---+|\*\*\*+|___+)$/.test(t)) { flushPara(); closeList(); out.push("<hr>"); continue; }
    if ((m = /^[-*+]\s+(.*)$/.exec(t))) { flushPara(); if (list !== "ul") { closeList(); out.push("<ul>"); list = "ul"; } out.push(`<li>${inline(m[1])}</li>`); continue; }
    if ((m = /^\d+[.)]\s+(.*)$/.exec(t))) { flushPara(); if (list !== "ol") { closeList(); out.push("<ol>"); list = "ol"; } out.push(`<li>${inline(m[1])}</li>`); continue; }
    if ((m = /^>\s?(.*)$/.exec(t))) {
      flushPara(); closeList();
      d = dispAt(m[1].trim());   // a quoted formula keeps its block form, hence its scroll box
      out.push(`<blockquote>${d >= 0 ? mathBlock(maths[d].tex) : inline(m[1])}</blockquote>`);
      continue;
    }
    para.push(t);
  }
  flushPara(); closeList();
  // Restore the math left inside paragraphs, list items and quotes. Always inline here: a display
  // formula that shared its line with prose renders inline rather than dropping a <div> inside a
  // <p>. Every index here was minted above, so there is no missing-formula case to guard.
  return out.join("\n").replace(phAll, (_m, k) => mathToMathML(maths[+k].tex, false))
    .replace(cphAll, (_m, k) => `<code>${esc(codes[+k])}</code>`);
}

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
// make_changes, right = cumulative token sum, plus a vertical cursor at the scrolled turn.
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
  // committed variants that no turn produced (a dropped make_changes span) — shown as gap markers
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
    // dots at make_changes turns
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
    el("span", { class: "m", html: `<span>k</span> ${v.k ?? "—"}` }),
    // how many of the multistart's starts reached the best fval; how they ended on hover
    v.starts ? el("span", { class: "m", title: Object.entries(v.starts.exitflag_counts || {})
        .map(([f, n]) => `${f.replace(/^ExitFlag\./, "")} ${n}`).join(", ") || false },
      el("span", { text: "at best" }), ` ${v.starts.n_at_best ?? "—"}/${v.starts.n_starts ?? "—"}`) : null);
  card.append(head, metrics);
  if (v.justification) card.append(el("div", { class: "vc-just" },
    el("span", { class: "lbl", text: "justification" }), v.justification));
  // On the INDRA arm model.ant is the BioNetGen expansion of what the agent wrote; this is what
  // it wrote.
  const rows = v.statements && v.statements.rows;
  if (rows && rows.length) card.append(el("details", { class: "vc-stmts" },
    el("summary", { text: `${rows.length} statement${rows.length === 1 ? "" : "s"}` }),
    el("pre", { class: "vc-ant", text: pretty(v.statements) })));
  card.append(el("pre", { class: "vc-ant", html: highlightAntimony(v.antimony) }));
}

// ---------------------------------------------------------------- turn stream
// A span that ended in ERROR (`error` is its status message): a badge, and the message itself.
// Without them a delegation that died and was run again read as a duplicate with no result (#125).
const failedBadge = (error) => error ? el("span", { class: "badge-failed", title: error, text: "failed" }) : null;
const errorEl = (error) => error ? el("div", { class: "span-error", text: error }) : null;

function toolCallEl(tc) {
  if (tc.delegation) {
    const d = tc.delegation;
    const nsub = d.items.filter(x => x.type === "assistant").length;
    const det = el("details", { class: "delegation" + (d.in_progress ? " running" : "") + (tc.error ? " failed" : ""), open: d.in_progress || false },
      el("summary", {},
        el("span", { class: "deleg-badge", text: (d.subagent_type || "analyst") }),
        el("span", { class: "deleg-desc", text: d.description || "" }),
        d.in_progress ? el("span", { class: "deleg-live", html: `<span class="live-dot"></span>running` }) : null,
        failedBadge(tc.error),
        el("span", { class: "deleg-meta", text: `${nsub} turn${nsub === 1 ? "" : "s"}${d.in_progress ? " so far" : ""}` })));
    const box = el("div", { class: "subturns" }, errorEl(tc.error));
    let cur = null;
    for (const it of d.items) {
      if (it.type === "assistant") {
        cur = el("div", { class: "subturn" },
          el("div", { class: "turn-role", text: d.subagent_type || "analyst" }, failedBadge(it.error)),
          el("div", { class: "assistant-text md" + (it.text ? "" : " empty"),
            ...(it.text ? { html: mdToHtml(it.text) } : { text: "(tool calls only)" }) }),
          errorEl(it.error));
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
  // `set_model` is the pre-consolidation name; traces recorded before the collapse still use it.
  const isSet = tc.name === "make_changes" || tc.name === "set_model";
  const det = el("details", { class: "toolcall" + (isSet ? " setmodel" : "") + (tc.error ? " failed" : "") },
    el("summary", {}, el("span", { class: "tool-name", text: tc.name || "tool" }), failedBadge(tc.error)));
  const body = el("div", { class: "kv" }, errorEl(tc.error));
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
    failedBadge(t.error),
    badge);
  const body = el("div", { class: "turn-body" },
    el("div", { class: "assistant-text md" + (t.text ? "" : " empty"),
      ...(t.text ? { html: mdToHtml(t.text) } : { text: "(tool calls only)" }) }),
    errorEl(t.error));
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
    // The delta that DECIDED the verdict, and the scale it is on: `delta_criterion` once the
    // judge has run, else canonical ΔBIC (the same number whenever BIC is the criterion). Showing
    // ΔBIC beside a verdict decided on ΔAICc put a passing run outside its own bar.
    let dbic = m.delta_criterion != null ? m.delta_criterion : m.delta_bic;
    let dlabel = (m.delta_criterion != null ? (m.criterion || "criterion")
                                            : (m.reference_criterion || "bic")).toUpperCase();
    if (dbic == null && m.bic_reference != null) {
      const fb = vs.filter(v => v.bic != null).map(v => v.bic);
      // best-so-far while live: the marker publishes canonical BIC only, so say BIC.
      if (fb.length) { dbic = Math.min(...fb) - m.bic_reference; dlabel = "BIC"; }
    }
    let kv = m.k;
    if (kv == null) { const fv = vs.filter(v => v.nll != null); if (fv.length) kv = fv[fv.length - 1].k; }
    addB(`<b>Δ${dlabel}</b> ${dbic == null ? "—" : (dbic >= 0 ? "+" : "") + fmt(dbic)}`
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
    } else if (px.complete === false) {  // a finished run whose trace Phoenix holds only part of
      note.hidden = false; note.classList.remove("live-note");
      note.textContent = "Conversation incomplete — " + px.note;
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
