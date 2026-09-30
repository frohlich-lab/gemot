"""The explorer's LaTeX -> MathML converter, exercised in a real JS engine.

``mdToHtml`` and its math converter live in ``explorer/static/app.js``, which no Python test can
reach -- and the properties that matter here are exactly the ones review cannot eyeball: that
every leaf is escaped, that only whitelisted MathML tags and attributes are ever emitted, and that
an unsupported command degrades to its literal source instead of vanishing. So run the real thing:
slice off app.js's pure-function prelude (everything above ``(function main()``, which is the DOM
half) and evaluate it in ``node``.

Skipped where node is absent -- but CI's ubuntu runner ships one, and ci.yml adds an explicit
``setup-node`` step so that skip never happens silently there (the same reason the SWIG install is
explicit rather than transitive).
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path

import pytest

APP_JS = Path(__file__).parent.parent / "src" / "gemot" / "explorer" / "static" / "app.js"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")

# Where app.js stops being pure string->string helpers and starts touching the DOM. Splitting
# here is what lets a bare `node` evaluate the file at all.
_PRELUDE_END = "\n(function main()"

# Every MathML element the converter is allowed to emit. MathML Core only -- no `mfenced`, no
# `menclose` (dropped from Core, unimplemented in Chromium).
_MATHML_OK = {
    "math",
    "mrow",
    "mi",
    "mn",
    "mo",
    "mtext",
    "mspace",
    "mfrac",
    "msqrt",
    "mroot",
    "msub",
    "msup",
    "msubsup",
    "munder",
    "mover",
    "munderover",
    "mtable",
    "mtr",
    "mtd",
}
_HTML_OK = {
    "p",
    "code",
    "pre",
    "strong",
    "em",
    "a",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "ul",
    "ol",
    "li",
    "blockquote",
    "hr",
    "span",
    "div",
    "table",
    "thead",
    "tbody",
    "tr",
    "th",
    "td",
}
# Every attribute the renderer may put on an element. The values are fixed strings chosen by
# app.js, never model text -- which is what keeps esc() (which does not escape the double quote)
# sufficient. `a@href` is the pre-existing link rule, quote-escaped there.
_ATTRS_OK = {
    ("div", "class"),
    ("span", "class"),
    ("code", "class"),
    ("pre", "class"),
    ("mtext", "class"),
    ("mrow", "class"),
    ("mtd", "class"),
    ("mtable", "class"),
    ("mtable", "columnalign"),
    ("math", "display"),
    ("mo", "stretchy"),
    ("mover", "accent"),
    ("mspace", "width"),
    ("a", "href"),
    ("a", "target"),
    ("a", "rel"),
    ("table", "class"),
    ("th", "class"),
    ("td", "class"),
} | {(f"h{n}", "class") for n in range(1, 7)}


def _prelude() -> str:
    src = APP_JS.read_text(encoding="utf-8")
    head, sep, _ = src.partition(_PRELUDE_END)
    assert sep, f"{APP_JS.name} no longer has a `(function main()` boundary to slice at"
    return head


def _md(*sources: str) -> list[str]:
    """``mdToHtml`` applied to each source, evaluated in node. Cases go over stdin so no LaTeX
    backslash ever has to survive an argv round-trip."""
    runner = (
        _prelude()
        + """
let raw = ""; process.stdin.on("data", d => raw += d);
process.stdin.on("end", () => {
  process.stdout.write(JSON.stringify(JSON.parse(raw).map(mdToHtml)));
});
"""
    )
    proc = subprocess.run(
        ["node", "-e", runner], input=json.dumps(list(sources)), capture_output=True, text=True
    )
    assert proc.returncode == 0, f"node failed evaluating app.js:\n{proc.stderr}"
    return json.loads(proc.stdout)


def _tags(html: str) -> list[str]:
    return [t.lower() for t in re.findall(r"<\s*/?\s*([A-Za-z][\w-]*)", html)]


class _Parse(HTMLParser):
    """What a browser would actually build -- the only honest way to ask whether model text ever
    became an element name or an attribute."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tags: list[str] = []
        self.attrs: list[tuple[str, str, str]] = []
        self.text: list[str] = []

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        self.attrs.extend((tag, k, v or "") for k, v in attrs)

    def handle_data(self, data):
        self.text.append(data)


def _dom(html: str) -> _Parse:
    p = _Parse()
    p.feed(html)
    p.close()
    return p


# -- the invariant ---------------------------------------------------------
def test_math_emits_only_whitelisted_tags():
    """The escape-first invariant, checked from the outside: whatever the model wrote, the only
    element names that come out are ones this converter chose."""
    for html in _md(
        r"inline \(\frac{a_i}{d}\) and display \[ \sum_{i=1}^{16} x_i \approx 1. \]",
        r"$$ P(\text{state } S) = \prod_{i \in S} \frac{a_i}{a_i + 1} $$",
        r"\(\chi^2 \approx 1.77\times 10^5,\qquad \chi^2_\mathrm{dof}\approx753.\)",
        "\\[\n\\boxed{J_* = \\begin{pmatrix} -0.463 & -0.852 \\\\ 0.269 & 0 \\end{pmatrix}}\n\\]",
        r"\[ x_S \xrightleftharpoons[k^-_j]{k^+_j} x_{S\cup\{j\}}, \]",
    ):
        assert set(_tags(html)) <= _MATHML_OK | _HTML_OK, html


@pytest.mark.parametrize(
    "src",
    [
        r"\(<img src=x onerror=alert(1)>\)",
        r"\(a \href{javascript:alert(1)}{b}\)",
        r"\(</math><script>alert(1)</script>\)",
        r"$$ \text{</math><script>alert(1)</script>} $$",
        r"\(\mathrm{<svg/onload=alert(1)>}\)",
        r"\[\begin{evil}<script>1</script>\end{evil}\]",
        '\\[ \\begin{aligned} <a href="javascript:1">x</a> & = 2 \\end{aligned} \\]',
        r"\(\text{" '"' r' onmouseover="alert(1)}\)',
        r"- \(\text{</li></ul><img src=x>}\)",
        "> \\[ a </blockquote><script>1</script> \\]",
        # a formula inside a link URL: math is restored after the link rule, so a placeholder left in
        # the href would splice unescaped MathML (quotes and all) into an attribute value
        r'[ref](https://e.com/\(\text{" style="animation:live-pulse 1s" onanimationstart="x()" q="}\))',
    ],
)
def test_math_cannot_inject_markup(src):
    r"""Model text is escaped before anything else, so `<`/`&` inside math stay text. Asserting on
    what a parser BUILDS rather than on substrings, because an escaped payload legitimately still
    spells "script" and "onerror" -- as text."""
    (html,) = _md(src)
    dom = _dom(html)
    assert set(dom.tags) <= _MATHML_OK | _HTML_OK, html
    assert {(t, k) for t, k, _ in dom.attrs} <= _ATTRS_OK, html
    for tag, key, val in dom.attrs:
        assert not key.startswith("on"), html
        if key == "href":
            assert not val.strip().lower().startswith("javascript:"), html


def test_the_payload_survives_as_text():
    r"""The other half of escape-first: nothing is dropped, it is just escaped and split across
    MathML leaves (one <mi> per letter, like TeX), so read the text back out to check."""
    (html,) = _md(r"\(a < b\) and \[<img src=x onerror=alert(1)>\]")
    assert "<img" not in html
    assert "<mo>&lt;</mo>" in html
    text = "".join(_dom(html).text)
    assert "".join(text.split()) == "a<band<imgsrc=xonerror=alert(1)>", text


def test_a_math_body_cannot_borrow_the_code_block_placeholder():
    """The fenced-code stash restores `@@n@@` by index, so a formula must not be able to swallow
    one: that would delete the code block it stands for. The math sentinel itself is unforgeable
    (a per-call random tag), so a literal one in the prose is just text."""
    (html,) = _md("```\nreal code\n```\n\n\\[@@0@@\\] and \\(@@M0@@\\)")
    assert "real code" in html  # the code block still resolves to its own text
    assert "<math" not in html  # neither body reached into a stash ...
    assert html.count("@@0@@") == 1 and "@@M0@@" in html  # ... they stayed literal source


def test_a_mispaired_delimiter_does_not_eat_a_code_block():
    """One unclosed `\\[` pairing with a `\\]` further down used to swallow the fenced-code
    placeholder between them, and the code vanished from the page."""
    for html in _md(
        "opened \\[a\n\n```\nS1 -> P; k1*S1\n```\n\nand closed b\\]",
        "cost $$a\n\n```python\nprint('important')\n```\n\nb$$",
    ):
        assert "S1 -&gt; P; k1*S1" in html or "print('important')" in html, html


def test_a_forged_placeholder_does_not_duplicate_a_formula():
    """The sentinel carries a per-call random tag, so model text cannot name someone else's
    formula (which would both destroy the literal text and render the formula twice)."""
    (html,) = _md("see \\(x\\) and the literal text \u0011abc12345M0\u0011 here")
    assert html.count("<math") == 1, html


# -- never drop content ----------------------------------------------------
def test_an_unsupported_command_keeps_its_literal_source():
    """Never silently drop content -- an unknown command must still be readable, and marked so the
    reader can tell raw source from a rendered formula."""
    (html,) = _md(r"\(\somethingundefined{x} + \frac{a}{b}\)")
    assert r"\somethingundefined" in html
    assert 'class="md-math-raw"' in html
    assert "<mfrac>" in html  # the rest of the formula still renders


def test_a_malformed_formula_falls_back_to_its_source():
    r"""Unbalanced braces mean the scanner cannot know the shape; showing the LaTeX beats showing
    a confidently wrong formula, and beats showing nothing at all."""
    for src in (r"\(\frac{a}{b\)", r"\(\text\)", r"\(\dot\)"):
        (html,) = _md(src)
        assert 'class="md-math-raw"' in html, html
        assert src[2:-2] in html, html
        assert "<math" not in html, html


def test_a_relation_at_the_start_of_a_line_is_not_a_quote_marker():
    r"""A blockquoted display formula carries `> ` prefixes into its body, so they are stripped --
    but only when EVERY line has one, or `a\n> b` loses its inequality and reads as `ab`."""
    (html,) = _md("\\[\n  a\n  > b\n\\]")
    assert "<mo>&gt;</mo>" in html, html
    (html,) = _md("> as follows\n>\n> \\[\n>   v_{r1}= \\frac{S}{K_m+S}\n>   \\]\n")
    assert "&gt;" not in html, html  # prefixes gone from the formula
    assert '<blockquote><div class="md-math">' in html  # and it kept its scroll box


def test_a_concentration_bracket_after_a_row_break_survives():
    r"""`\\[4pt]` is a row skip; `\\ [S] = 2` starts the next row with a species concentration."""
    (html,) = _md(r"\[\begin{aligned}v &= 1 \\[S] &= 2\end{aligned}\]")
    assert "<mo>[</mo>" in html, html
    (html,) = _md("$$\\begin{aligned}\na &= 1 \\\\[4pt]\nb &= 2\n\\end{aligned}$$")
    assert "4pt" not in html and html.count("<mtr>") == 2, html


# -- the delimiters -------------------------------------------------------
def test_display_math_is_a_block_not_a_paragraph():
    r"""``\[...\]`` on its own line must not land inside a ``<p>``: it is block-level, and the CSS
    that keeps a wide formula from being clipped only applies to the block form."""
    (html,) = _md(r"\[ x_S \propto \prod_{i\in S}\frac{a_i}{d}, \]")
    assert 'display="block"' in html
    assert html.startswith('<div class="md-math">')


def test_a_row_break_inside_dollars_is_not_a_display_opener():
    r"""The scan is one left-to-right alternation with $$ first, so the ``\\[4pt]`` row skip inside
    an aligned block cannot be mistaken for a ``\[`` display opener and shred it."""
    (html,) = _md("$$\\begin{aligned}\na &= 1 \\\\[4pt]\nb &= 2\n\\end{aligned}$$")
    assert html.count("<mtr>") == 2
    assert "4pt" not in html


@pytest.mark.parametrize(
    "src",
    [
        "costs $5 to $10 per run",
        "pay $foo and $bar",
        "$1,000 and $2,500 budgets",
        "the `$` modifier and a `$` again",
        "Total spend: $12 (compute) + $3 (storage)",
        "budget $50-$80 per sweep",
        "at $0.50/token vs $2 elsewhere",
    ],
)
def test_a_bare_dollar_in_prose_is_not_math(src):
    """Single ``$...$`` is real math in these traces, so it is a delimiter -- but it is the one a
    lone `$` in prose can fake, hence the it-must-look-like-a-formula guard. A body that opens
    with a digit needs real notation in it, or two prices on a line become a formula and the `$`
    signs the model wrote are deleted."""
    (html,) = _md(src)
    assert "<math" not in html, html
    assert html.count("$") == src.count("$"), html


def test_single_dollar_math_still_renders():
    (html,) = _md(r"the timescale $\tau$ and $ \frac{dX}{dt} = rX $")
    assert html.count("<math") == 2, html


def test_a_rejected_dollar_does_not_eat_the_next_formula():
    """A rejected match resumes ONE character in, not past the whole match, or the `$` in a price
    swallows the opener of the real formula that follows it."""
    price, code = _md("cost $5 and $$a=b$$ holds", r"the `$` modifier, then $\tau$ and $K_p$")
    assert price.count("<math") == 1 and "$$" not in price, price
    assert code.count("<math") == 2, code
    assert "<code>$</code>" in code, code


def test_math_survives_the_inline_markdown_rules():
    r"""Math is stashed BEFORE the bold/italic/code pass, the way fenced code already is --
    otherwise the italic rule bridges the gap between two starred formulas."""
    (html,) = _md(r"the fit gave \(\sigma^* = 2 \cdot 10^{-3}\) and \(b^*\) exactly")
    assert "<em>" not in html and "<strong>" not in html


def test_math_works_inside_a_list_item():
    (html,) = _md(r"- the means span \(10^{-3}\) to \(7.4\times10^{-1}\)")
    assert "<li>" in html and "<math" in html


def test_display_math_in_a_list_keeps_one_list():
    """An indented display formula belongs to the item above it: closing and reopening the list
    would restart an <ol>'s numbering at 1."""
    (html,) = _md("1. Rate law:\n   \\[ v = \\frac{V S}{K+S} \\]\n2. Inhibition\n3. Final")
    assert html.count("<ol>") == 1, html
    assert '<div class="md-math">' in html, html


def test_a_math_fence_renders_as_a_formula():
    """```math is a formula the model wanted displayed, not source to read."""
    (html,) = _md("```math\n\\frac{\\text{attack} \\cdot X}{1 + h}\n```")
    assert "<mfrac>" in html and "<pre" not in html


# -- the shapes a modeller actually writes --------------------------------
def test_a_braceless_argument_takes_one_character():
    r"""TeX reads \tfrac12 as one half. Taking the whole number token instead renders a different
    formula with no fallback marker -- the one failure mode the reader cannot see."""
    (html,) = _md(r"\(\tfrac12\log(252)\approx2.8\)")
    assert "<mfrac><mn>1</mn><mn>2</mn></mfrac>" in html, html
    (html,) = _md(r"\(\frac{12}{34}\)")
    assert "<mn>12</mn>" in html and "<mn>34</mn>" in html, html


def test_an_aligned_table_has_one_row_per_row():
    (html,) = _md(r"\[\begin{aligned} a &= 1 \\ b &= 2 \\ \end{aligned}\]")
    assert html.count("<mtr>") == 2, html  # a trailing `\\` opens no empty row
    (html,) = _md(r"\[\begin{aligned} a = 1 \\ b &= 2 \end{aligned}\]")
    assert 'md-math-r"' not in html.split("</mtr>")[0], (
        html
    )  # a row with no `&` is not pulled right


def test_a_function_name_keeps_its_space():
    r"""\log f must not set as "logf"."""
    (html,) = _md(r"\[ \frac{d \log f}{d \log p} = 1 \]")
    assert "<mi>log</mi><mspace" in html, html
    (html,) = _md(r"\[ \max_{\beta>0} x \]")
    assert "<munder><mi>max</mi>" in html, html  # ... and \max still stacks its limit


# -- the markdown that already worked keeps working ------------------------
def test_plain_markdown_is_unchanged():
    """A regression net for the slice above: the math pass must not disturb the subset #35 added."""
    got = _md(
        "**bold** and `code`",
        "- one\n- two",
        "# Header",
        "```python\nx = {'a': 1}\n```",
        "> a quote\n",
        "text with @@0@@ literal",
    )
    assert got == [
        "<p><strong>bold</strong> and <code>code</code></p>",
        "<ul>\n<li>one</li>\n<li>two</li>\n</ul>",
        '<h1 class="md-h">Header</h1>',
        "<pre class=\"md-pre\"><code>x = {'a': 1}</code></pre>",
        "<blockquote>a quote</blockquote>",
        "<p>text with @@0@@ literal</p>",
    ]


def test_snake_case_never_becomes_italics():
    """The #35 rule, still load-bearing: `S_init` is an identifier, not emphasis."""
    (html,) = _md("set S_init and A_init from the data")
    assert "<em>" not in html


# -- review follow-ups (#143) ----------------------------------------------
def test_limits_keeps_its_operator_as_the_base():
    r"""\limits retargets the scripts that follow it, so it is not an atom.

    Dropping it as a no-op left "" on the stack for the next script to take as its base, which
    emitted an <msubsup> with no base at all and orphaned the operator it was meant to stack --
    the one command whose entire purpose is stacking, defeating the stacking.
    """
    (html,) = _md(r"\[ \sum\limits_{i=1}^{N} x_i \]")
    assert "<munderover><mo>∑</mo>" in html, html
    assert "<munderover><mrow></mrow>" not in html, html
    # \limits forces stacking even inline, where a bare \sum_ would set its scripts to the side
    (html,) = _md(r"\(\sum\limits_{i=1}^{N} x_i\)")
    assert "<munderover><mo>∑</mo>" in html, html
    (html,) = _md(r"\(\sum_{i=1}^{N} x_i\)")
    assert "<msubsup><mo>∑</mo>" in html, html


def test_a_script_base_is_exactly_one_element():
    r"""A function name pulls a thin space in for its operand; a script is not an operand.

    ``\lim\limits_{x \to 0}`` would otherwise hand <munder> three children (<mi>, <mspace>,
    <mrow>) -- malformed MathML that renders as a stray gap before the limit.
    """
    (html,) = _md(r"\(\lim\limits_{x \to 0} f\)")
    assert "<munder><mi>lim</mi><mrow>" in html, html
    assert "<mi>lim</mi><mspace" not in html, html


@pytest.mark.parametrize(
    "prose",
    [
        "export $SLURM_JOB_ID and $HOME are set",
        "we sweep $k_deg and $k_syn jointly",
        "budget $N (compute) and $M (storage)",
        "the $x/y ratio for the run",
    ],
)
def test_single_dollar_does_not_eat_prose(prose):
    """Two shell/parameter names either side of a sentence are not a formula.

    `_` satisfied the old character test, which is exactly the character this file refuses to
    treat as markup everywhere else -- snake_case ids are everywhere in this domain.
    """
    (html,) = _md(prose)
    assert "<math" not in html, html
    assert html.count("$") == prose.count("$"), html  # neither delimiter is swallowed


@pytest.mark.parametrize(
    "formula",
    [
        r"$V_{\max}$",
        "$ n > 2 $",
        "$t = 0.5$",
        r"$ V_{max}S/(K_m + S + S^2/K_i) $",
        r"$\tau \to 0$",
        r"$v = V_{\max} \frac{S}{K_m + S}$",
    ],
)
def test_real_single_dollar_formulas_still_render(formula):
    """The prose guard must keep every single-$ form the traces actually contain."""
    (html,) = _md(formula)
    assert "<math" in html and "md-math-raw" not in html, html


def test_inline_code_keeps_latex_as_source():
    """LaTeX the model quoted AS SOURCE stays source, and MathML never lands inside <code>."""
    got = _md(r"use `\(x^2\)` to write a square", r"the antimony `$K_p$` id")
    for html in got:
        assert "<math" not in html, html
        assert "<code>" in html, html
    assert r"\(x^2\)" in got[0], got[0]
    assert "$K_p$" in got[1], got[1]
    # a quoted id next to a real formula: the code span stays literal, the formula still renders
    (html,) = _md(r"`k_deg` is fixed, and \(x^2\) is not")
    assert "<code>k_deg</code>" in html and "<math" in html, html


def test_a_nested_environment_keeps_its_own_rows():
    r"""mathSplit counts \begin/\end, not only braces.

    A \begin{cases} inside \begin{aligned} otherwise had its `\\` rows split as the OUTER table's,
    shredding the inner environment into raw <mtext> with "cases" set as italic math letters.
    """
    (html,) = _md(
        r"\[\begin{aligned} x &= \begin{cases} a \\ b \end{cases} \\ y &= 2 \end{aligned}\]"
    )
    assert html.count("<mtable") == 2, html  # outer aligned + inner cases
    assert "md-math-raw" not in html, html
    assert "<mi>c</mi><mi>a</mi><mi>s</mi>" not in html, html


def test_an_unbalanced_brace_shows_source_instead_of_one_cell():
    r"""A negative depth used to silence every later split, collapsing the body into one cell.

    The `&` alignment marks then rendered as literal operators. Malformed input, so the whole
    formula falls back to visible source.
    """
    (html,) = _md(r"\[\begin{aligned} f(x} &= 1 \\ g &= 2 \end{aligned}\]")
    assert "md-math-raw" in html, html
    assert "<mo>&amp;</mo>" not in html, html


def test_an_optional_argument_may_follow_a_space():
    r"""TeX accepts `\sqrt [3]{V}`; both sibling argument readers already skipped spaces."""
    (html,) = _md(r"\(\sqrt [3]{V}\)")
    assert "<mroot>" in html, html
    assert "<msqrt><mo>[" not in html, html
    # ... while a bracket that follows no command at all is still a concentration
    (html,) = _md(r"\(k [S] + 1\)")
    assert "<mo>[</mo>" in html, html


@pytest.mark.parametrize(
    "tex,want",
    [
        (r"\(\text{50\% of } x\)", "<mtext>50% of </mtext>"),
        (r"\(\text{a\_b}\)", "<mtext>a_b</mtext>"),
        (r"\(\text{A\&B}\)", "<mtext>A&amp;B</mtext>"),
    ],
)
def test_text_mode_escapes_do_not_leak_their_backslash(tex, want):
    """TeX's own escapes are not part of the text, and a plain <mtext> cannot signal raw source."""
    (html,) = _md(tex)
    assert want in html, html


def test_text_mode_still_keeps_a_trailing_space():
    r"""The reason \text takes its argument verbatim in the first place."""
    (html,) = _md(r"\(\text{state } S\)")
    assert "<mtext>state </mtext>" in html, html


def test_display_math_under_an_open_list_stays_in_the_item():
    """Reviewed and kept: keying this on indentation splits an <ol> and restarts its numbering.

    A line straight after an item with no blank line is that item's lazy continuation; a formula
    that really is top-level carries a blank line, which closes the list before this branch runs.
    """
    (html,) = _md("1. first\n\\[ x = 1 \\]\n2. second")
    assert html.count("<ol>") == 1, html
    (html,) = _md("1. first\n\n\\[ x = 1 \\]\n\n2. second")
    assert '<div class="md-math"' in html and "<li>first</li>" in html, html


# -- GFM pipe tables -------------------------------------------------------
def test_a_pipe_table_becomes_a_table():
    """The analysts tabulate constantly; the block loop used to join every row into one paragraph."""
    (html,) = _md(
        "| Observable | n | Measured range |\n|---|---:|---:|\n"
        "| `0ac` | 18 | 0.687–0.786 |\n| `4ac` | 18 | 0.00141–0.00688 |"
    )
    assert html.count("<tr>") == 3, html  # header + two body rows
    assert "<th>Observable</th>" in html, html
    assert '<th class="md-t-r">n</th>' in html, html  # alignment from the delimiter row
    assert "<td><code>0ac</code></td>" in html, html  # cells still run the inline rules
    assert "0.687–0.786" in html, html
    assert "|---|" not in html, html  # the delimiter row is consumed


def test_a_table_scrolls_only_inside_its_own_wrapper():
    """.turn is overflow:hidden, so a table too wide to fold needs the same box a formula gets."""
    (html,) = _md("| a | b |\n|---|---|\n| 1 | 2 |")
    assert html.startswith('<div class="md-table-wrap">'), html


def test_a_rule_under_a_line_with_a_pipe_is_not_a_table():
    """The cell COUNT has to match, or `a | b` over a bare `---` becomes a one-column table."""
    (html,) = _md("a | b\n---")
    assert "<table" not in html, html
    assert "<hr>" in html, html


def test_table_cells_carry_math_and_alignment():
    (html,) = _md(r"| a | b |" + "\n|:--|--:|\n" + r"| \(x^2\) | 2 |")
    assert '<td class="md-t-l"><math>' in html, html
    assert '<td class="md-t-r">2</td>' in html, html


def test_a_pipe_inside_a_code_span_does_not_split_a_cell():
    """Code spans are stashed before the table is parsed, so an Antimony `a|b` stays one cell."""
    (html,) = _md("| id | expr |\n|---|---|\n| r1 | `x|y` |")
    assert html.count("<td>") == 2, html


def test_a_short_row_is_padded_to_the_header():
    (html,) = _md("| a | b | c |\n|---|---|---|\n| 1 |")
    assert html.count("<td>") == 3, html
