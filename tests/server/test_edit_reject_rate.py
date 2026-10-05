"""The rejection reasons ``result.json`` buckets, held to the server that writes them.

``runner.clock`` buckets a refused edit by matching the server's own rejection prose, so each
bucket is pinned against the installed server's source, and the server's rejection summary (what
the agent and the buckets both read) is pinned here too. The counting itself is in
``tests/test_edit_reject_rate.py``.
"""

from __future__ import annotations


def test_every_reject_bucket_matches_a_message_the_server_can_emit(server_src):
    """Buckets are matched against English prose, so a reworded message turns a live bucket into a
    dead one with nothing failing. Two buckets were already dead on arrival: they were mined from
    traces predating #118, which turned the sigma cap into a warning. Pin them against the real
    strings in the server source instead of literals typed into this file."""
    from gemot.runner.clock import _REJECT_PATTERNS

    src = "\n".join(p.read_text() for p in server_src.rglob("*.py")).lower()
    dead = [label for label, needle in _REJECT_PATTERNS if needle not in src]
    assert not dead, f"reject buckets matching no message the server emits: {dead}"


# -- the rejection message must carry its own reasons (Aug 8-10 sweep) -----------------------
def test_rejection_summary_names_the_reasons():
    from petab_mcp.operations import _rejection_summary

    out = _rejection_summary(
        [
            "'RAF_total' ships FIXED (estimate=0) in this problem — a known constant",
            "Extraneous parameter(s) in parameter table: {'pEGFR_Akt', 'pAkt_S6'}",
        ]
    )
    assert "resulting problem is invalid" in out  # bucket substring preserved
    assert "RAF_total" in out and "pAkt_S6" in out  # the actionable detail is IN the message
    assert "more in `issues`" not in out  # nothing elided at 2 issues


def test_rejection_summary_caps_and_says_what_it_elided():
    from petab_mcp.operations import _rejection_summary

    out = _rejection_summary([f"issue number {i}" for i in range(7)])
    assert "issue number 0" in out and "issue number 2" in out
    assert "issue number 3" not in out  # capped at 3
    assert "(+4 more in `issues`)" in out  # a truncated view never looks complete


def test_rejection_summary_collapses_multiline_libsbml_noise():
    from petab_mcp.operations import _rejection_summary

    long_sbml = (
        "libSBML Error (SBML component consistency): The value of a 'symbol' field in any "
        "<initialAssignment>\n   definition cannot also appear as the value of a "
        "'variable' field.\n   " + "x" * 400
    )
    out = _rejection_summary([long_sbml])
    assert "\n" not in out  # one line, not a wall of text
    assert len(out) < 420  # per-issue truncation actually applied


def test_rejection_summary_falls_back_when_there_is_no_error_level_issue():
    from petab_mcp.operations import _rejection_summary

    assert _rejection_summary([]) == "the resulting problem is invalid."
