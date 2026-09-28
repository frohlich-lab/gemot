"""Authoring tool: prune terms a published problem gates with a parameter FIXED AT EXACTLY 0 OR 1.

This is not used at run time — it is how an example's ``TRUE_ANT`` was derived, kept in the repo so
the transformation is reproducible and auditable rather than a hand-edit nobody can re-check.

**Why.** ``materialize`` ships a SUBSET parameter table, and a published-FIXED value is shipped only
when it is genuine experimental setup (see the module docstring of :mod:`benchmark`). A parameter
the published problem fixes at exactly ``0`` or ``1`` is not setup — it is a model-selection FINDING.
``0`` records that a link is ABSENT; ``1`` records that two quantities were tested and found
EQUIVALENT (isensee's ``xi_i_*``: prodrug and free analogue are equipotent). Both are conclusions a
zero-shot build is meant to reach, so neither may reach the agent.
But merely leaving it out of the shipped table is not enough: the example's ``TRUE_ANT`` still names
the symbol, the server's ``reconcile`` mints it as an ESTIMATED parameter, and the candidate's ``k``
overshoots the reference's by one ``ln(n)`` BIC penalty apiece.

**What this does.** Substitutes the published value symbolically, drops any reaction whose rate law
thereby becomes identically zero, simplifies the rate laws / rules / initial assignments that merely
lose a term, and removes the now-unreferenced parameters. Only ``0`` and ``1`` are accepted, and for
the same reason the server fixes an initial condition of exactly 0 or 1 by value convention (ICs
pattern 4): they are the annihilator and the identity, so the SYMBOL disappears and no magic number
is left behind in its place. Substituting an exact value is numerically EXACT, so the pruned model
reproduces the published objective at the published optimum bit-for-bit: it *is* the effective
published model — what the published problem actually simulates.
The withheld symbols then never appear, reconcile never creates them, and ``k`` stays at parity with
no change to the reference. (A published constant railed onto a NON-zero bound cannot be pruned this
way, because dropping it would change the numbers; that case is handled instead by
``write_reference_for(free_parameters=...)``, which frees it on both sides — see ``alkan``.)

Used to derive ``TRUE_ANT`` for ``isensee`` (6 zeros + 4 unit prodrug ratios), ``lucarelli`` (33
zeros -> 10 reactions, 12 rate laws) and ``raimundez`` (7 zeros, all in assignment rules). Always
re-verify the result against the pristine problem's objective after pruning.

    python -m petab_harness.prune_fixed_terms <BenchmarkName> <out.ant> <param> [<param> ...]
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path
from typing import Any


def _symbol_locals(model) -> dict:
    """Map every id the model defines to a plain sympy Symbol.

    ``sympify`` without this parses against sympy's own namespace, where a number of perfectly
    ordinary SBML ids are NOT free symbols: ``E`` is Euler's number, ``I`` the imaginary unit,
    ``pi``/``oo``/``nan`` constants, and ``N``/``S``/``O``/``Q``/``beta``/``gamma``/``zeta``
    functions or registries. ``sympify("E^2*k")`` silently returns ``k*exp(2)`` — the species
    vanishes into a literal and the pruned model is wrong with nothing to notice it. Binding every
    model id to a Symbol first makes the parse faithful.

    Bind only ids the model DECLARES, never every token in a formula: harvesting formula text would
    also capture the call names (``exp``, ``log``, ``piecewise``) and shadow sympy's own functions,
    so ``ka * exp(x)`` would fail to parse with "'Symbol' object is not callable".
    """
    import sympy as sp

    ids: set[str] = set()
    for getter, count in (
        (model.getSpecies, model.getNumSpecies()),
        (model.getParameter, model.getNumParameters()),
        (model.getCompartment, model.getNumCompartments()),
        (model.getReaction, model.getNumReactions()),
    ):
        ids |= {getter(i).getId() for i in range(count)}
    ids |= {model.getRule(i).getVariable() for i in range(model.getNumRules())}
    # Kinetic-law LOCAL parameters are declared inside the reaction, not on the model, so they are
    # invisible to getParameter() — and a local named E/I/N/S/Q/beta/gamma would collide exactly
    # like a global one.
    for i in range(model.getNumReactions()):
        kl = model.getReaction(i).getKineticLaw()
        if kl is not None:
            ids |= {kl.getLocalParameter(j).getId() for j in range(kl.getNumLocalParameters())}
    # `time` is SBML's reserved symbol rather than a declared id, but must still parse as a symbol.
    ids.add("time")
    # A user-defined FUNCTION is a call name, so it must NOT be bound to a Symbol (that would make
    # it non-callable); drop any id that names one.
    functions = {
        model.getFunctionDefinition(i).getId() for i in range(model.getNumFunctionDefinitions())
    }
    ids -= functions
    return {name: sp.Symbol(name) for name in ids if name and name.isidentifier()}


def _to_sympy(formula: str, zeros: dict[str, int], symbols: dict):
    import sympy as sp

    expr = sp.sympify(formula.replace("^", "**"), locals=symbols)
    return sp.simplify(expr.subs({sp.Symbol(k): v for k, v in zeros.items()}))


def _to_l3(expr) -> str:
    """sympy's ``**`` back to SBML L3's ``^``."""
    return str(expr).replace("**", "^")


def _parse_l3(formula: str, where: str, known: set[str] | None = None):
    """``parseL3Formula`` that refuses to fail silently.

    libsbml returns None for anything L3 cannot parse — notably sympy's own printing of a native
    Piecewise, ``Piecewise((0, x < 0), (1, True))`` — and ``setMath(None)`` raises nothing, so the
    reaction or rule would end up with NO math and the derived TRUE_ANT would be quietly wrong.
    For a tool whose entire purpose is an auditable derivation, that is the worst failure mode.
    """
    import libsbml

    # sympy prints its non-finite constants as bare words — `zoo` (complex infinity, from a
    # division whose denominator substituted to 0), `oo`, `nan` — and libsbml parses each as a
    # perfectly valid AST_NAME, so a None check alone would let a reference to an undeclared
    # symbol through as the reaction's rate law. Reject them by name before parsing.
    # Scope the check to names the MODEL does not declare: a model may legitimately have a
    # parameter called `E` or `nan`-like id, and `_symbol_locals` binds those, so sympy printing
    # them back is correct. Only an undeclared `zoo`/`oo`/`nan`/`inf` means the substitution went
    # non-finite (e.g. a denominator that substituted to 0).
    bad = ({"zoo", "oo", "nan", "inf"} - set(known or ())) & set(
        re.findall(r"[A-Za-z_]\w*", formula)
    )
    if bad:
        raise ValueError(
            f"{where}: simplification produced sympy constant(s) {sorted(bad)} in {formula!r} — "
            "the substitution made the expression non-finite (or a symbol was parsed as a sympy "
            "constant). Refusing to write it as a rate law."
        )
    ast = libsbml.parseL3Formula(formula)
    if ast is None:
        raise ValueError(
            f"{where}: libsbml could not parse the simplified formula {formula!r} "
            f"({libsbml.getLastParseL3Error()}). Refusing to write empty math."
        )
    return ast


def prune(name: str, fixed_params: list[str], collection: Path | None = None) -> tuple[str, dict]:
    """Return ``(antimony, report)`` for the published model with ``fixed_params`` substituted.

    Raises unless every named parameter is fixed at exactly 0 or 1 in the published problem: any
    other value would either change the model or leave a hard-coded literal behind, and both are
    things this must never do.
    """
    import libsbml
    import sympy as sp
    from petab.v1.C import NOMINAL_VALUE

    from petab_harness.benchmark import _load_problem

    prob = _load_problem(name, collection)
    doc = libsbml.readSBMLFromString(libsbml.writeSBMLToString(prob.sbml_document))
    model = doc.getModel()
    pdf = prob.parameter_df

    zeros: dict[str, int] = {}
    for p in fixed_params:
        sbml_p = model.getParameter(p)
        if sbml_p is None:
            raise KeyError(f"{name}: {p} is not an SBML parameter")
        value = float(pdf.loc[p, NOMINAL_VALUE]) if p in pdf.index else float(sbml_p.getValue())
        if value not in (0.0, 1.0):
            raise ValueError(
                f"{name}: {p} is {value}, not exactly 0 or 1 — substituting it would leave a "
                "hard-coded literal in the rate law (or change the model). A bound-railed constant "
                "at any other value belongs in write_reference_for(free_parameters=)."
            )
        zeros[p] = int(value)
    symbols = _symbol_locals(model)

    # Seed the surviving parameters at the published optimum, as the TRUE_ANT round-trip does.
    for i in range(model.getNumParameters()):
        pid = model.getParameter(i).getId()
        if pid in pdf.index:
            try:
                value = float(pdf.loc[pid, NOMINAL_VALUE])
            except (TypeError, ValueError):
                continue
            # An absent nominalValue arrives as NaN, which float() accepts happily — the guard above
            # only catches the cases that cannot occur. Leave the SBML value in place instead of
            # writing `param = nan` into the emitted Antimony.
            if math.isfinite(value):
                model.getParameter(i).setValue(value)

    removed_reactions: list[str] = []
    simplified: list[tuple[str, str, str]] = []
    for i in reversed(range(model.getNumReactions())):
        rxn = model.getReaction(i)
        kl = rxn.getKineticLaw()
        if kl is None:
            continue
        before = libsbml.formulaToL3String(kl.getMath())
        expr = _to_sympy(before, zeros, symbols)
        if expr == 0:
            removed_reactions.append(rxn.getId())
            model.removeReaction(i)
        elif sp.simplify(sp.sympify(before.replace("^", "**"), locals=symbols) - expr) != 0:
            after = _to_l3(expr)
            simplified.append((rxn.getId(), before, after))
            kl.setMath(_parse_l3(after, f"reaction {rxn.getId()}", set(symbols)))

    for getter, count, kind in (
        (model.getRule, model.getNumRules(), "rule"),
        (model.getInitialAssignment, model.getNumInitialAssignments(), "initialAssignment"),
    ):
        for i in range(count):
            node = getter(i)
            before = libsbml.formulaToL3String(node.getMath())
            if any(re.search(rf"\b{p}\b", before) for p in fixed_params):
                target = node.getVariable() if kind == "rule" else node.getSymbol()
                node.setMath(
                    _parse_l3(
                        _to_l3(_to_sympy(before, zeros, symbols)), f"{kind} {target}", set(symbols)
                    )
                )
                simplified.append(
                    (f"{kind} {target}", before, libsbml.formulaToL3String(node.getMath()))
                )

    # Drop the zeroed parameters once nothing references them. Orphan SPECIES are left alone:
    # the server's reconcile.prune_orphans removes those (and their IC parameters) on set_model.
    def referenced() -> set[str]:
        """Every id the model still refers to.

        Walks the AST via petab_mcp's own ``_ast_names`` rather than regexing the printed formula:
        a regex over the text also matches CALL names (``exp``, ``log``, ``piecewise``), so it
        misclassifies in both directions. Covers reactions, rules (math AND target), initial
        assignments, events, and reaction participants — a parameter referenced only from an event
        assignment would otherwise be deleted and leave the event dangling, the same breakage this
        module's rule-target handling exists to prevent.
        """
        from petab_mcp.reconcile import _ast_names

        out: set[str] = set()
        for i in range(model.getNumReactions()):
            rxn = model.getReaction(i)
            kl = rxn.getKineticLaw()
            if kl is not None:
                out |= _ast_names(kl.getMath())
            for j in range(rxn.getNumReactants()):
                out.add(rxn.getReactant(j).getSpecies())
            for j in range(rxn.getNumProducts()):
                out.add(rxn.getProduct(j).getSpecies())
            for j in range(rxn.getNumModifiers()):
                out.add(rxn.getModifier(j).getSpecies())
        for i in range(model.getNumRules()):
            rule = model.getRule(i)
            out |= _ast_names(rule.getMath())
            out.add(rule.getVariable())
        for i in range(model.getNumInitialAssignments()):
            ia = model.getInitialAssignment(i)
            out |= _ast_names(ia.getMath())
            out.add(ia.getSymbol())
        for i in range(model.getNumEvents()):
            ev = model.getEvent(i)
            if ev.getTrigger() is not None:
                out |= _ast_names(ev.getTrigger().getMath())
            for j in range(ev.getNumEventAssignments()):
                ea = ev.getEventAssignment(j)
                out |= _ast_names(ea.getMath())
                out.add(ea.getVariable())
        return out

    still_used = referenced()
    removed_parameters = [p for p in fixed_params if p not in still_used]
    for p in removed_parameters:
        model.removeParameter(p)

    import antimony

    antimony.clearPreviousLoads()
    if antimony.loadSBMLString(libsbml.writeSBMLToString(doc)) < 0:
        raise RuntimeError(f"antimony could not load the pruned {name}: {antimony.getLastError()}")
    report: dict[str, Any] = {
        "removed_reactions": removed_reactions,
        "simplified": simplified,
        "removed_parameters": removed_parameters,
        "still_referenced": sorted(set(fixed_params) - set(removed_parameters)),
    }
    return antimony.getAntimonyString(None), report


def main(argv: list[str]) -> int:
    # Scoped to the CLI: a module-level filterwarnings() would silence every warning process-wide
    # for anything that merely imports this module (benchmark.py, simulate.py, a test).
    import warnings

    warnings.filterwarnings("ignore")
    if len(argv) < 3:
        print(__doc__)
        return 2
    name, out = argv[0], Path(argv[1])
    ant, report = prune(name, argv[2:])
    out.write_text(ant)
    print(f"[prune] {name} -> {out}")
    print(
        f"[prune]   removed reactions ({len(report['removed_reactions'])}): "
        f"{report['removed_reactions']}"
    )
    print(f"[prune]   simplified ({len(report['simplified'])}):")
    for where, before, after in report["simplified"]:
        print(f"[prune]     {where}: {before[:60]} -> {after[:60]}")
    print(
        f"[prune]   removed parameters ({len(report['removed_parameters'])}): "
        f"{report['removed_parameters']}"
    )
    if report["still_referenced"]:
        print(f"[prune]   STILL REFERENCED (not removed): {report['still_referenced']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
