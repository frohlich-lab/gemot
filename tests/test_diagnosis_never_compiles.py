"""THE CONTRACT: no diagnosis path may trigger AMICI codegen or a C++ build.

A diagnosis describes a fit that already happened, and the module that fit was compiled with
already exists — so asking "which parameters can this fit not pin?" must never cost a build. It did:
the de-dup no-op re-authors the working SBML into a hash nothing is built for, and a diagnosis that
resolved its AMICI output folder from live session state then compiled that never-fitted structure.
Measured on the reviewed candidate: 22.4 s (blasi no-op), 57 s (toy, auto_fit off) and 12.5 s
(unauthored blasi), charged to no clock at all when it went through the finish gate's un-timed probe.

The fix is :class:`petab_mcp.fitting.FitRecord` — a diagnosis reads the built problem the fit left
behind, so there is no hash to resolve. This file is what keeps that true: it enumerates EVERY
diagnosis entry point and asserts, against a build root, that the directory listing and the compile
accounting are both unchanged. A new entry point belongs in ``_ENTRY_POINTS`` — a reader adding one
gets the check for free, and the loud failure here is a listing diff, not an exception (a diagnosis
itself must fail OPEN for the agent: the section absent with an honest reason).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fastmcp import FastMCP

from petab_mcp import identifiability as I
from petab_mcp import operations as O
from petab_mcp import server as S
from petab_mcp.engine import AmiciEngine
from petab_mcp.references import StubReferenceResolver
from petab_mcp.session import PetabSession

YAML_PATH = Path(__file__).parent / "data" / "toy_v1" / "problem.yaml"

# The toy model, re-authorable with a different STARTING VALUE for k1. That is the drift that makes
# the de-dup no-op dangerous: `structure_fingerprint` ignores nominal values (so the fit is skipped
# as identical) while the SBML TEXT — and therefore the AMICI build key — is new.
_TOY = """model conversion_reaction
  compartment compartment_ = 1;
  species A in compartment_;
  species B in compartment_;
  A = A_init;
  B = B_init;
  A_init = 1.0;
  B_init = 0.0;
  k1 = {k1};
  k2 = 0.1;
  r1: A => B; k1 * A;
  r2: B => A; k2 * B;
end
"""


def _listing(root: Path) -> set[str]:
    """Every path under the build root. A build writes a `pypesto_<hash>` tree, so a set of
    relative paths catches codegen even when it lands inside an existing directory."""
    return {str(p.relative_to(root)) for p in root.rglob("*")}


class _Fixture:
    """One session, its build root, and the REAL fit tools registered over it.

    The tools are registered from ``server``'s own registrar rather than through
    ``create_server`` so the direct calls and the tool calls see ONE session — the de-dup no-op
    state only exists in memory (the working model differs from HEAD), so it cannot be reproduced
    by a second session reading the same provenance directory.
    """

    def __init__(self, tmp_path, *, auto_fit=True, model_less=False):
        self.tmp_path = Path(tmp_path)
        self.auto_fit, self.model_less = auto_fit, model_less
        self.root = self.tmp_path / "build"
        self.root.mkdir(exist_ok=True)
        yaml_path = _model_less_problem(self.tmp_path) if model_less else YAML_PATH
        self.session = PetabSession(
            str(yaml_path),
            provenance_dir=str(self.tmp_path / "prov"),
            resolver=StubReferenceResolver(),
        )
        self.session.engine = AmiciEngine(self.session, build_root=str(self.root))
        self.session.auto_fit = auto_fit
        self.session.auto_fit_starts = 2
        self.mcp = FastMCP("petab-mcp-contract-test")
        S._register_fit_tools(self.mcp, S._Ctx(self.session, self.session.engine))

    def reopen(self):
        """A DURABLE RESUME: a second session over the same provenance dir and the same build root,
        as a restarted process gets. It restores HEAD but holds no fit record, and its working model's
        SBML is serialised in a different DECLARATION ORDER than the fitted one — so it hashes to a
        module nothing has built, and a read that re-derives its build target from live state compiles
        the whole structure again (measured on the pre-fix code: 19.4 s, 122 new files under the build
        root, charged to no clock at all)."""
        return _Fixture(self.tmp_path, auto_fit=self.auto_fit, model_less=self.model_less)

    def call(self, name, **args):
        return asyncio.run(self.mcp.call_tool(name, args)).structured_content

    def state(self):
        return (
            _listing(self.root),
            self.session.n_compiles,
            self.session.compile_seconds,
            set(self.session.engine._pypesto_hashes),
        )


def _model_less_problem(work: Path) -> Path:
    """A copy of the toy PEtab problem shipping an EMPTY model, so nothing is ever compiled."""
    import shutil

    import libsbml

    d = work / "empty"
    d.mkdir()
    for name in (
        "problem.yaml",
        "conditions.tsv",
        "measurements.tsv",
        "observables.tsv",
        "parameters.tsv",
    ):
        shutil.copy(YAML_PATH.parent / name, d / name)
    doc = libsbml.SBMLDocument(3, 2)
    model = doc.createModel()
    model.setId("conversion_reaction")
    model.setMetaId("conversion_reaction")
    libsbml.writeSBMLToFile(doc, str(d / "conversion_reaction.xml"))
    return d / "problem.yaml"


async def _probe_diagnosis(fx):
    """The finish gate's own read, through the harness helper it uses — the path where a compile was
    charged to NO clock, since the driver's probe is deliberately un-timed."""
    from gemot.agent.probe import _current_diagnosis

    async def reader(name, args):
        return await fx.mcp.call_tool(name, args)

    return await _current_diagnosis(reader)


#: Every way a diagnosis can be asked for. Named so a failure says WHICH surface built something.
_ENTRY_POINTS = {
    "merge_hints": lambda fx: I.fit_merge_hints(fx.session),
    "fit_diagnosis_block": lambda fx: I.fit_diagnosis_block(fx.session),
    "stored_merge_hints": lambda fx: I.stored_merge_hints(fx.session),
    "identifiability": lambda fx: I.identifiability(fx.session),
    "get_fit_summary": lambda fx: fx.call("get_fit_summary"),
    "diagnosis_default": lambda fx: fx.call("get_fit_diagnosis"),
    "diagnosis_residuals": lambda fx: fx.call("get_fit_diagnosis", sections="residuals"),
    "diagnosis_parameters": lambda fx: fx.call("get_fit_diagnosis", sections="parameters"),
    "diagnosis_all": lambda fx: fx.call("get_fit_diagnosis", sections="all"),
    # the judge's literal call (runner/judge.py) and the finish gate's un-timed probe
    "judge": lambda fx: fx.call("get_fit_diagnosis", sections="residuals"),
    "finish_gate_probe": lambda fx: asyncio.run(_probe_diagnosis(fx)),
}


def _assert_nothing_built(fx, label):
    # The baseline is taken BEFORE the first read, and that is the whole point: the FIRST read is the
    # only one that can build, because it is the one that finds no stored simulation and materialises
    # one. Warming `get_fit_summary` first — on the theory that an on-demand simulation is a
    # legitimate, accounted build — moved the baseline past the 16 s this file exists to forbid, and
    # made every state below pass vacuously. A read never compiles: `fitting.ensure_simulation`
    # refuses a structure with no module and the section reports why (see the fail-open test below).
    before = fx.state()
    for name, entry in _ENTRY_POINTS.items():
        entry(fx)
        after = fx.state()
        assert after[0] == before[0], (
            f"{label}/{name} wrote to the build root: "
            f"{sorted(after[0] - before[0])[:5]} — a diagnosis compiled a model"
        )
        assert after[1:] == before[1:], (
            f"{label}/{name} moved the compile accounting {before[1:]} -> {after[1:]}"
        )


_CASES = [
    "fitted_head",
    "dedup_noop",
    "dedup_noop_unsimulated",
    "resumed",
    "auto_fit_off",
    "unauthored",
]


@pytest.mark.parametrize("case", _CASES)
def test_no_diagnosis_entry_point_builds_a_model(tmp_path, case):
    """Every entry point, in every state a run reaches, against the build root.

    Three of these built. ``dedup_noop``: HEAD is a real fit, the working SBML has been re-authored to
    the same structure with a different starting value, so the fit is SKIPPED while
    ``session.sbml_hash()`` names a module nobody built — a diagnosis that resolved its own build
    target compiled it there (22.4 s on blasi). ``dedup_noop_unsimulated`` is the same state without a
    stored simulation, where the RESIDUAL half materialises one: 19.4 s, of a structure never fitted.
    ``resumed`` is a restarted process, whose restored model serialises in a different declaration
    order and so hashes to a module nothing built: another 19.4 s, charged to no clock.
    """
    fx = _Fixture(tmp_path, auto_fit=(case != "auto_fit_off"), model_less=(case == "unauthored"))
    if case == "unauthored":
        # Nothing authored, nothing fitted: the state a run is in before its first make_changes.
        assert fx.session.model_authored() is False
    else:
        O.set_model(fx.session, _TOY.format(k1=0.5), justification="build the toy")
        fx.session.fit()
    if case.startswith("dedup_noop"):
        O.set_model(fx.session, _TOY.format(k1=0.7), justification="same structure, new guess")
        noop = fx.session.fit()
        assert noop["no_change"] is True, "the de-dup skip did not fire — this case is vacuous"
    if case == "dedup_noop_unsimulated":
        # The records a read would otherwise find: a process that has just restarted has none.
        fx.session.simulation = None
    if case == "resumed":
        fx = fx.reopen()
        assert fx.session.fit_record is None, "the resume kept a fit record — this case is vacuous"

    _assert_nothing_built(fx, case)
    if case != "fitted_head":
        # What makes these the sharp cases, asserted AFTER the reads so it is the reads that are shown
        # not to have built it: no module exists for the structure the working state names.
        assert not fx.session.engine.module_built(fx.session.sbml_hash())


def test_the_diagnosis_still_answers_where_it_refuses_to_build(tmp_path):
    """Failing open is the other half of the contract: the section must be ABSENT with an honest
    reason, never an exception and never a silent empty payload that reads as "nothing wrong"."""
    fx = _Fixture(tmp_path, auto_fit=False)
    O.set_model(fx.session, _TOY.format(k1=0.5), justification="build the toy")
    fx.session.fit()  # auto_fit off -> status "skipped", so no optimum was frozen

    ident = fx.call("get_fit_diagnosis", sections="parameters")["identifiability"]
    assert ident["available"] is False
    assert ident["note"] == I.NO_OPTIMUM_NOTE
    # ...and the residual half, which has no simulation and may not compile one to get it, says so
    # rather than returning bare empty rows that read as a clean fit
    residuals = fx.call("get_fit_diagnosis")
    assert residuals["per_observable"] == [] and "compile" in residuals["status"]


def test_a_resumed_process_still_reports_the_fits_parameter_geometry(tmp_path):
    """Failing open must not become failing SILENT. The freeze made the parameter half depend on
    in-process state, so a restarted process reported `available: false` for a fit that HAD happened —
    and the harness parsimony gate, which refuses a finish on a rank-deficient FIM, stopped refusing
    for the rest of the run. The verdict therefore travels with the fit
    (identifiability.fit_diagnosis_block), and is re-served without recomputing anything."""
    fx = _Fixture(tmp_path)
    O.set_model(fx.session, _TOY.format(k1=0.5), justification="build the toy")
    fx.session.fit()
    fitted = fx.call("get_fit_diagnosis", sections="parameters")["identifiability"]
    assert fitted["available"] is True and fitted["fim"]

    resumed = fx.reopen()
    after = resumed.call("get_fit_diagnosis", sections="parameters")["identifiability"]
    assert after["fim"] == fitted["fim"] and after["parameters"] == fitted["parameters"]
    assert "stale" not in after
    # ...and the score read beside it names the same tie (or the same absence of one)
    assert resumed.call("get_fit_summary").get("merge_hints") == fx.call("get_fit_summary").get(
        "merge_hints"
    )
