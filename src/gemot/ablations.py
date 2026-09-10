"""THE registry of ablations: every switch that changes WHAT A RUN DOES, in one place.

An ablation changes the agent's situation — what it is told, what tools it holds, what it may look
up — while leaving the code identical. It is therefore the one class of run input that ``git_sha``
cannot capture, which is what the three rules below exist to handle.

THE MEMBERSHIP TEST, since a switch is easy to add in the wrong place: an arm changes WHAT AN ANSWER
IS, or whether the agent can get one at all. A switch that changes only where an answer comes from,
or how fast it arrives, is transport — it cannot move a result, so registering it would make
:func:`arm` churn on differences that no analysis can care about. Transport still needs recording,
but as PROVENANCE (a source, a fetch date, a row count in ``result.json``), which is a different
job with a different consumer.

**Discoverable.** ``python -m petab_harness.ablations`` prints the catalogue: every arm, its
default, what the non-default side does, and whether it moves ``prompt_version``. A switch parsed
where it is used can only be found by grepping for it, so "which arms can I run?" has no answer
short of reading the whole source.

**Recorded.** :func:`state` and :func:`arm` go into every ``result.json``, and ``ablation_arm``
into the metrics CSV. An unrecorded arm is worse than no arm: both halves of an A/B land in one
table with no column to separate them, and the pooled cell reads as a result. Note that an arm
changing no prompt text leaves ``prompt_version`` byte-identical across the halves, so nothing else
in the record can tell them apart.

**One mechanism.** Every arm is an environment variable. Not a CLI flag: a flag cannot be set from
a batch script, which is where a sweep configures itself, and an arm settable two ways gets
recorded from whichever one the recorder happens to read. Nor a switch parsed at each use site —
two copies of a predicate that must agree can disagree, and a disagreement here is a half-ablated
run, with a tool removed from the toolset while the prompt still instructs the agent to call it.

WHY IT LIVES IN THE HARNESS: an ablation is an experiment concept — the harness runs the arms,
records them and sweeps over them. It also must live here rather than in the server package,
because ``petab_harness.prompts`` has to import without ``petab_mcp`` (see
``tests/test_import_lightness``) and ``slurm/make_sweep.py`` imports from a bare interpreter. The
server consequently reads its one switch from the environment itself; ``test_ablations`` runs both
that gate and this registry over every spelling, so the twice-written rule cannot drift.

ADDING ONE: append an :class:`Ablation` and read it through :func:`enabled`. Recording, the CLI and
the metrics column all follow from this tuple, and ``test_ablations.py`` fails if a ``PETAB_*``
variable gates behaviour anywhere in ``src/`` without being registered here or listed as
non-ablation configuration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# Spellings that mean "off". Everything else means on. See `_switch`.
_OFF = frozenset({"0", "false", "no", "off"})


def _switch(env: str, *, default_on: bool) -> bool:
    """Read one on/off environment variable: off only for an explicit off-word, else on."""
    return os.environ.get(env, "1" if default_on else "0").strip().lower() not in _OFF


@dataclass(frozen=True)
class Ablation:
    """One arm: what it is called, how to set it, and what the non-default side does.

    ``changes_prompt`` is the field to read when planning a comparison. A switch that alters prompt
    TEXT moves ``prompt_version``, so it must not share a sweep with any unrelated prompt edit: the
    two become inseparable in the record, and no later analysis can attribute the difference.
    """

    name: str
    env: str
    default_on: bool
    changes_prompt: bool
    effect: str                        # what the NON-DEFAULT side does, in one line

    @property
    def setting(self) -> str:
        """The literal assignment that flips this arm away from its default."""
        return f"{self.env}={'0' if self.default_on else '1'}"

    def enabled(self) -> bool:
        return _switch(self.env, default_on=self.default_on)

    def is_default(self) -> bool:
        return self.enabled() is self.default_on


ABLATIONS: tuple[Ablation, ...] = (
    # DEFAULT OFF. A 115-run controlled A/B measured this arm at 10/58 vs 30/57 (p=8e-5): it is
    # HARMFUL as built.
    #
    # The harm is REGIME-DEPENDENT, and the pooled number hides that — it is carried almost entirely
    # by sneyd (1/20 vs 15/20; median best bic -522 vs -576). On fujita the arms are
    # indistinguishable (median best bic +141.1 ON vs +143.9 OFF) and blasi was a null result. The
    # difference is whether the problem HAS real structure the tests are too weak to establish:
    # sneyd's dose-response slope is 1.85 (cooperative, and missed), fujita's are 0.58-1.19 (shallow,
    # so a null verdict there is roughly correct and costs nothing). Do not read this arm as
    # uniformly bad, and do not judge a fix on fujita — it cannot discriminate.
    #
    # THE WORDING WAS THE FIRST DIAGNOSIS AND IT WAS WRONG — do not spend a third sweep on it. The
    # payload used to phrase absence of evidence as a finding ("one timescale is sufficient",
    # "consistent with a single binding step"); that was rewritten to report positive requirements
    # only and stay silent otherwise, and a re-run moved sneyd to just 4/10 — better (it broke a
    # hard k<=8 ceiling) but still far short of 15/20 with the arm off.
    #
    # THE REAL MECHANISM, from 40 traced runs: a null verdict is not a neutral one, and what it
    # suppresses is a REPAIR rather than a prior. The feature that decides sneyd is a power in a rate
    # law (80% of best models with the arm OFF vs 35% ON, worth 26-33 bic in BOTH arms) — and the
    # reference model has no exponent at all, being a six-state network with saturating rate laws at
    # k=15. The exponent is the agent's surrogate for states it has not built: it sharpens a
    # too-broad response at k~9 without adding a species. Suppress it and the run keeps a model that
    # is both too small and too flat.
    #
    # The timing is the whole effect. `get_data_regimes` is consulted BEFORE the first model in
    # 10/10 traced runs, so it speaks exactly where the up-front structural bet is made. Runs that
    # author the exponent into their FIRST model: 12/20 with the arm off, 1/20 with it on; runs that
    # never get there at all: 1/20 off, 9/20 on. The rest re-derive it late from residuals, and the
    # arms' bic curves separate from precisely that delay (identical over fits 1-3, apart from 5).
    # Not literature-driven — only 2 of 19 introductions cite curated biology, and both arms cite
    # PMIDs in 20/20 first models, so this is not a biocurator fix. Both quantitative tests return
    # null on every problem we have: `dose_response` needs a log-log slope > 4.5 to fire (sneyd's is
    # 1.85) and `timescales` is untestable or on the wrong side of its BIC bar. See the notes on `z`
    # in `petab_mcp.regimes._dose_response`.
    #
    # STALE BY CONSTRUCTION — every number above was produced by the tool AS IT WAS, and the tool
    # has changed since, WITHOUT a re-run. Nothing here has been re-measured. What moved after the
    # last sweep:
    #   - `_shape` and `_conservation` now fill `requirement`, which they never did. `_shape` is the
    #     only test that fires in practice (90/90 on sneyd, 89 on fujita), so the agent's "act on a
    #     non-null requirement" rule went from matching NOTHING on either problem to matching that.
    #   - `_dose_response`'s `suggestive` tier now has to clear the selection bias instead of being
    #     the bare `slope > 1`. sneyd and fujita pEGFR both drop from `suggestive` to `none`, which
    #     removes the only positive signal the tool gave about the exponent this arm turns on.
    #   - a failed two-exponential fit reports `testable: false` instead of a design verdict, and
    #     the lead prompt / stall advice / independence-audit skill were rewritten (prompt_version
    #     moves, so the halves of any OLD sweep cannot be pooled with a new one).
    #
    # The first two push in OPPOSITE directions and neither was measured, so the current arm's
    # behaviour is genuinely unknown — better and worse are both live. Do NOT read the 10/58 vs
    # 30/57 above as its present strength, and do not compare fresh ON runs against those numbers or
    # against the 4/10 re-run: a future A/B has to measure BOTH arms again on one code version.
    # The arm stays OFF, so none of this is in the default path.
    Ablation(
        name="data_regimes",
        env="PETAB_DATA_REGIMES",
        default_on=False,
        changes_prompt=True,
        effect="ON adds the `get_data_regimes` tool to every agent's toolset AND the "
               "independence-audit pointer to the lead prompt and the server's structural "
               "nudges — they ship and are tested together. Measured HARMFUL as built; see above",
    ),
    # DEFAULT ON since 2026-08-19 — for COST, not capability. Five A/Bs measured no outcome
    # effect and a -28% token saving on alkan (p=0.0065); see the `effect` string below for
    # the numbers and the size dependence. The successor to
    # the retired draft_editing arm, keeping the one component of that design that measured WELL
    # and deleting the rest. What the draft era established across three A/Bs (sneyd/fujita, 57
    # draft-arm runs): (a) search/replace patching itself is reliable — 1.4% mechanical failure
    # over 1,540 patches, ZERO Antimony-semantics failures, 0/84 unauthorized edits; (b) the
    # apparatus around it was the cost — a separate editor subagent, draft files, a lint gate and
    # a delegation protocol ran 2.0-2.5x the LLM calls of baseline with outcomes indistinguishable
    # (and 0/7 vs 3/10 on the last fujita cell); (c) the original motivation had meanwhile
    # expired — 0 of 120 make_changes calls on the lucarelli re-run were statically rejected,
    # vs the 81% static-rejection rate that justified drafts. So: keep the patch semantics, put
    # them ON make_changes (edits= list, all-or-nothing, same scored pipeline; base_version=
    # patches a prior version's stored text = rollback-and-change in one call), and delete the
    # subagent, the files, and the gate.
    #
    # Then search/replace itself was removed too, and the arm now addresses model elements BY ID
    # rather than by quoted text. The reason: text patching requires the agent to reproduce its
    # own prior text exactly, and it does not. Substring edits mismatched their target far more
    # often than wholesale authoring did, which is why the reject rate, not the edit semantics,
    # was the binding constraint. An ID names a reaction whatever its text says.
    Ablation(
        name="diff_edits",
        env="PETAB_DIFF_EDITS",
        default_on=True,
        changes_prompt=True,
        effect="ON registers make_changes WITH `model_edits` (a BATCH of ID-addressed operations "
               "— add_reaction, remove_reaction, set_rate_law, set_initial, set_assignment, "
               "remove_assignment, rename — each an "
               "Antimony statement, applied server-side, all-or-nothing) + `base_version` (edit a "
               "prior version's stored source), adds `get_model` to the lead's toolset, and adds "
               "the edit-by-operation guidance to its prompt. Batched deliberately: one fit per "
               "call, not per edit. MEASURED over five A/Bs (~90 runs, sneyd/fujita/isensee/alkan): "
               "NO outcome effect — sneyd 5/10 vs 6/10, fujita 1/10 vs 3/10, isensee 2/8 vs 1/7, "
               "alkan 1/10 vs 1/10 — but a real COST saving that scales with model size. On alkan "
               "(6v6 with full token accounting) median 7.03M vs 9.77M total tokens (-28%), "
               "completion 110k vs 182k (-40%), 42 vs 48 LLM calls, exact two-sided "
               "Mann-Whitney p = 0.0065. It does NOT generalise down: sneyd and fujita measured at "
               "PARITY, because a wholesale re-author is ~292 tokens on sneyd and ~462 on fujita "
               "but thousands on alkan's 53-reaction models, while a structured batch stays ~80. "
               "Rejections also fell from 74 (substring arm) to 17 across 23 runs, with zero "
               "unknown-reaction-id and zero base_version-alone errors. DEFAULT ON for the cost, "
               "not for capability",
    ),
    # DEFAULT ON since 2026-08-15, when the A/B ran: measured helpful on sneyd (7/10 vs 2/10
    # basin-hit at n=10/cell, p=0.07). Motivated by the sneyd basin post-mortem (50 runs,
    # 32 with recoverable first-commit telemetry): the single separating variable between runs
    # that found the reference structure and runs that plateaued was the estimated-parameter
    # count of the FIRST committed model (k1 >= 11: 12/12 IN vs 2/20 OUT, AUC 0.963, MWU
    # p = 1.4e-5, replicating in both sweeps and all three arms) — and decomposing k1 on the 15
    # readable model bodies, free Hill/gating exponents carry 48% of the gap (IN mean 2.33 vs
    # OUT 0.67). Topology does NOT separate (same 4-state/5-reaction graphs on both sides), and
    # "prefer saturating rate laws" is vacuous (0/15 first commits were pure mass action). The
    # sharpest case: one OUT run wrote the winners' exact bell-shaped algebra with the exponents
    # HARDCODED as the literal 2 (k1=7, dBIC +24.9); an IN run with the same structure and free
    # n_* finished at -105.7. This arm turns that one difference into an instruction. All of the
    # above is observational — whether FORCING free exponents reproduces the association is
    # exactly what the A/B must test (readout: sneyd basin-hit-rate, scored against the
    # capacity-without-structure failure mode: 12 params across 3 reactions, trimmed to 7,
    # finished +31.3).
    Ablation(
        name="free_exponents",
        env="PETAB_FREE_EXPONENTS",
        # ON by default: measured, and the A/B moved both the mechanism and the outcome.
        # 2026-08-15 sneyd 2x2 (jobs 51901211/51901212, 10 runs/cell, sha 8fe8b8a): basin-hit
        # 7/10 vs 2/10 (median dBIC -22.0 vs +15.7), and the FIRST commit moved exactly the
        # targeted component — free exponents 1.4 -> 3.4 per model, hardcoded exponents 0.9 -> 0,
        # reactions/rate constants flat (so it is not "build bigger"), k1 median 10 -> 12.5 with
        # fe's minimum equal to base's maximum.
        #
        # HOW STRONG THIS IS, honestly. The MECHANISM half is what carries the decision: the
        # intervention changed the component it named and nothing else, which is a within-arm
        # fact and does not depend on the rate comparison. The RATE half is weak — 7/10 vs 2/10
        # is Fisher p=0.070 two-sided at n=10/cell, against the n~50 that FINDINGS.md says sneyd
        # bimodality demands, and the control drew 2/10 against its own 30-44% history, so some
        # of the gap is a low control. Only sneyd is measured; the clause is generic but its
        # effect elsewhere is not. Compare data_regimes above, whose verdict rests on n=115.
        #
        # NOT claimed: that "exponent freedom is the carrier and k1 its proxy". The tempting
        # statistic (freeExp>=3 is 9/14 IN vs 0/6 below) pools BOTH arms of an intervention that
        # exists to force freeExp up, so arm assignment drives predictor and outcome together and
        # it cannot separate them. What is fair to say is narrower: the two control-arm winners
        # opened below the k1>=11 threshold while carrying 3 free exponents, which is one
        # observation against k1 being the whole story — a hypothesis for a bigger sweep.
        default_on=True,
        changes_prompt=True,
        # Per the field contract this describes the NON-DEFAULT side (=0), like every other entry.
        effect="OFF drops the lead-prompt clause that tells it to declare every "
               "cooperativity/gating exponent as an estimated parameter WHENEVER the term is "
               "introduced — first model, later edit, or sharpening an existing rate law "
               "(never hardcoding 1 or 2) and trim after the first fit — i.e. the pre-2026-08-15 "
               "behaviour. ON is the default: measured helpful on sneyd, 7/10 vs 2/10 basin-hit "
               "at n=10/cell (p=0.07); see the note above for what that does and does not support",
    ),
    Ablation(
        name="bio_literature",
        env="PETAB_BIO_LITERATURE",
        default_on=False,
        changes_prompt=False,
        effect="ON additionally gives the biocurator EuropePMC + bioRxiv literature search. Off by "
               "default because literature search over a benchmark problem's distinctive entity "
               "panel could surface its withheld source paper; the structured knowledge "
               "(interactions/pathways/ontologies/OmniPath) is general biology. The biology_blind "
               "arm disables both kinds of knowledge access",
    ),
    Ablation(
        name="biology_blind",
        env="PETAB_BIOLOGY_BLIND",
        default_on=False,
        changes_prompt=True,
        effect="ON anonymizes the agent's PEtab identifiers and removes descriptive metadata, "
               "scenario biology, the biocurator and external knowledge tools. The panel receives "
               "numerical modeling instructions; a private mapping preserves reference scoring",
    ),
)

_BY_NAME = {a.name: a for a in ABLATIONS}


def enabled(name: str) -> bool:
    """Whether the named ablation is on. Raises on an unknown name — a typo must not read as off."""
    try:
        return _BY_NAME[name].enabled()
    except KeyError:
        raise KeyError(f"unknown ablation {name!r}; registered: {sorted(_BY_NAME)}") from None


def state() -> dict[str, bool]:
    """Every ablation's resolved value, for ``result.json``.

    Records ALL of them, not only the non-default ones: a run silent about ``bio_literature`` is
    indistinguishable from one recorded before the field existed, and "absent" would then have to
    be read as "probably the default", which is how provenance rots.
    """
    return {a.name: a.enabled() for a in ABLATIONS}


def arm() -> str:
    """A compact canonical name for this configuration — the flat column a sweep groups by.

    ``"baseline"`` when every switch sits at its default, otherwise the deviations joined in
    registry order (``"data_regimes=off"``). Naming only the DEVIATIONS keeps the string stable when
    a new ablation is added at its default, so old runs stay comparable to new ones.
    """
    off = [f"{a.name}={'on' if a.enabled() else 'off'}" for a in ABLATIONS if not a.is_default()]
    return ",".join(off) if off else "baseline"


def state_key() -> str:
    """EVERY switch's resolved value, in registry order — the default-INDEPENDENT column.

    :func:`arm` names only the deviations, which is what you want to group by but is defined
    RELATIVE to the current defaults. So when a default flips, the same string denotes a
    different configuration on either side of that commit: after 2026-08-15 ``"baseline"`` means
    ``free_exponents=on`` where before it meant ``off``, and pooling a sweep that spans the
    change silently averages opposite configurations — precisely the failure the module
    docstring's "Recorded" rule exists to prevent, reintroduced by the default rather than by a
    missing column. This string always says what actually ran, so a groupby on it is safe across
    any default change (and a new arm appended at its default DOES move it, which is the
    trade — group by `arm` for continuity, by this for correctness).
    """
    return ",".join(f"{name}={'on' if on else 'off'}" for name, on in state().items())


def describe(*, use_color: bool = False) -> str:
    """The catalogue printed by ``python -m petab_harness.ablations``."""
    bold, dim, reset = ("\033[1m", "\033[2m", "\033[0m") if use_color else ("", "", "")
    rows = []
    for a in ABLATIONS:
        mark = "" if a.is_default() else "   <- NOT the default"
        prompt = "moves prompt_version" if a.changes_prompt else "prompt unchanged"
        # `setting` is derived per-switch rather than hard-coded to `=0`: the default sides differ,
        # so the value that flips an arm depends on which way it points. A catalogue whose printed
        # instructions silently no-op is worse than none, so `test_ablations.py` executes every one
        # of these strings and asserts the arm actually moves.
        rows.append(
            f"{bold}{a.name}{reset}  [{'on ' if a.enabled() else 'off'}]{mark}\n"
            f"  {dim}set{reset}     {a.setting}   "
            f"(default {'on' if a.default_on else 'off'}; {prompt})\n"
            f"  {dim}does{reset}    {a.effect}\n"
        )
    return (
        f"{bold}Ablations{reset} — switches that change what a run does, without changing the "
        f"code.\nEvery arm is an environment variable; export it before the run.\n"
        f"Current arm: {bold}{arm()}{reset}\n\n" + "\n".join(rows) +
        f"\n{dim}Every run records these in result.json as `ablations`, plus the flat "
        f"`ablation_arm` column in the metrics CSV, so an A/B can always be split after the "
        f"fact.{reset}"
    )


def _main(argv: "list[str] | None" = None) -> int:
    import argparse
    import json
    import sys

    p = argparse.ArgumentParser(
        prog="python -m petab_harness.ablations",
        description="List the ablations available in this repo and their current values.",
    )
    p.add_argument("--json", action="store_true",
                   help="emit the registry as JSON (name, env, default, current, effect)")
    args = p.parse_args(argv)
    if args.json:
        print(json.dumps({
            "arm": arm(),
            "ablations": [
                {"name": a.name, "env": a.env, "setting": a.setting, "default_on": a.default_on,
                 "current": a.enabled(), "changes_prompt": a.changes_prompt, "effect": a.effect}
                for a in ABLATIONS
            ],
        }, indent=2))
    else:
        print(describe(use_color=sys.stdout.isatty()))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via _main in tests
    raise SystemExit(_main())
