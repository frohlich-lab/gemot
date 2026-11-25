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

**Discoverable.** ``python -m gemot.ablations`` prints the catalogue: every arm, its
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
because ``gemot.prompts`` has to import without ``petab_mcp`` (see
``tests/test_import_lightness``) and ``slurm/make_sweep.py`` imports from a bare interpreter.

RETIRED ARMS (``data_regimes``, ``diff_edits``, ``free_exponents``; why in ``docs/RATIONALE.md``)
are gone from this tuple and their code paths fixed at the old default, here and in the server.

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
    effect: str  # what the NON-DEFAULT side does, in one line

    @property
    def setting(self) -> str:
        """The literal assignment that flips this arm away from its default."""
        return f"{self.env}={'0' if self.default_on else '1'}"

    def enabled(self) -> bool:
        return _switch(self.env, default_on=self.default_on)

    def is_default(self) -> bool:
        return self.enabled() is self.default_on


ABLATIONS: tuple[Ablation, ...] = (
    Ablation(
        name="bio_literature",
        env="PETAB_BIO_LITERATURE",
        default_on=False,
        changes_prompt=False,
        effect="ON additionally gives the biocurator EuropePMC + bioRxiv literature search. Off by "
        "default because literature search over a benchmark problem's distinctive entity "
        "panel could surface its withheld source paper; the structured knowledge "
        "(interactions/pathways/ontologies/OmniPath) is general biology. On the panel, "
        "biocurator=off or biology_blind=on disables both kinds of knowledge access",
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
    Ablation(
        name="biocurator",
        env="PETAB_BIOCURATOR",
        default_on=True,
        changes_prompt=True,
        effect="OFF removes the biocurator from the deepagents panel, its external knowledge "
        "tools and the lead's curator guidance. Biological identifiers, descriptions, "
        "scenario context and the three modeling analysts are preserved. The "
        "biology_blind arm also removes the curator, regardless of this setting",
    ),
    Ablation(
        name="indra_statements",
        env="PETAB_INDRA_STATEMENTS",
        default_on=False,
        changes_prompt=True,
        effect="ON starts the server with an EMPTY entity map, which offers make_changes' INDRA "
        "`statements` route, and has the lead build the model as statements. Nothing binds the "
        "observables or inputs for it: the readouts and how the condition targets enter are its "
        "own, so no reference structure reaches the agent",
    ),
    Ablation(
        name="causal_constraints",
        env="PETAB_CAUSAL_CONSTRAINTS",
        default_on=False,
        changes_prompt=True,
        effect="ON starts the server with the problem's curated causal constraints "
        "(PEtabAgentBench's `indra/constraints.txt`), claims every model must make, and tells the "
        "lead they are the user's prior knowledge; make_changes refuses a model breaking one. "
        "Needs indra_statements on, and a problem that ships the file: either missing refuses the "
        "run. The lead never gets `set_constraints`",
    ),
    Ablation(
        name="reference_handoff",
        env="PETAB_REFERENCE_HANDOFF",
        default_on=True,
        changes_prompt=True,
        effect="OFF withholds what a problem's author wrote knowing the reference model: its "
        "known-constants note and the entity names curated for the biocurator. The agent has the "
        "problem's tables, its constants still shipping as named fixed rows, and the biocurator "
        "the measured entities and inputs with their identifiers, derived (`entity_ids`)",
    ),
    Ablation(
        name="fix_pointers",
        env="PETAB_FIX_POINTERS",
        default_on=True,
        changes_prompt=True,
        effect="OFF withholds the numbers that point the agent at what to change or where the fit is "
        "off: the merge hints and a tie's payoff, the ranked candidates, the regime grades, the "
        "per-observable residual tables and outliers, sigma heterogeneity and which observables a "
        "verdict flags (create_server(fix_pointers=False)), and the prompt lines that read them. "
        "Every score, total and verdict, and all prose, stay",
    ),
    Ablation(
        name="progress_signals",
        env="PETAB_PROGRESS_SIGNALS",
        default_on=True,
        changes_prompt=True,
        effect="OFF withholds the numbers that tell the agent how well it is doing: the server's "
        "best version, new-best flags, score differences against the parent or the best, chi2 "
        "p-value and reduced chi2 (create_server(progress_signals=False)), the time left on FIT "
        "COST, the subagents' turns-left notices, and the chi2 finish refusal that quotes the "
        "p-value; and the run no longer stops on the server's streak or plateau. Prose changes only "
        "where it names a withheld number. Each fit's own nll, criterion, k and chi2 stay, and the "
        "judge scores as before",
    ),
    Ablation(
        name="adversarial_prior",
        env="PETAB_ADVERSARIAL_PRIOR",
        default_on=False,
        changes_prompt=True,
        effect="ON shows the agent the scored model of an earlier run on the same problem (the "
        "archived run PETAB_PRIOR_RUN names, recorded in result.json as `prior_run`) and tells it "
        "to assume that mechanism is wrong and build a rival, or report that none fits as well. "
        "Refused without a prior, with one from another problem or another arm setting, and on "
        "biology_blind. Nothing enforces that the model differs; see gemot.prior_run",
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
    registry order (``"biocurator=off"``). Naming only the DEVIATIONS keeps the string stable when
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
    any default change (and a new arm appended at its default, or a retired one removed, DOES move
    it, which is the trade — group by `arm` for continuity, by this for correctness).
    """
    return ",".join(f"{name}={'on' if on else 'off'}" for name, on in state().items())


def describe(*, use_color: bool = False) -> str:
    """The catalogue printed by ``python -m gemot.ablations``."""
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
        f"Current arm: {bold}{arm()}{reset}\n\n"
        + "\n".join(rows)
        + f"\n{dim}Every run records these in result.json as `ablations`, plus the flat "
        f"`ablation_arm` column in the metrics CSV, so an A/B can always be split after the "
        f"fact.{reset}"
    )


def _main(argv: "list[str] | None" = None) -> int:
    import argparse
    import json
    import sys

    p = argparse.ArgumentParser(
        prog="python -m gemot.ablations",
        description="List the ablations available in this repo and their current values.",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="emit the registry as JSON (name, env, default, current, effect)",
    )
    args = p.parse_args(argv)
    if args.json:
        print(
            json.dumps(
                {
                    "arm": arm(),
                    "ablations": [
                        {
                            "name": a.name,
                            "env": a.env,
                            "setting": a.setting,
                            "default_on": a.default_on,
                            "current": a.enabled(),
                            "changes_prompt": a.changes_prompt,
                            "effect": a.effect,
                        }
                        for a in ABLATIONS
                    ],
                },
                indent=2,
            )
        )
    else:
        print(describe(use_color=sys.stdout.isatty()))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via _main in tests
    raise SystemExit(_main())
