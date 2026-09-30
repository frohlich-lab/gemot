"""The numbers the manuscript's prose states, as macros computed from the paper's inputs.

The prose is written by hand in Overleaf; its numbers must not be. Every count, median, range
and threshold a sentence quotes is computed here from the frozen inputs the figures draw
(``paper/*.json``), so a refreshed sweep changes the sentence together with the figure instead
of leaving it quoting the runs it replaced. The macros land in ``values.tex`` beside the
per-example ones, each with the definition it was computed by as a comment.

A number can carry a count, not a quantifier: "in every problem", "all 25", "none of the four"
stay words in the prose. :func:`checks` states each such claim as a predicate over the same
inputs, and the test suite fails when one stops holding -- the sentence then needs rewriting,
not re-rendering.

Formatting follows the prose it replaces: counts in text group thousands with a comma
(``1,755``); signed values are for math mode and group from five digits with a thin space
(``+44\\,625``, ``-2974``); counts the prose spells out have a ``Word`` twin (``five``).
"""

from __future__ import annotations

import ast
import csv
import importlib.util
import json
import math
import re
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from petab_harness import paper_fujita, paper_fujita_models

#: The split the text draws between "smaller" and "larger" problems, in training datapoints.
#: Figure 2's columns are ordered by that count, and the five problems above this line are the
#: ones whose usage and parsimony the text reports separately.
LARGER_PROBLEM_DATAPOINTS = 500
#: "Matched or improved on the reference by BIC": the conventional grade below which a BIC
#: difference is not worth more than a bare mention (Kass & Raftery 1995).
BIC_MATCH = 2
#: Below minus this, "very strong" evidence for the candidate (Kass & Raftery 1995); Figure 2
#: draws it as the dashed band.
BIC_STRONG = 10

_ONES = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
)
_TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")


def word(n: int) -> str:
    """``59`` -> ``fifty-nine``: the prose spells out small counts."""
    if not 0 <= n < 100:
        raise ValueError(f"{n} has no spelled-out form here")
    if n < 20:
        return _ONES[n]
    tens, ones = divmod(n, 10)
    return _TENS[tens] + (f"-{_ONES[ones]}" if ones else "")


def count(n: int) -> str:
    """A count in running text: ``1,755``."""
    return f"{n:,}"


def signed(value: float, digits: int = 0) -> str:
    """A signed value for math mode: ``+3``, ``-2974``, ``+44\\,625``; zero carries no sign."""
    text = f"{abs(value):.{digits}f}"
    integer, _, fraction = text.partition(".")
    if len(integer) >= 5:
        head = len(integer) % 3 or 3
        integer = r"\,".join(
            [integer[:head]] + [integer[i : i + 3] for i in range(head, len(integer), 3)]
        )
    body = integer + (f".{fraction}" if fraction else "")
    if float(text) == 0:
        return body
    return ("-" if value < 0 else "+") + body


def decimal(value: float, digits: int) -> str:
    return f"{value:.{digits}f}"


def hours(value: float) -> str:
    """One decimal below a hundred hours, whole hours with grouping above: ``2.0``, ``17,230``."""
    return f"{value:.1f}" if value < 100 else count(round(value))


def listing(names: list[str]) -> str:
    """``Perelson, Bruno, and Blasi``."""
    if len(names) <= 2:
        return " and ".join(names)
    return ", ".join(names[:-1]) + ", and " + names[-1]


def display(example: str) -> str:
    return example.capitalize()


def cap(text: str) -> str:
    """Sentence-initial form of a spelled-out count: ``fifty-nine`` -> ``Fifty-nine``."""
    return text[:1].upper() + text[1:]


def power(value: float) -> str:
    """A small constant for math mode: ``10^{-16}``, or ``2.5\\times10^{-5}``."""
    exponent = math.floor(math.log10(abs(value)))
    mantissa = value / 10**exponent
    if math.isclose(mantissa, 1.0):
        return f"10^{{{exponent}}}"
    return f"{mantissa:g}\\times10^{{{exponent}}}"


def number(value: float) -> str:
    """A constant as the code writes it, without trailing zeros: ``0.05``, ``1.428``, ``200``."""
    return f"{value:g}" if isinstance(value, float) else str(value)


def date(iso: str) -> str:
    """``2026-09-12`` -> ``12 September 2026``."""
    moment = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    return f"{moment.day} {moment:%B %Y}"


def date_range(first: datetime, last: datetime) -> str:
    """``10 and 21 September 2026``, naming the months and years only where they differ."""
    if (first.year, first.month) == (last.year, last.month):
        return f"{first.day} and {last.day} {last:%B %Y}"
    if first.year == last.year:
        return f"{first.day} {first:%B} and {last.day} {last:%B %Y}"
    return f"{first.day} {first:%B %Y} and {last.day} {last:%B %Y}"


def alternatives(names: list[str]) -> str:
    """``Crauste, Lucarelli or Raimundez``."""
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " or " + names[-1]


@dataclass(frozen=True)
class Claim:
    """A quantified sentence of the prose, as a predicate over the inputs."""

    where: str
    text: str
    holds: bool
    detail: str


def _finite(values):
    return [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]


# ------------------------------------------------------------------------------ Figure 2


def _sweep_groups(sweep: dict) -> tuple[list[str], list[str]]:
    order = sweep["example_order"]
    larger = [e for e in order if sweep["counts"][e]["n_datapoints"] >= LARGER_PROBLEM_DATAPOINTS]
    return [e for e in order if e not in larger], larger


def _runs(sweep: dict, examples: list[str]) -> list[dict]:
    return [run for run in sweep["runs"] if run["example"] in examples]


def _median_by_example(sweep: dict, examples: list[str], key: str) -> list[float]:
    return [statistics.median(_finite(run[key] for run in _runs(sweep, [e]))) for e in examples]


def _topology_defined(run: dict) -> bool:
    """A comparison Figure 2's bottom row can draw: the normalised edit distance exists."""
    return run.get("topo_collapsed_normalised") is not None


def _connectivity_changed(run: dict) -> bool:
    return bool(run["topo_new_connections"] or run["topo_deleted_connections"])


def _depth_only(run: dict) -> bool:
    return not _connectivity_changed(run) and bool(
        run["topo_extra_depth"] or run["topo_removed_depth"]
    )


def sweep_macros(sweep: dict, configurations: dict) -> dict[str, tuple[str, str]]:
    """Figure 2's text numbers: ``macro -> (value, how it is computed)``."""
    smaller, larger = _sweep_groups(sweep)
    runs = sweep["runs"]
    out: dict[str, tuple[str, str]] = {}

    def put(name, value, how):
        out["sweepFigure" + name] = (value, how)

    put(
        "LargerProblemThreshold",
        count(LARGER_PROBLEM_DATAPOINTS),
        "training datapoints from which a problem counts as larger",
    )
    put("SmallerProblemCount", str(len(smaller)), "problems below the threshold")
    put("SmallerProblemCountWord", word(len(smaller)), "the same, spelled out")
    put("LargerProblemCount", str(len(larger)), "problems at or above the threshold")
    put("LargerProblemCountWord", word(len(larger)), "the same, spelled out")
    put("SmallerRunCount", str(len(_runs(sweep, smaller))), "runs on the smaller problems")
    put("LargerRunCount", str(len(_runs(sweep, larger))), "runs on the larger problems")
    for key, name in (
        ("n_observables", "Observables"),
        ("n_datapoints", "Datapoints"),
        ("n_conditions", "Conditions"),
        ("n_modalities", "Modalities"),
    ):
        values = [sweep["counts"][e][key] for e in sweep["example_order"]]
        put("Min" + name, count(min(values)), f"smallest training {key} over the problems")
        put("Max" + name, count(max(values)), f"largest training {key} over the problems")

    put(
        "MaxModalitiesWord",
        word(max(sweep["counts"][e]["n_modalities"] for e in sweep["example_order"])),
        "the same, spelled out",
    )
    put(
        "TotalDatapoints",
        count(sum(sweep["counts"][e]["n_datapoints"] for e in sweep["example_order"])),
        "training datapoints summed over the problems",
    )
    put(
        "ModalityCount",
        str(len({m for e in sweep["example_order"] for m in sweep["counts"][e]["modalities"]})),
        "distinct measurement modalities over the problems",
    )

    for group, names in (("Smaller", smaller), ("Larger", larger)):
        rs = _runs(sweep, names)
        put(
            group + "MedianTokensMillions",
            decimal(statistics.median(_finite(r["tokens"] for r in rs)) / 1e6, 1),
            f"median LLM tokens per run over the {group.lower()} problems' runs, pooled",
        )
        put(
            group + "MedianCPUHours",
            hours(statistics.median(_finite(r["cpu_h"] for r in rs))),
            f"median CPU-hours per run over the {group.lower()} problems' runs, pooled",
        )
    put(
        "TotalTokensMillions",
        decimal(sum(_finite(r["tokens"] for r in runs)) / 1e6, 1),
        "LLM tokens summed over every run",
    )
    put(
        "TotalCPUHours",
        hours(sum(_finite(r["cpu_h"] for r in runs))),
        "CPU-hours summed over every run",
    )

    put("BICMatchThreshold", str(BIC_MATCH), "the ΔBIC below which a run matched the reference")
    put(
        "BICMatchCount",
        str(sum(r["delta_bic"] < BIC_MATCH for r in runs)),
        "runs with ΔBIC below the match threshold",
    )
    below = {e: sum(r["delta_bic"] < 0 for r in _runs(sweep, [e])) for e in sweep["example_order"]}
    fewest = min(below.values())
    put("BICBelowReferenceFewest", str(fewest), "the fewest runs any one problem has with ΔBIC < 0")
    put("BICBelowReferenceFewestWord", word(fewest), "the same, spelled out")
    aic_short = [
        display(e)
        for e in sweep["example_order"]
        if sum(r["delta_aic"] < 0 for r in _runs(sweep, [e])) < fewest
    ]
    put(
        "AICBelowReferenceExceptions",
        listing(aic_short),
        "problems with fewer runs at ΔAIC < 0 than the ΔBIC minimum above",
    )
    strong = [r for r in runs if r["delta_bic"] < -BIC_STRONG]
    strong_problems = [e for e in sweep["example_order"] if any(r["example"] == e for r in strong)]
    put("StrongEvidenceThreshold", str(BIC_STRONG), "minus the ΔBIC of very strong evidence")
    put("StrongEvidenceCount", str(len(strong)), "runs with ΔBIC below minus that")
    put("StrongEvidenceProblemCount", str(len(strong_problems)), "problems with at least one")
    put(
        "StrongEvidenceExceptions",
        listing([display(e) for e in sweep["example_order"] if e not in strong_problems]),
        "problems with none",
    )
    put("BetterFitCount", str(sum(r["delta_nll"] < 0 for r in runs)), "runs with ΔNLLH < 0")

    literal = {
        run["archive_id"]
        for run in configurations["runs"]
        if run["cohort"] == "figure2" and run["hardcoded_constants"]
    }
    with_literals = [r for r in runs if r["archive_id"] in literal]
    put(
        "LiteralModelCount",
        str(len(with_literals)),
        "delivered models with at least one hardcoded numeric literal (run_configurations.json)",
    )
    put(
        "LiteralProblemCount",
        str(len({r["example"] for r in with_literals})),
        "problems with at least one such model",
    )

    for group, names in (("Smaller", smaller), ("Larger", larger)):
        for key, name in (("delta_k", "Parameters"), ("delta_states", "States")):
            medians = _median_by_example(sweep, names, key)
            put(
                f"{group}MedianDelta{name}Low",
                signed(min(medians)),
                f"lowest per-problem median {key} over the {group.lower()} problems",
            )
            put(
                f"{group}MedianDelta{name}High",
                signed(max(medians)),
                f"highest per-problem median {key} over the {group.lower()} problems",
            )

    defined = [r for r in runs if _topology_defined(r)]
    connectivity = [r for r in defined if _connectivity_changed(r)]
    depth_only = [r for r in defined if _depth_only(r)]
    put("TopologyDefinedCount", str(len(defined)), "runs with a normalised edit distance")
    put(
        "TopologyDefinedProblemCount",
        str(len({r["example"] for r in defined})),
        "problems with at least one",
    )
    put(
        "TopologyDifferCount",
        str(len(connectivity) + len(depth_only)),
        "defined comparisons that differ in connectivity or latent depth",
    )
    put(
        "TopologyConnectivityCount",
        str(len(connectivity)),
        "defined comparisons with an added or deleted connection",
    )
    put(
        "TopologyDepthOnlyCount",
        str(len(depth_only)),
        "defined comparisons that differ only in route length",
    )
    put("TopologyDepthOnlyCountWord", word(len(depth_only)), "the same, spelled out")
    put(
        "TopologyMatchCount",
        str(len(defined) - len(connectivity) - len(depth_only)),
        "defined comparisons that match both summaries",
    )
    put(
        "TopologyAddedCount",
        str(sum(bool(r["topo_new_connections"]) for r in defined)),
        "defined comparisons with an added connection",
    )
    put(
        "TopologyDeletedCount",
        str(sum(bool(r["topo_deleted_connections"]) for r in defined)),
        "defined comparisons with a deleted connection",
    )
    big = _runs(sweep, larger)
    put(
        "LargerAddedConnectionsCount",
        str(sum(bool(r["topo_new_connections"]) for r in big)),
        "larger-problem runs that add a connection",
    )
    put(
        "LargerFewerStatesCount",
        str(sum(r["delta_states"] < 0 for r in big)),
        "larger-problem runs with fewer states than the reference",
    )
    put(
        "LargerSameStatesCount",
        str(sum(r["delta_states"] == 0 for r in big)),
        "larger-problem runs with as many states as the reference",
    )
    put(
        "LargerMoreStatesCount",
        str(sum(r["delta_states"] > 0 for r in big)),
        "larger-problem runs with more states than the reference",
    )
    return out


def sweep_checks(sweep: dict) -> list[Claim]:
    smaller, larger = _sweep_groups(sweep)
    big = _runs(sweep, larger)
    added = sum(bool(r["topo_new_connections"]) for r in big)
    return [
        Claim(
            "results-topology",
            "All larger-problem runs added connections",
            added == len(big),
            f"{added}/{len(big)}",
        ),
    ]


# ------------------------------------------------------------------------------ Figure 3


def _fujita_arms(metadata: dict) -> dict[str, dict]:
    """arm key -> spec, in the figure's order."""
    return paper_fujita.arms(metadata)


def arms_launched(metadata: dict) -> dict[str, int]:
    """arm code -> runs launched, the models it delivered and the runs that built none."""
    return {
        spec["code"]: spec["n_models"] + spec["n_no_model"]
        for spec in _fujita_arms(metadata).values()
    }


#: How the prose names each language model's pool of Figure 3 arms.
FAMILY_WORDS = {"OpenWeight": "open-weight", "Frontier": "frontier"}


def _name(metadata: dict, row: dict) -> str:
    return paper_fujita.run_name(metadata, row["label"])


def fujita_macros(
    metadata: dict, rows: list[dict], configurations: dict
) -> dict[str, tuple[str, str]]:
    """Figure 3's text numbers: ``macro -> (value, how it is computed)``."""
    arms = _fujita_arms(metadata)
    code = {arm: spec["code"] for arm, spec in arms.items()}
    out: dict[str, tuple[str, str]] = {}

    def put(name, value, how):
        out[name] = (value, how)

    put("fujitaFigureArmCountWord", word(len(arms)), "configurations, spelled out")
    models = sorted({spec["factors"]["model"] for spec in arms.values()})
    put("fujitaFigureLLMCountWord", word(len(models)), "distinct language models, spelled out")
    launched = {arm: spec["n_models"] + spec["n_no_model"] for arm, spec in arms.items()}
    if len(set(launched.values())) != 1:
        raise ValueError(f"arms launched different numbers of runs: {launched}")
    per_arm = next(iter(launched.values()))
    put("fujitaFigureRunsPerArm", str(per_arm), "runs launched per configuration")
    put("fujitaFigureRunsPerArmWord", word(per_arm), "the same, spelled out")

    pools = {
        "OpenWeight": [r for r in rows if code[r["arm"]].startswith("O")],
        "Frontier": [r for r in rows if code[r["arm"]].startswith("F")],
    }
    for family, sample in pools.items():
        pairs = [
            (r["training_dbic"], r["heldout_delta_nll"]) for r in sample if r["complete_heldout"]
        ]
        put(
            f"fujitaFigure{family}DBICHeldoutSpearman",
            decimal(_spearman(pairs), 1),
            f"Spearman rank correlation of training ΔBIC and held-out ΔNLL, {FAMILY_WORDS[family]} "
            "language model, completely scored models",
        )
        put(f"fujitaFigure{family}DBICHeldoutSpearmanN", str(len(pairs)), "models it ranks")

    for row in rows:
        prefix = "fujitaModel" + paper_fujita.macro_name(_name(metadata, row))
        put(prefix + "TrainingDBIC", signed(row["training_dbic"]), "training ΔBIC")
        put(
            prefix + "HeldoutDNLL",
            signed(row["heldout_delta_nll"]) if row["complete_heldout"] else "unscorable",
            "held-out ΔNLL against the reference",
        )

    for arm, spec in arms.items():
        prefix = "fujitaFigureArm" + paper_fujita.macro_name(spec["code"])
        mine = [r for r in rows if r["arm"] == arm]
        complete = [r for r in mine if r["complete_heldout"]]
        beat = [r for r in complete if r["heldout_delta_nll"] < 0]
        put(prefix + "Runs", str(launched[arm]), "runs launched")
        put(prefix + "ModelsWord", word(len(mine)), "models delivered, spelled out")
        put(prefix + "Complete", str(len(complete)), "models with a complete held-out score")
        put(prefix + "CompleteWord", word(len(complete)), "the same, spelled out")
        put(
            prefix + "BeatReference",
            str(len(beat)),
            "completely scored models with held-out ΔNLL < 0",
        )
        put(
            prefix + "MedianTrainingDBIC",
            signed(statistics.median(r["training_dbic"] for r in mine)),
            "median training ΔBIC over the delivered models",
        )
        # A failed held-out integration counts as predicting worse than any score (Figure 3D).
        median = statistics.median(
            r["heldout_delta_nll"] if r["complete_heldout"] else math.inf for r in mine
        )
        put(
            prefix + "MedianHeldoutDNLL",
            signed(median) if math.isfinite(median) else r"\infty",
            "median held-out ΔNLL over the delivered models, a failed integration ranked worst",
        )
        tool, rest, _ = paper_fujita.time_totals(metadata, arm)
        put(
            prefix + "HoursPerModel",
            decimal(tool + rest, 1),
            "wall hours per delivered model, every launched run charged (Figure 3F)",
        )
        literal = sum(
            1
            for run in configurations["runs"]
            if run["cohort"] == "figure3"
            and run["arm"] == spec["code"]
            and run["hardcoded_constants"]
        )
        put(
            prefix + "LiteralModels",
            str(literal),
            "delivered models with at least one hardcoded numeric literal",
        )

    visible = [r for r in rows if code[r["arm"]] in ("O1", "O2", "O3") and r["complete_heldout"]]
    put(
        "fujitaFigureVisibleOpenWeightComplete",
        str(len(visible)),
        "completely scored biology-visible GLM models (O1-O3)",
    )
    put(
        "fujitaFigureVisibleOpenWeightBeatReference",
        str(sum(r["heldout_delta_nll"] < 0 for r in visible)),
        "of which held-out ΔNLL < 0",
    )
    put(
        "fujitaFigureVisibleOpenWeightModels",
        str(sum(code[r["arm"]] in ("O1", "O2", "O3") for r in rows)),
        "biology-visible GLM models delivered (O1-O3), the denominator Figure 3D uses",
    )
    frontier = pools["Frontier"]
    put(
        "fujitaFigureFrontierModelsWord",
        word(len(frontier)),
        "models the frontier language model delivered, spelled out",
    )
    open_scored = [r for r in pools["OpenWeight"] if r["complete_heldout"]]
    frontier_scored = [r for r in frontier if r["complete_heldout"]]
    best_open = min(open_scored, key=lambda r: r["heldout_delta_nll"])
    best_frontier = min(frontier_scored, key=lambda r: r["heldout_delta_nll"])
    put(
        "fujitaFigureBestOpenWeightLabel",
        _name(metadata, best_open),
        "the GLM model with the lowest held-out ΔNLL",
    )
    put("fujitaFigureBestOpenWeightHeldoutDNLL", signed(best_open["heldout_delta_nll"]), "its ΔNLL")
    put(
        "fujitaFigureBestFrontierLabel",
        _name(metadata, best_frontier),
        "the frontier model with the lowest held-out ΔNLL",
    )
    put(
        "fujitaFigureBestFrontierHeldoutDNLL",
        signed(best_frontier["heldout_delta_nll"]),
        "its ΔNLL",
    )
    put(
        "fujitaFigureBestOpenWeightShareOfFrontier",
        str(round(100 * best_open["heldout_delta_nll"] / best_frontier["heldout_delta_nll"])),
        "the best GLM improvement as a percentage of the best frontier improvement",
    )

    for family, prefix in (("OpenWeight", "O"), ("Frontier", "F")):
        mine = [arm for arm in arms if code[arm].startswith(prefix)]
        per_model = [sum(paper_fujita.time_totals(metadata, arm)[:2]) for arm in mine]
        usd = [sum(paper_fujita.cost_totals(metadata, arm)[:2]) for arm in mine]
        put(
            f"fujitaFigure{family}HoursPerModelLow",
            str(round(min(per_model))),
            "fewest wall hours per delivered model over the arms, rounded",
        )
        put(
            f"fujitaFigure{family}HoursPerModelHigh",
            str(round(max(per_model))),
            "most wall hours per delivered model over the arms, rounded",
        )
        put(
            f"fujitaFigure{family}USDPerModelCeiling",
            str(math.ceil(max(usd))),
            "whole dollars no arm's cost per delivered model reaches",
        )
        put(
            f"fujitaFigure{family}USDPerModelFloor",
            str(math.floor(min(usd))),
            "whole dollars every arm's cost per delivered model exceeds",
        )
    return out


def _spearman(pairs: list[tuple[float, float]]) -> float:
    from scipy.stats import spearmanr

    xs, ys = zip(*pairs, strict=True)
    return float(spearmanr(xs, ys).statistic)


def fujita_checks(metadata: dict, rows: list[dict]) -> list[Claim]:
    arms = _fujita_arms(metadata)
    code = {arm: spec["code"] for arm, spec in arms.items()}
    by_code = {spec["code"]: [r for r in rows if r["arm"] == arm] for arm, spec in arms.items()}
    named = {_name(metadata, r): r for r in rows}
    scored = sorted(
        (r for r in rows if r["complete_heldout"]), key=lambda r: r["heldout_delta_nll"]
    )
    frontier = [r for r in rows if code[r["arm"]].startswith("F")]
    c2_retained = sum(r["contains_reference_topology"] for r in by_code["O2"])
    unscored_blinded = [_name(metadata, r) for r in by_code["O4"] if not r["complete_heldout"]]
    unscored = sorted(_name(metadata, r) for r in rows if not r["complete_heldout"])

    def median_heldout(arm_code):
        """Over every delivered model; a failed held-out integration ranks worst (Figure 3D)."""
        return statistics.median(
            r["heldout_delta_nll"] if r["complete_heldout"] else math.inf for r in by_code[arm_code]
        )

    return [
        Claim(
            "results-heldout",
            "O1c: lower training ΔBIC than O1a, far worse held-out ΔNLL",
            named["O1c"]["training_dbic"] < named["O1a"]["training_dbic"]
            and named["O1c"]["heldout_delta_nll"] > named["O1a"]["heldout_delta_nll"],
            "O1c vs O1a",
        ),
        Claim(
            "results-heldout",
            "F2c and F3d are the two best held-out models",
            {_name(metadata, r) for r in scored[:2]} == {"F2c", "F3d"},
            ", ".join(_name(metadata, r) for r in scored[:2]),
        ),
        Claim(
            "results-heldout",
            "All O2 models retained the reference's readout connections",
            c2_retained == len(by_code["O2"]),
            f"{c2_retained}/{len(by_code['O2'])}",
        ),
        # "Did not impair generalisation" and "predicted as well as biology-visible runs".
        Claim(
            "results-heldout",
            "Blinded runs have a lower median held-out ΔNLL than each "
            "biology-visible GLM arm (O4 below O1, O2 and O3)",
            all(median_heldout("O4") < median_heldout(c) for c in ("O1", "O2", "O3")),
            ", ".join(f"{c} {median_heldout(c):+.0f}" for c in ("O1", "O2", "O3", "O4")),
        ),
        Claim(
            "evaluation",
            "O4b is the only blinded model without a complete held-out score, and "
            "no held-out condition integrates for it",
            unscored_blinded == ["O4b"] and named["O4b"]["conditions_ok"] == 0,
            f"unscored {', '.join(unscored_blinded)}; O4b {named['O4b']['conditions_ok']}/10",
        ),
        # The appendix names the delivered models without a complete score, twice.
        Claim(
            "evaluation",
            "O2c and O4b are the only delivered models without a complete held-out score",
            unscored == ["O2c", "O4b"],
            ", ".join(unscored),
        ),
        Claim(
            "results-heldout",
            "Both biology-visible GLM panels (O1, O2) delivered every run",
            all(len(by_code[c]) == arms_launched(metadata)[c] for c in ("O1", "O2")),
            ", ".join(f"{c} {len(by_code[c])}/{arms_launched(metadata)[c]}" for c in ("O1", "O2")),
        ),
        Claim(
            "results-heldout",
            "Every frontier candidate beat the reference",
            all(r["complete_heldout"] and r["heldout_delta_nll"] < 0 for r in frontier),
            f"{sum(r['complete_heldout'] and r['heldout_delta_nll'] < 0 for r in frontier)}"
            f"/{len(frontier)}",
        ),
        Claim(
            "results-heldout",
            "Bare Astra (F3) has a lower median held-out ΔNLL than F2",
            median_heldout("F3") < median_heldout("F2"),
            "F3 vs F2",
        ),
    ]


# ------------------------------------------------------------------------------ Figure 4


def heldout_macros(heldout: dict) -> dict[str, tuple[str, str]]:
    examples = heldout["examples"]
    return {
        "heldoutFigureExampleCountWord": (
            word(len(examples)),
            "systems Figure 4 scores, spelled out",
        ),
        # Fujita (Figure 3) plus Figure 4's systems.
        "heldoutBenchmarkCountWord": (
            word(1 + len(examples)),
            "benchmarks with held-out data, Fujita included, spelled out",
        ),
    }


# ------------------------------------------------------------------------------ code constants


def _definitions(root: Path, path: str) -> ast.Module:
    if path.startswith(_MCP):
        # The server's path in its own repository; read where that repository is installed.
        source = Path(importlib.util.find_spec("petab_mcp").origin).parent / path[len(_MCP) :]
    else:
        source = root / path
    return ast.parse(source.read_text(encoding="utf-8"))


def _literal(node: ast.expr):
    """The value an assignment writes, looking through an env-overridable default:
    ``_env_float("PETAB_MCP_FIT_PER_START_TIMEOUT", 180.0)`` is the coded 180, which is what a
    run gets unless its environment says otherwise -- importing the module would report the
    reading machine's environment instead."""
    if isinstance(node, ast.Call) and node.args:
        return ast.literal_eval(node.args[-1])
    return ast.literal_eval(node)


def module_constant(root: Path, path: str, name: str):
    for node in _definitions(root, path).body:
        targets = (
            node.targets
            if isinstance(node, ast.Assign)
            else [node.target]
            if isinstance(node, ast.AnnAssign)
            else []
        )
        for target in targets:
            if isinstance(target, ast.Name) and target.id == name:
                return _literal(node.value)
            if isinstance(target, ast.Tuple) and isinstance(node.value, ast.Tuple):
                for element, value in zip(target.elts, node.value.elts, strict=True):
                    if isinstance(element, ast.Name) and element.id == name:
                        return _literal(value)
    raise KeyError(f"{path}: no module-level {name}")


def local_constant(root: Path, path: str, function: str, name: str):
    """A constant assigned inside one function, tuple unpacking included."""
    for node in ast.walk(_definitions(root, path)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function:
            for assign in (sub for sub in ast.walk(node) if isinstance(sub, ast.Assign)):
                for target in assign.targets:
                    if isinstance(target, ast.Name) and target.id == name:
                        return _literal(assign.value)
                    if isinstance(target, ast.Tuple) and isinstance(assign.value, ast.Tuple):
                        for element, value in zip(target.elts, assign.value.elts, strict=True):
                            if isinstance(element, ast.Name) and element.id == name:
                                return _literal(value)
    raise KeyError(f"{path}: no {name} in {function}")


def signature_default(root: Path, path: str, function: str, parameter: str):
    for node in ast.walk(_definitions(root, path)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function:
            arguments = node.args
            positional = arguments.posonlyargs + arguments.args
            pairs = list(
                zip(
                    positional[len(positional) - len(arguments.defaults) :],
                    arguments.defaults,
                    strict=True,
                )
            )
            pairs += [
                (a, d)
                for a, d in zip(arguments.kwonlyargs, arguments.kw_defaults, strict=True)
                if d is not None
            ]
            for argument, default in pairs:
                if argument.arg == parameter:
                    return ast.literal_eval(default)
    raise KeyError(f"{path}: no default for {function}({parameter})")


def argparse_default(root: Path, path: str, flag: str):
    for node in ast.walk(_definitions(root, path)):
        if (
            isinstance(node, ast.Call)
            and getattr(node.func, "attr", "") == "add_argument"
            and any(isinstance(a, ast.Constant) and a.value == flag for a in node.args)
        ):
            for keyword in node.keywords:
                if keyword.arg == "default":
                    return ast.literal_eval(keyword.value)
    raise KeyError(f"{path}: no default for {flag}")


_MCP = "src/petab_mcp/"
#: Every code default the appendix quotes: ``(macro, file, reader, reader arguments)``. Read
#: statically from the source, so the number printed is the number the code ships.
CODE_DEFAULTS = (
    # fit scores and numerical consistency
    ("NewBestMinDelta", _MCP + "constants.py", "module", ("NEW_BEST_MIN_DELTA_SCORE",)),
    ("ObjectiveAgreementTolerance", _MCP + "fitting.py", "module", ("OBJECTIVE_AGREEMENT_RTOL",)),
    # individual residuals and observable summaries
    ("WorstResiduals", _MCP + "server.py", "signature", ("get_fit_diagnosis", "worst_n")),
    ("ObservableRows", _MCP + "server.py", "signature", ("get_fit_diagnosis", "max_observables")),
    ("BiasRows", _MCP + "data_access.py", "local", ("fit_diagnosis", "_BIAS_KEEP")),
    ("FlaggedRows", _MCP + "data_access.py", "local", ("fit_diagnosis", "_FLAG_KEEP")),
    # bias and temporal structure
    ("StructureAlpha", _MCP + "data_access.py", "module", ("_STRUCTURE_ALPHA",)),
    ("MinimumPowerPoints", _MCP + "residual_tests.py", "module", ("MIN_POWER_N",)),
    # replicate lack of fit
    ("ReplicateMinDF", _MCP + "constants.py", "module", ("REPLICATE_LOF_MIN_DF",)),
    ("ReplicateMinPoints", _MCP + "constants.py", "module", ("REPLICATE_LOF_MIN_POINTS",)),
    ("ReplicateAlpha", _MCP + "constants.py", "module", ("REPLICATE_LOF_ALPHA",)),
    (
        "ReplicateReportedPoints",
        _MCP + "constants.py",
        "module",
        ("REPLICATE_LOF_MAX_REPORTED_POINTS",),
    ),
    ("ReplicatePoweredDF", _MCP + "constants.py", "module", ("REPLICATE_LOF_MIN_LOF_DF",)),
    # estimated noise scales
    ("SigmaInflated", _MCP + "constants.py", "module", ("NOISE_SIGMA_MAX_FRACTION_OF_DATA_STD",)),
    ("SigmaCollapsed", _MCP + "constants.py", "module", ("NOISE_SIGMA_COLLAPSE_FRACTION",)),
    ("SigmaMinPoints", _MCP + "constants.py", "module", ("NOISE_SIGMA_MIN_POINTS",)),
    ("SigmaReplicateMinDOF", _MCP + "constants.py", "module", ("NOISE_SIGMA_REPLICATE_MIN_DOF",)),
    # parameters, bounds and local uncertainty
    ("AtBoundTolerance", _MCP + "constants.py", "module", ("PARAMETER_AT_BOUND_RELATIVE_TOL",)),
    (
        "CorrelationThreshold",
        _MCP + "identifiability.py",
        "signature",
        ("_identifiability_stats", "corr_threshold"),
    ),
    (
        "FlatDirection",
        _MCP + "identifiability.py",
        "signature",
        ("_identifiability_stats", "flat_rel"),
    ),
    ("IdentifiabilityRows", _MCP + "server.py", "signature", ("get_fit_diagnosis", "top_n")),
    (
        "ScaledStandardError",
        _MCP + "identifiability.py",
        "signature",
        ("_identifiability_stats", "se_unconstrained"),
    ),
    (
        "MultistartSpread",
        _MCP + "identifiability.py",
        "signature",
        ("_identifiability_stats", "spread_unstable"),
    ),
    ("MergeHints", _MCP + "identifiability.py", "module", ("MERGE_HINT_LIMIT",)),
    # optimisation, revision, stopping
    ("FittingLogTail", _MCP + "server.py", "signature", ("get_fitting_log", "tail_lines")),
    ("NestingTolerance", _MCP + "nesting.py", "module", ("_NLL_WORSE_TOLERANCE",)),
    (
        "StructureRefusals",
        "src/petab_harness/agent/finish_gate.py",
        "module",
        ("_MAX_STRUCTURE_FINISH_REFUSALS",),
    ),
    (
        "SigmaRefusals",
        "src/petab_harness/agent/finish_gate.py",
        "module",
        ("_MAX_SIGMA_FINISH_REFUSALS",),
    ),
    (
        "ParsimonyRefusals",
        "src/petab_harness/agent/finish_gate.py",
        "module",
        ("_MAX_PARSIMONY_FINISH_REFUSALS",),
    ),
    # the MCP tools' pages
    ("MeasurementRows", _MCP + "server.py", "signature", ("get_measurements", "max_rows")),
    ("QueryRows", _MCP + "server.py", "signature", ("query_data", "max_rows")),
    ("MaxPageRows", _MCP + "data_access.py", "module", ("MAX_PAGE_ROWS",)),
    ("ProvenanceRecords", _MCP + "server.py", "signature", ("get_provenance", "limit")),
    ("ErrorRecords", _MCP + "server.py", "signature", ("get_errors", "limit")),
    # fitting and warm starts
    ("WarmStartJitter", _MCP + "fitting.py", "module", ("WARM_START_JITTER",)),
    ("FitSeed", _MCP + "fitting.py", "module", ("FIT_SEED",)),
    ("PerStartTimeout", "src/petab_harness/fit_budget.py", "module", ("FIT_PER_START_TIMEOUT_S",)),
    ("AmiciAbsoluteTolerance", "src/petab_harness/fit_budget.py", "module", ("FIT_AMICI_ATOL",)),
    # the runners' command-line defaults
    ("RunnerStarts", "src/petab_harness/runner/cli.py", "argparse", ("--n-starts",)),
    ("RunnerRecursionLimit", "src/petab_harness/runner/cli.py", "argparse", ("--recursion-limit",)),
    ("SweepStarts", "slurm/make_sweep.py", "argparse", ("--n-starts",)),
    ("SweepRecursionLimit", "slurm/make_sweep.py", "argparse", ("--recursion-limit",)),
    ("SweepWallMargin", "slurm/make_sweep.py", "argparse", ("--wall-time-margin",)),
    # the tolerance the held-out annex's tighter drawings use (Raimundez, both cohorts)
    ("DrawAbsoluteTolerance", "slurm/_simulate_raimundez_heldout.py", "module", ("ATOL",)),
    ("DrawRelativeTolerance", "slurm/_simulate_raimundez_heldout.py", "module", ("RTOL",)),
    # the neural-ODE baseline
    (
        "BasalSampleRows",
        "src/petab_harness/neural_ode/distil.py",
        "module",
        ("_BASAL_SAMPLE_ROWS",),
    ),
    (
        "DistilledPoints",
        "slurm/_heldout_benchmarks_input.py",
        "signature",
        ("distilled_points", "top"),
    ),
)
_READERS = {
    "module": module_constant,
    "local": local_constant,
    "signature": signature_default,
    "argparse": argparse_default,
}


def code_macros(root: Path) -> dict[str, tuple[str, str]]:
    """``\\code<Name>`` as the code writes it, ``...Word`` for whole numbers the prose spells
    out, ``...Power`` for a small constant in powers of ten and ``...Percent`` for a fraction
    the prose states as a percentage."""
    out: dict[str, tuple[str, str]] = {}
    values = {}
    for name, path, reader, arguments in CODE_DEFAULTS:
        value = _READERS[reader](root, path, *arguments)
        values[name] = value
        where = f"{path}: {'.'.join(str(a) for a in arguments)}"
        out["code" + name] = (
            count(value) if isinstance(value, int) and value >= 1000 else number(value),
            where,
        )
        if float(value).is_integer() and 0 <= value < 100:
            out["code" + name + "Word"] = (word(int(value)), "the same, spelled out")
        if isinstance(value, float) and 0 < value < 1:
            out["code" + name + "Power"] = (power(value), "the same, in powers of ten")
            if value >= 1e-4:
                out["code" + name + "Percent"] = (
                    number(round(100 * value, 6)),
                    "the same, in percent",
                )
    # Derived in the text's own terms.
    out["codeDiagnosisObservableRows"] = (
        str(values["ObservableRows"] + values["BiasRows"] + values["FlaggedRows"]),
        "most observable rows a diagnosis returns: the largest, the most biased, the flagged",
    )
    out["codeReplicateMinReplicatesWord"] = (
        word(values["ReplicateMinDF"] + 1),
        "finite replicates a design point needs",
    )
    weak = [word(n) for n in range(1, values["ReplicatePoweredDF"])]
    out["codeReplicateWeakDF"] = (
        alternatives(weak) if len(weak) > 1 else weak[0],
        "lack-of-fit degrees of freedom the test calls weak",
    )
    refusals = {values[k] for k in ("StructureRefusals", "SigmaRefusals", "ParsimonyRefusals")}
    if len(refusals) != 1:
        raise ValueError(f"the finish gate's refusal limits differ: {refusals}")
    return out


# ------------------------------------------------------------------------------ appendices

_NUMBERS = ("One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten")


def _counted(name: str, n: int) -> dict[str, tuple[str, str]]:
    """A count and its spelled-out forms, lower case and sentence-initial."""
    return {
        name: (str(n), ""),
        name + "Word": (word(n), "spelled out"),
        name + "WordCap": (cap(word(n)), "spelled out, sentence-initial"),
    }


def model_names(sweep: dict) -> dict[str, str]:
    """task -> ``crauste_2``: the benchmark annex's names, numbered by start time per problem."""
    names = {}
    for example in sweep["example_order"]:
        runs = sorted(_runs(sweep, [example]), key=lambda r: (r["started"], r["task"]))
        for number_, run in enumerate(runs, 1):
            names[run["task"]] = f"{example}_{number_}"
    return names


_DIGITS = ("Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine")


def _macro_part(text: str) -> str:
    """``alpha_act`` -> ``AlphaAct``, ``k_M2`` -> ``KMTwo``: letters only, as LaTeX needs."""
    parts = []
    for piece in re.split(r"[_\-]", text):
        spelled = "".join(_DIGITS[int(char)] if char.isdigit() else char for char in piece)
        parts.append(spelled[:1].upper() + spelled[1:])
    return "".join(parts)


def _model_macro(name: str) -> str:
    example, number_ = name.rsplit("_", 1)
    return "sweepModel" + _macro_part(example) + _NUMBERS[int(number_) - 1]


def _component(value: float | None) -> str:
    """A normalised edit-distance component as the annex table prints it: ``0.250``, ``0``."""
    return "--" if value is None else ("0" if value == 0 else f"{value:.3f}")


#: The Figure 2 models whose fitted parameters the annex prose quotes (mechanistic examples and
#: the further structural examples); every other model has its summary macros only.
DISCUSSED_MODELS = (
    "crauste_2",
    "crauste_3",
    "brannmark_2",
    "armistead_1",
    "armistead_5",
    "sneyd_5",
    "fiedler_5",
    "raimundez_1",
)


def _raw(value: float) -> str:
    """Full precision for ``\\resultnum`` to round, never exponent notation."""
    text = f"{value:.12f}".rstrip("0").rstrip(".")
    return "0" if text in ("", "-0") else text


def _scientific(value: float, figures: int = 3) -> str:
    """``1.17\\times10^{-7}`` for math mode."""
    exponent = math.floor(math.log10(abs(value)))
    return f"{value / 10**exponent:.{figures - 1}f}\\times10^{{{exponent}}}"


def sweep_appendix_macros(sweep: dict, root: Path) -> dict[str, tuple[str, str]]:
    runs = sweep["runs"]
    out: dict[str, tuple[str, str]] = {}
    started = sorted(datetime.fromisoformat(r["started"]) for r in runs)
    out["sweepFigureStartDateRange"] = (
        date_range(started[0], started[-1]),
        "first and last run start, UTC",
    )
    out["sweepFigureRunsPerExampleWord"] = (word(sweep["per_example"]), "spelled out")
    for example in sweep["example_order"]:
        prefix = "sweepFigure" + _macro_part(example)
        mine = _runs(sweep, [example])
        counts = sweep["counts"][example]
        for key, value, how in (
            ("RunCountWord", len(mine), "runs"),
            ("ObservablesWord", counts["n_observables"], "training observables"),
            ("ConditionsWord", counts["n_conditions"], "training conditions"),
        ):
            if value < 100:  # the prose spells out only small counts
                out[prefix + key] = (word(value), f"{how}, spelled out")
        starts = {r["n_starts"] for r in mine}
        if len(starts) != 1:
            raise ValueError(f"{example}: runs recorded different multistart budgets {starts}")
        out[prefix + "Starts"] = (count(starts.pop()), "recorded multistart budget")
        defined = [r for r in mine if _topology_defined(r)]
        changed = sum(r["topo_collapsed_normalised"] > 0 for r in defined)
        out[prefix + "TopologyChanged"] = (
            f"{changed}/{len(mine)}" if defined else "--",
            "runs with a nonzero normalised edit distance",
        )
        out[prefix + "TopologyMatchCountWord"] = (
            word(len(defined) - changed),
            "defined runs matching both summaries",
        )
        out[prefix + "TopologyMatchCountWordCap"] = (
            cap(word(len(defined) - changed)),
            "the same, sentence-initial",
        )
    budgets = statistics.multimode(r["n_starts"] for r in runs)
    out["sweepFigureDefaultStarts"] = (count(budgets[0]), "the multistart budget most problems had")
    missing = [r for r in runs if r["n_fits"] is None]
    out.update(
        {
            k: (v, "runs without a recorded fit count" + (f", {how}" if how else ""))
            for k, (v, how) in _counted("sweepFigureMissingFitCount", len(missing)).items()
        }
    )
    out["sweepFigureMissingFitCountProblems"] = (
        listing(
            [display(e) for e in sweep["example_order"] if any(r["example"] == e for r in missing)]
        ),
        "problems they belong to",
    )
    anchored = [r for r in runs if (r["topo_interventions_anchored"] or 0) >= 1]
    declared = [r for r in runs if (r["topo_interventions"] or 0) >= 1]
    for name, sample, how in (
        (
            "sweepFigureTopologyAnchoredCount",
            anchored,
            "runs with an intervention anchored in both models",
        ),
        (
            "sweepFigureTopologyUnanchoredCount",
            [r for r in declared if r not in anchored],
            "runs declaring an intervention without a shared anchor",
        ),
    ):
        out.update({k: (v, how) for k, (v, _) in _counted(name, len(sample)).items()})
    defined = [r for r in runs if _topology_defined(r)]
    above = [r for r in defined if r["topo_collapsed_normalised"] > 1]
    out.update(
        {
            k: (v, "defined comparisons with a normalised total above one")
            for k, (v, _) in _counted("sweepFigureTopologyAboveOneCount", len(above)).items()
        }
    )
    out["sweepFigureTopologyMaxNormalised"] = (
        f"{max(r['topo_collapsed_normalised'] for r in defined):.2f}",
        "largest normalised edit distance",
    )
    for count_name, value in (
        (
            "LargerSameStatesCount",
            sum(r["delta_states"] == 0 for r in _runs(sweep, _sweep_groups(sweep)[1])),
        ),
        (
            "LargerMoreStatesCount",
            sum(r["delta_states"] > 0 for r in _runs(sweep, _sweep_groups(sweep)[1])),
        ),
    ):
        out["sweepFigure" + count_name + "Word"] = (word(value), "spelled out")

    names = model_names(sweep)
    for run in runs:
        prefix = _model_macro(names[run["task"]])
        out[prefix + "DeltaBIC"] = (signed(run["delta_bic"], 2), f"{names[run['task']]}: ΔBIC")
        out[prefix + "States"] = (str(run["n_states"]), "states")
        out[prefix + "StatesWord"] = (word(run["n_states"]), "states, spelled out")
        out[prefix + "ReferenceStatesWord"] = (
            word(run["reference_n_states"]),
            "the reference's states, spelled out",
        )
        out[prefix + "FewerStatesWord"] = (
            word(max(0, -run["delta_states"])),
            "states fewer than the reference, spelled out",
        )
        size = (
            run["topo_collapsed_distance"] / run["topo_collapsed_normalised"]
            if run["topo_collapsed_normalised"]
            else None
        )
        for key, name in (
            ("topo_new_connections", "NewConnections"),
            ("topo_extra_depth", "ExtraDepth"),
            ("topo_deleted_connections", "DeletedConnections"),
            ("topo_removed_depth", "RemovedDepth"),
        ):
            out[prefix + name] = (_component(run[key]), "normalised edit-distance component")
        if size and run["topo_extra_depth"] is not None:
            # The export stores the components rounded, so the product is an integer only up to
            # that rounding; a count this cannot recover exactly is not emitted.
            hops = run["topo_extra_depth"] * size
            if abs(hops - round(hops)) < 0.05 and round(hops) < 100:
                out[prefix + "ExtraHopsWord"] = (
                    word(round(hops)),
                    "extra hops: the depth component times the collapsed reference size",
                )
    tasks = {name: task for task, name in names.items()}
    for name in DISCUSSED_MODELS:
        prefix = _model_macro(name)
        with (root / "paper/example_models" / f"{name}.parameters.tsv").open(newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                value = float(row["nominalValue"])
                key = prefix + "Param" + _macro_part(row["parameterId"])
                if key in out:
                    raise ValueError(f"{name}: two parameters both become {key}")
                out[key] = (_raw(value), f"{name} ({tasks[name]}): fitted {row['parameterId']}")
                if value:
                    out[key + "Scientific"] = (_scientific(value), "the same, three figures")
    return out


def trajectory_macros(trajectories: dict, root: Path) -> dict[str, tuple[str, str]]:
    """The supplementary best-so-far figure: runs whose delivered version was not their best."""
    late = {}
    for example, runs in trajectories["figure2"].items():
        for run in runs:
            if run["points"] and run["final_dbic"] > min(p[1] for p in run["points"]) + 1e-9:
                late.setdefault(example, []).append(run)
    most = max(late, key=lambda example: len(late[example]))
    total = sum(map(len, late.values()))
    return {
        "bicTrajectoriesNonBestFinalCount": (
            str(total),
            "Figure 2 runs that ended on a version other than their best",
        ),
        "bicTrajectoriesNonBestFinalCountWord": (word(total), "the same, spelled out"),
        "bicTrajectoriesNonBestFinalMostProblem": (display(most), "the problem with the most"),
        "bicTrajectoriesNonBestFinalMostCountWord": (
            word(len(late[most])),
            "how many, spelled out",
        ),
        "bicTrajectoriesLinearThreshold": (
            number(
                module_constant(root, "src/petab_harness/paper_bic_trajectories.py", "LINTHRESH")
            ),
            "the symmetric-log axis's linear range",
        ),
    }


#: The Figure 3 models whose fitted parameters the annex prose quotes.
DISCUSSED_FUJITA_MODELS = ("O1d",)


def fujita_appendix_macros(
    metadata: dict, rows: list[dict], configurations: dict, root: Path
) -> dict[str, tuple[str, str]]:
    arms = _fujita_arms(metadata)
    code = {arm: spec["code"] for arm, spec in arms.items()}
    evaluation = metadata["evaluation"]
    out: dict[str, tuple[str, str]] = {}
    experiments = metadata["source"]["experiments"]
    out["fujitaFigureExperimentCountWord"] = (word(len(experiments)), "experiments, spelled out")
    for experiment in experiments:
        name = _macro_part(experiment["key"].rsplit("-", 1)[0])
        out[f"fujitaFigure{name}MaxSteps"] = (
            count(experiment["evaluation"]["max_steps"]),
            f"{experiment['key']}: integrator step limit",
        )
    out["fujitaFigureMaxSteps"] = (count(evaluation["max_steps"]), "the manifest's step limit")
    runs = [r for r in configurations["runs"] if r["cohort"] == "figure3"]
    out["fujitaFigureRevisionCountWord"] = (
        word(len({r["git_sha"] for r in runs})),
        "run-code revisions, spelled out",
    )
    for key, name, formatter in (
        ("fit_workers", "FitWorkers", str),
        ("n_starts", "FitStarts", str),
        ("per_start_timeout_s", "PerStartSeconds", str),
        ("max_wall_seconds", "WallSeconds", count),
    ):
        values = {r[key] for r in runs}
        if len(values) != 1:
            raise ValueError(f"Figure 3 runs do not share one {key}: {values}")
        out["fujitaFigure" + name] = (formatter(values.pop()), f"every Figure 3 run's {key}")
    analysts = {tuple(r["subagents"]) for r in runs if r["arm"] == "O2"}
    if len(analysts) != 1:
        raise ValueError(f"O2 runs seated different panels: {analysts}")
    out["fujitaFigureAnalystCountWord"] = (
        word(len(analysts.pop())),
        "the analysts a curator-free panel keeps, spelled out",
    )
    bare = {
        row["head_version"] for row in rows if arms[row["arm"]]["factors"]["driver"] == "barebones"
    }
    if len(bare) != 1:
        raise ValueError(f"bare-agent rows carry several head versions: {bare}")
    out["fujitaFigureBareHeadVersion"] = (str(bare.pop()), "the CSV's version for a bare agent")
    out["fujitaFigureBarebonesArmCountWord"] = (
        word(sum(spec["factors"]["driver"] == "barebones" for spec in arms.values())),
        "bare-agent arms, spelled out",
    )
    out["fujitaFigureAbsoluteTolerancePower"] = (power(evaluation["atol"]), "held-out atol")
    out["fujitaFigureRelativeTolerancePower"] = (power(evaluation["rtol"]), "held-out rtol")
    out["fujitaFigureConditionCountWord"] = (
        word(evaluation["n_conditions"]),
        "held-out conditions, spelled out",
    )
    out["fujitaFigureNoModelCountWord"] = (
        word(len(metadata["no_model_runs"])),
        "runs without a model, spelled out",
    )
    out["fujitaFigureCompleteCount"] = (
        str(sum(r["complete_heldout"] for r in rows)),
        "models with a complete held-out score",
    )
    canonical = metadata["training_dbic_renormalisation"]["canonical_reference"]
    out["fujitaFigureTrainingReferenceNLLExact"] = (
        repr(canonical["nll"]),
        "the reference's training NLL, as stored",
    )
    out["fujitaFigureReferenceNLLExact"] = (
        repr(metadata["reference"]["heldout_nll"]),
        "the reference's held-out NLL, as stored",
    )
    out["fujitaFigureReferenceK"] = (str(canonical["k"]), "the reference's parameter count")
    out["fujitaFigureReferenceN"] = (str(canonical["n"]), "its training measurements")
    out["fujitaFigureReferenceEdgeCountWord"] = (
        word(len(metadata["reference_network"]["edges"])),
        "reference readout connections, spelled out",
    )
    pools = {
        "OpenWeight": [r for r in rows if code[r["arm"]].startswith("O")],
        "Frontier": [r for r in rows if code[r["arm"]].startswith("F")],
    }
    for family, sample in pools.items():
        pairs = [
            (r["training_dbic"], r["heldout_delta_nll"]) for r in sample if r["complete_heldout"]
        ]
        out[f"fujitaFigure{family}DBICHeldoutSpearmanPrecise"] = (
            decimal(_spearman(pairs), 2),
            "the same correlation, two decimals",
        )
        out[f"fujitaFigure{family}DBICHeldoutSpearmanNWord"] = (word(len(pairs)), "spelled out")
    out["fujitaFigureCostQuotedDate"] = (
        date(metadata["cost_model"]["cpu"]["quoted"]),
        "the CPU price's quote date",
    )
    for arm, spec in arms.items():
        prefix = "fujitaFigureArm" + paper_fujita.macro_name(spec["code"])
        out[prefix + "NoModelWord"] = (
            word(spec["n_no_model"]),
            "runs without a model, spelled out",
        )
    deltas = metadata["validation_deltas"]
    systems = [entry["example"] for entry in metadata["validation_examples"]]
    out["fujitaFigureValidationExampleCountWord"] = (word(len(systems)), "spelled out")
    per_system = {len([d for d in deltas if d["example"] == s]) for s in systems}
    if len(per_system) != 1:
        raise ValueError(f"validation systems have different run counts: {per_system}")
    out["fujitaFigureValidationRunsPerExampleWord"] = (word(per_system.pop()), "spelled out")
    improved = {s: sum(d["delta_nll"] < 0 for d in deltas if d["example"] == s) for s in systems}
    for system, n in improved.items():
        prefix = "fujitaValidation" + display(system)
        out.update(
            {
                prefix + "ImprovedCount" + k[len("x") :]: (
                    v,
                    "candidates with ΔNLL < 0" + (f", {how}" if how else ""),
                )
                for k, (v, how) in _counted("x", n).items()
            }
        )
    out["fujitaValidationNoneImprovedExamples"] = (
        alternatives([display(s) for s in sorted(systems) if not improved[s]]),
        "systems where no candidate improved on the reference",
    )
    # Hops from pAkt to pS6: the reference's, and each model's increment over it.
    downstream = next(
        edge["hops"]
        for edge in metadata["reference_network"]["edges"]
        if "pAkt" in edge["source"] and "pS6" in edge["target"]
    )
    out["fujitaFigureReferenceDownstreamHopsWord"] = (
        word(downstream),
        "the reference's pAkt-to-pS6 hops",
    )
    for row in rows:
        depth = row["downstream_depth"]
        if isinstance(depth, int) and depth >= 0:
            prefix = "fujitaModel" + paper_fujita.macro_name(_name(metadata, row))
            out[prefix + "DownstreamDepthWord"] = (word(depth), "pAkt-to-pS6 hops, spelled out")
            if depth >= downstream:
                out[prefix + "DownstreamIncrementWord"] = (
                    word(depth - downstream),
                    "hops beyond the reference's",
                )
    for label in DISCUSSED_FUJITA_MODELS:
        prefix = "fujitaModel" + paper_fujita.macro_name(label)
        path = root / "paper/fujita_models" / f"{label}.parameters.tsv"
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                value = float(row["nominalValue"])
                key = prefix + "Param" + _macro_part(row["parameterId"])
                if key in out:
                    raise ValueError(f"{label}: two parameters both become {key}")
                out[key] = (_raw(value), f"{label}: fitted {row['parameterId']}")
                if value:
                    out[key + "Scientific"] = (_scientific(value), "the same, three figures")
    kinds = [
        paper_fujita_models.source_kind({r["task"]: r for r in metadata["runs"]}[row["task"]])
        for row in paper_fujita_models.annexed(metadata, rows)
    ]
    for kind, name in (("archived", "Archived"), ("converted", "Converted")):
        out.update(
            {
                f"fujitaAnnex{name}Count" + k[1:]: (v, f"annexed listings that are {kind}")
                for k, (v, _) in _counted("x", kinds.count(kind)).items()
            }
        )
    return out


def heldout_appendix_macros(heldout: dict, root: Path) -> dict[str, tuple[str, str]]:
    out: dict[str, tuple[str, str]] = {}
    distilled = 0
    for entry in heldout["examples"]:
        example = entry["example"]
        prefix = "heldoutFigure" + display(example)
        for arm, name in (
            ("regular", "Visible"),
            ("blinded", "Blinded"),
            ("distilled", "Distilled"),
        ):
            if arm in entry["arms"]:
                scored = len(entry["arms"][arm]["points"])
                out[prefix + name + "RunCount"] = (str(scored), f"{name.lower()} runs scored")
                out[prefix + name + "RunCountWord"] = (word(scored), "the same, spelled out")
        if "distilled" in entry["arms"]:
            distilled += len(entry["arms"]["distilled"]["points"])
        units = root / "examples" / example / "validation_source" / "units.tsv"
        if units.is_file():
            with units.open(newline="") as handle:
                n = len({row["unitId"] for row in csv.DictReader(handle, delimiter="\t")})
            out.update(
                {
                    prefix + "UnitCount" + k[1:]: (v, "held-out units (refit separately)")
                    for k, (v, _) in _counted("x", n).items()
                }
            )
    out.update(
        {
            "heldoutFigureDistilledPointCount" + k[1:]: (v, "neural-ODE points Figure 4 plots")
            for k, (v, _) in _counted("x", distilled).items()
        }
    )
    blinded = sum(
        len(e["arms"]["blinded"]["points"]) for e in heldout["examples"] if "blinded" in e["arms"]
    )
    out.update(
        {
            "heldoutFigureBlindedRunTotal" + k[1:]: (v, "biology-blinded runs Figure 4 scores")
            for k, (v, _) in _counted("x", blinded).items()
        }
    )
    return out


def combined_macros(sweep: dict, fujita: tuple) -> dict[str, tuple[str, str]]:
    """Numbers that need two figures' inputs."""
    metadata, _ = fujita
    declared = sweep["counts"]["fujita"]["n_conditions"] + metadata["evaluation"]["n_conditions"]
    return {
        "fujitaFigureDeclaredConditionCountWord": (
            word(declared),
            "Fujita's conditions: the training ones plus the held-out ones",
        )
    }


def prompt_macros(root: Path) -> dict[str, tuple[str, str]]:
    manifest = json.loads((root / "paper/prompts/manifest.json").read_text(encoding="utf-8"))
    analysts = [f for f in manifest["files"] if Path(f["file"]).name.startswith("analyst_")]
    out = {"promptFileCount": (str(len(manifest["files"])), "prompt files the export ships")}
    out.update(
        {
            "promptAnalystCount" + k[1:]: (v, "read-only analyst prompts")
            for k, (v, _) in _counted("x", len(analysts)).items()
        }
    )
    return out


def neural_grid_macros(grid: dict) -> dict[str, tuple[str, str]]:
    out = {"neuralGridCellCount": (str(grid["n_cells"]), "cells in the grid")}
    cells = {spec["n_cells"] for spec in grid["examples"].values()}
    if len(cells) != 1:
        raise ValueError(f"benchmarks have different grid sizes: {cells}")
    out["neuralGridCellsPerExample"] = (str(cells.pop()), "cells per benchmark")
    for axis, name in (
        ("width", "Width"),
        ("n_augmented", "Augmented"),
        ("multiplicative", "Multiplicative"),
        ("horizon_stages", "Horizon"),
        ("l21_weight", "Sparsity"),
    ):
        sizes = {len(spec["axes"][axis]) for spec in grid["examples"].values()}
        if len(sizes) != 1:
            raise ValueError(f"{axis}: benchmarks have different numbers of values")
        out[f"neuralGrid{name}CountWord"] = (word(sizes.pop()), f"values of {axis}, spelled out")
    stages = sorted(
        {s for spec in grid["examples"].values() for s in spec["axes"]["horizon_stages"]}
    )
    out["neuralGridHorizonStagesWords"] = (
        alternatives([word(s) for s in stages]),
        "horizon stages, spelled out",
    )
    request = grid["launch"]["request"]
    out["neuralGridCPUsWord"] = (word(int(request["cpus-per-task"])), "CPUs per cell, spelled out")
    out["neuralGridMemoryGB"] = (request["mem"].rstrip("G"), "memory per cell, GB")
    hours_, minutes, _ = (int(x) for x in request["time"].split(":"))
    out["neuralGridTimeHours"] = (
        str(hours_ + minutes / 60).removesuffix(".0"),
        "scheduler time limit per cell, hours",
    )
    out["neuralGridRestartsWord"] = (
        word(grid["fixed"]["train"]["n_restarts"]),
        "training restarts, spelled out",
    )
    out["neuralGridJudgeStarts"] = (str(grid["judge"]["n_starts"]), "the judge's refit starts")
    budgets = grid["proposals_per_cell"]["complexity_budgets"]
    out["neuralGridComplexityMin"] = (str(min(budgets)), "smallest distilled-expression size")
    out["neuralGridComplexitySecond"] = (str(sorted(budgets)[1]), "the next size")
    out["neuralGridComplexityMax"] = (str(max(budgets)), "largest distilled-expression size")
    return out


def code_checks(root: Path) -> list[Claim]:
    """What the prose says about the code that no number carries."""
    server = _MCP + "server.py"
    provenance = signature_default(root, server, "get_provenance", "limit")
    errors = signature_default(root, server, "get_errors", "limit")
    fraction = module_constant(root, _MCP + "fitting.py", "WARM_START_FRACTION")
    return [
        Claim(
            "mcp-tools",
            "get_provenance and get_errors share one default limit",
            provenance == errors,
            f"{provenance} vs {errors}",
        ),
        Claim(
            "implementation",
            "warm starts take the floor of half the starts",
            fraction == 0.5,
            f"WARM_START_FRACTION = {fraction}",
        ),
    ]


def spread_macros(summary: dict, root: Path) -> dict[str, tuple[str, str]]:
    """The Fujita optimum-spread appendix, from ``paper/fujita_spread/summary.json``."""
    starts = summary["n_starts"]
    if len(starts) != 1:
        raise ValueError(f"the spread study used several multistart sizes: {starts}")
    shift = summary["tolerance_shift"]
    level = float(summary["band_rule"].rsplit(":", 1)[1])
    capped = [f"{c['name']} {c['kept']} of {c['in_band']}" for c in summary["capped"]]
    unscored = [f"{u['count']} in {u['name']}" for u in summary["unscored_members"]]
    out = {
        "fujitaSpreadStarts": (str(starts[0]), "multistart size per model"),
        "fujitaSpreadConvergedLow": (str(summary["converged"][0]), "fewest converged starts"),
        "fujitaSpreadConvergedHigh": (str(summary["converged"][1]), "most converged starts"),
        "fujitaSpreadMaxScored": (
            str(summary["max_scored"]),
            "band members scored per model, at most",
        ),
        "fujitaSpreadCapped": (listing(capped), "bands the cap truncated: kept of in band"),
        "fujitaSpreadCappedCountWord": (word(len(capped)), "how many, spelled out"),
        "fujitaSpreadUnscoredCount": (
            str(sum(u["count"] for u in summary["unscored_members"])),
            "scored band members without a held-out value",
        ),
        "fujitaSpreadUnscored": (listing(unscored), "their models, with counts"),
        "fujitaSpreadBlowupFactorWord": (
            word(summary["blowup_factor"]),
            "peak over the largest held-out measurement that counts as a blow-up, spelled out",
        ),
        "fujitaSpreadBlowups": (str(summary["blowups"]), "band members that blew up"),
        "fujitaSpreadBlowupsRejected": (
            str(summary["blowups_rejected"]),
            "of which outside the tight band",
        ),
        "fujitaSpreadBlowupSurvivors": (
            listing(sorted(set(summary["blowup_survivors"]))),
            "models of the ones inside it",
        ),
        "fujitaSpreadShiftMax": (
            _scientific(shift["max"], 2),
            "largest training-NLL change of an archived optimum between the "
            "fit and the tight tolerance",
        ),
        "fujitaSpreadShiftMedian": (_scientific(shift["median"], 2), "the median change"),
        "fujitaSpreadShiftModels": (str(shift["n"]), "models with an archived optimum re-scored"),
        "fujitaSpreadShiftThreshold": (
            power(shift["threshold"]),
            "the change counted as not reproducing",
        ),
        "fujitaSpreadShiftAboveCount": (str(shift["above_threshold"]), "of which changed by more"),
        "fujitaSpreadHeldoutReproductionMax": (
            _scientific(summary["heldout_reproduction_max"], 2),
            "largest held-out NLL difference between an archived optimum and Figure 3",
        ),
        "fujitaSpreadBandLevelPercent": (
            number(round(100 * level, 6)),
            "the band's chi-square level",
        ),
        "fujitaSpreadMovedDecades": (
            number(module_constant(root, "slurm/_plot_spread_two_panel_tol.py", "MOVED")),
            "coordinate range below which a member is the same point re-found",
        ),
    }
    return out


def appendix_checks(sweep: dict | None, fujita: tuple | None) -> list[Claim]:
    out = []
    if sweep:
        budgets = {
            e: {r["n_starts"] for r in _runs(sweep, [e])}.pop() for e in sweep["example_order"]
        }
        default = statistics.multimode(r["n_starts"] for r in sweep["runs"])[0]
        odd = sorted(e for e, n in budgets.items() if n != default)
        out.append(
            Claim(
                "evaluation",
                "Only Crauste and Bachmann depart from the default budget",
                odd == ["bachmann", "crauste"],
                ", ".join(odd),
            )
        )
    if fujita:
        metadata, rows = fujita
        deltas = metadata["validation_deltas"]
        improved = sorted({d["example"] for d in deltas if d["delta_nll"] < 0})
        out.append(
            Claim(
                "evaluation",
                "Perelson is the only validation system with an improvement",
                improved == ["perelson"],
                ", ".join(improved),
            )
        )
    return out


# ------------------------------------------------------------------------------ assembly


def macros(
    root: Path,
    sweep: dict | None,
    fujita: tuple | None,
    heldout: dict | None,
    configurations: dict | None,
    grid: dict | None = None,
    trajectories: dict | None = None,
    spread: dict | None = None,
) -> dict[str, tuple[str, str]]:
    out: dict[str, tuple[str, str]] = {}

    def merge(new):
        clash = set(out) & set(new)
        if clash:
            raise ValueError(f"macros defined twice: {sorted(clash)}")
        out.update(new)

    if sweep and configurations:
        merge(sweep_macros(sweep, configurations))
    if sweep:
        merge(sweep_appendix_macros(sweep, root))
    if fujita and configurations:
        merge(fujita_macros(*fujita, configurations))
        merge(fujita_appendix_macros(*fujita, configurations, root))
    if heldout:
        merge(heldout_macros(heldout))
        merge(heldout_appendix_macros(heldout, root))
    if sweep and fujita:
        merge(combined_macros(sweep, fujita))
    if trajectories:
        merge(trajectory_macros(trajectories, root))
    if grid:
        merge(neural_grid_macros(grid))
    if spread:
        merge(spread_macros(spread, root))
    merge(prompt_macros(root))
    merge(code_macros(root))
    return out


def spread_checks(summary: dict) -> list[Claim]:
    return [
        Claim(
            "fujita-generalisation",
            "Every blow-up band member falls outside the tight band",
            summary["blowups_rejected"] == summary["blowups"],
            f"{summary['blowups_rejected']}/{summary['blowups']}",
        ),
        # "every band member of every model" -- the re-scoring cap truncated no band.
        Claim(
            "fujita-generalisation",
            "Every band member was re-scored: no band was capped",
            not summary["capped"],
            f"{len(summary['capped'])} capped",
        ),
    ]


def checks(
    sweep: dict | None, fujita: tuple | None, root: Path | None = None, spread: dict | None = None
) -> list[Claim]:
    out = []
    if sweep:
        out += sweep_checks(sweep)
    if fujita:
        out += fujita_checks(*fujita)
    if spread:
        out += spread_checks(spread)
    return out + appendix_checks(sweep, fujita) + (code_checks(root) if root else [])
