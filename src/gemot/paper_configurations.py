"""The agent runs' configurations and the neural-ODE baseline's grid, as the annex's tables.

Two frozen inputs, each built from the runs themselves by a ``slurm/`` script:

* ``paper/run_configurations.json`` (``_run_configurations_input.py``) -- one record per agent
  run that figures 2-4 score: model and provider pin, harness, panel, prompt and commit, the
  fit and sampling settings, the reasoning setting sent (none: each model's provider default,
  which the input records as OpenRouter lists it), and the Slurm budget the run was given.
* ``paper/neural_ode_grid.json`` (``_neural_ode_grid_input.py``) -- grid3's axes, the settings
  no axis varies as the trainer and distiller recorded them, and the launch.

:func:`export_data` renders them into ``paper/generated``: three ``tabular`` s for the annex
(what each cohort and arm ran with, the per-benchmark budgets, the grid) and a CSV per input with
every run and every cell. A value is shown as the runs recorded it; a table cell that lists
several values means the runs differed, and the CSV says which run had which.
"""

from __future__ import annotations

import csv
import io
import json
from collections import Counter
from pathlib import Path

#: The paper's names for the language models.
MODEL_NAMES = {"z-ai/glm-5.3-flash": "GLM-5.3-Flash", "openai/gpt-6-astra": "GPT-6-Astra"}
#: Table order: figure 2's single arm, figure 3's ladder, figure 4's blinded cohort.
COHORTS = (("figure2", "Figure 2"), ("figure3", "Figure 3"), ("figure4_blinded", "Figure 4"))
RUN_COLUMNS = (
    "cohort",
    "arm",
    "example",
    "run",
    "archive_id",
    "task",
    "model",
    "provider_pin",
    "served_by",
    "harness",
    "driver",
    "biocurator",
    "biology",
    "subagents",
    "ablation_arm",
    "ablations",
    "git_sha",
    "example_version",
    "prompt_version",
    "skills_version",
    "temperature",
    "recursion_limit",
    "max_output_tokens",
    "criterion",
    "bic_threshold",
    "n_starts",
    "per_start_timeout_s",
    "max_wall_seconds",
    "fit_workers",
    "budget_source",
    "disabled_tools",
    "omnipath_snapshot",
    "stop_reason",
    "log",
    # Everything else the record carries, so the CSV holds every field it says it does.
    "execute_timeout_s",
    "sandbox",
    "prompt_group_versions",
    "hardcoded_constants",
    "reasoning_sent",
    "reasoning_runaways",
    "reasoning_runaway_recovered",
)


def read_inputs(root: Path) -> tuple[dict | None, dict | None]:
    out = []
    for name in ("run_configurations.json", "neural_ode_grid.json"):
        path = Path(root) / "paper" / name
        payload = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
        if payload is not None and payload.get("schema_version") != 1:
            raise ValueError(f"unsupported schema in {name}; rebuild it")
        out.append(payload)
    return out[0], out[1]


def tex(text) -> str:
    return (
        str(text)
        .replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("#", r"\#")
    )


def listed(values, width=7, code=True) -> str:
    """Distinct values in order of frequency, a count after each when there are several."""
    counts = Counter(values)
    parts = []
    for value, n in counts.most_common():
        shown = "---" if value in (None, "") else str(value)[:width] if width else str(value)
        shown = rf"\texttt{{{tex(shown)}}}" if code and value not in (None, "") else tex(shown)
        parts.append(shown + (f" ({n})" if len(counts) > 1 else ""))
    return ", ".join(parts)


def hours(seconds) -> str:
    return "---" if seconds is None else f"{seconds / 3600:.3g}"


def support(run: dict) -> str:
    """Figure 3's ladder label for what a run was given: ``curator + biology + gemot`` ..., with
    the harness named as the figure prints it (``paper_fujita.support_label``)."""
    from gemot.paper_fujita import support_label

    parts = (
        (["curator"] if run["biocurator"] == "on" else [])
        + (["biology"] if run["biology"] == "visible" else [])
        + [run["harness"]]
    )
    return support_label(" + ".join(parts))


def _group(label: str, columns: int) -> str:
    return rf"\multicolumn{{{columns}}}{{l}}{{\emph{{{label}}}}} \\"


def render_cohorts(payload: dict, header: str) -> str:
    """One row per cohort and arm: which model, what it was given, which prompts and commit."""
    rows = []
    for key, figure in COHORTS:
        runs = [r for r in payload["runs"] if r["cohort"] == key]
        examples = len({r["example"] for r in runs})
        rows += [
            r"\midrule",
            _group(
                f"{figure}: {len(runs)} runs"
                + (f" over {examples} benchmarks" if examples > 1 else ""),
                6,
            ),
        ]
        for arm in sorted(dict.fromkeys(r["arm"] for r in runs)):
            mine = [r for r in runs if r["arm"] == arm]
            rows.append(
                " & ".join(
                    [
                        arm,
                        listed(
                            [MODEL_NAMES.get(r["model"], r["model"]) for r in mine],
                            width=0,
                            code=False,
                        ),
                        listed([support(r) for r in mine], width=0, code=False),
                        listed([r["prompt_version"] for r in mine]),
                        listed([r["git_sha"] for r in mine]),
                        str(len(mine)),
                    ]
                )
                + r" \\"
            )
    return (
        "\n".join(
            [
                header,
                "% One row per cohort and arm of the agent runs figures 2-4 score, from",
                "% paper/run_configurations.json; a count after a value means the runs differed.",
                "% Given: the support ladder of figure 3 (curator, biology context, harness), as the run",
                "% ran it. Prompt set: the run's prompt_version, a hash over its whole prompt registry.",
                # The commit list wraps: figure 2's runs span three checkouts.
                r"\begin{tabular}{llllp{1.3in}r}",
                r"\toprule",
                r"Arm & Language model & Given & Prompt set & Commit & Runs \\",
                *[row for row in rows if row],
                r"\bottomrule",
                r"\end{tabular}",
            ]
        )
        + "\n"
    )


def render_budgets(payload: dict, header: str) -> str:
    """One row per benchmark and cohort: the fit and wall budget each run was given."""
    rows = []
    for key, figure in COHORTS:
        runs = [r for r in payload["runs"] if r["cohort"] == key]
        groups = (
            [("all arms", runs)]
            if key == "figure3"
            else [
                (example, [r for r in runs if r["example"] == example])
                for example in dict.fromkeys(r["example"] for r in runs)
            ]
        )
        rows += [r"\midrule", _group(figure, 5)]
        for name, mine in groups:
            rows.append(
                " & ".join(
                    [
                        tex(name),
                        listed([r["n_starts"] for r in mine], width=0, code=False),
                        listed([r["per_start_timeout_s"] for r in mine], width=0, code=False),
                        listed([hours(r["max_wall_seconds"]) for r in mine], width=0, code=False),
                        listed([r["fit_workers"] for r in mine], width=0, code=False),
                    ]
                )
                + r" \\"
            )
    return (
        "\n".join(
            [
                header,
                "% Budgets as each run's Slurm log states them (the null-arms experiment of figure 3,",
                "% which kept no logs, from its manifest); paper/run_configurations.json.",
                r"\begin{tabular}{lrrrr}",
                r"\toprule",
                r"Benchmark & Starts & Per-start cap (s) & Wall budget (h) & Fit workers \\",
                *[row for row in rows if row],
                r"\bottomrule",
                r"\end{tabular}",
            ]
        )
        + "\n"
    )


def render_shared(payload: dict, header: str) -> str:
    """The run settings, stated once: a count after a value says how many runs had it."""

    def snapshot(run):
        return (run.get("omnipath_snapshot") or {}).get("fetched_at")

    sent = {"none": "none (provider default)"}
    fields = (
        ("Sampling temperature", lambda r: r.get("temperature")),
        ("Agent step limit", lambda r: r.get("recursion_limit")),
        ("Output tokens per call", lambda r: r.get("max_output_tokens")),
        (
            "Reasoning effort requested",
            lambda r: sent.get(r.get("reasoning_sent"), r.get("reasoning_sent")),
        ),
        ("Calls re-issued at low effort", lambda r: r.get("reasoning_runaways")),
        ("Selection criterion", lambda r: r.get("criterion")),
        (r"$\Delta$BIC threshold", lambda r: r.get("bic_threshold")),
        ("OmniPath snapshot", snapshot),
        ("Code-execution timeout (s)", lambda r: r.get("execute_timeout_s")),
    )

    # Sorted before counting, so values the same number of runs share read in order (1, 2, 3).
    def order(value):
        return (
            value is None,
            not isinstance(value, (int, float)),
            value if isinstance(value, (int, float)) else str(value),
        )

    rows = []
    for label, get in fields:
        values = sorted((get(r) for r in payload["runs"]), key=order)
        rows.append(f"{label} & {listed(values, width=0, code=False)} \\\\")
    pins = sorted(
        {
            (
                MODEL_NAMES.get(r["model"], r["model"]),
                ", ".join(r["provider_pin"] or []),
                ", ".join(r["served_by"] or []),
            )
            for r in payload["runs"]
        }
    )
    rows += [
        f"Provider, {tex(model)} & {tex(pin) or '---'} (served by {tex(served) or '---'}) \\\\"
        for model, pin, served in pins
    ]
    # What "provider default" was, per model: a property of the model, frozen with its date.
    reasoning = payload.get("provider_reasoning") or {}
    for model_id, block in sorted(
        reasoning.get("models", {}).items(), key=lambda item: MODEL_NAMES.get(item[0], item[0])
    ):
        levels = ", ".join(reversed(block.get("supported_efforts") or []))
        rows.append(
            f"Default reasoning, {tex(MODEL_NAMES.get(model_id, model_id))} & "
            f"{'on' if block.get('default_enabled') else 'off'}, "
            f"{tex(block.get('default_effort'))} effort (of {tex(levels)}) \\\\"
        )
    if reasoning:
        rows.append(
            f"Defaults as listed by & OpenRouter model metadata, "
            f"{tex(reasoning.get('fetched_at'))} \\\\"
        )
    return (
        "\n".join(
            [
                header,
                "% Settings recorded by every run; a count after a value says how many runs had it",
                "% (--- = not recorded); paper/run_configurations.json.",
                r"\begin{tabularx}{\linewidth}{lX}",
                r"\toprule",
                r"Setting & Value \\",
                r"\midrule",
                *rows,
                r"\bottomrule",
                r"\end{tabularx}",
            ]
        )
        + "\n"
    )


def _number(value) -> str:
    """``3.2\times10^{-3}`` for a small value, ``0.03`` or ``1`` otherwise; never rounded."""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if value and abs(value) < 1e-2:
            mantissa, exponent = f"{value:e}".split("e")
            return rf"${float(mantissa):g}\times10^{{{int(exponent)}}}$"
        return f"{value:g}"
    return tex(value)


def render_grid(grid: dict, header: str) -> str:
    """The grid's axes (per benchmark where they differ), then what no axis varies."""
    labels = {
        "width": "Hidden width",
        "n_augmented": "Augmented states",
        "multiplicative": "Multiplicative field",
        "horizon_stages": "Horizon stages",
        "l21_weight": r"$L_{2,1}$ weight (0 = none)",
    }
    examples = list(grid["examples"])
    rows = []
    for axis in ("width", "n_augmented", "multiplicative", "horizon_stages", "l21_weight"):
        per = {e: grid["examples"][e]["axes"].get(axis) for e in examples}
        distinct = {json.dumps(v) for v in per.values()}
        if len(distinct) == 1:
            values = ", ".join(_number(v) for v in next(iter(per.values())))
        else:
            values = "; ".join(
                f"{tex(e)} " + ", ".join(_number(v) for v in per[e]) for e in examples
            )
        rows.append(f"{labels[axis]} & {values} \\\\")
    train, distil = grid["recorded"]["train"], grid["recorded"]["distil"]
    fixed = [
        ("Network depth", train.get("depth")),
        ("Optimiser", str(train.get("optimiser")).capitalize()),
        ("Learning rate", train.get("learning_rate")),
        ("Steps", train.get("steps")),
        ("Restarts", train.get("n_restarts")),
        ("Patience (steps)", train.get("patience")),
        (
            "Solver tolerances (rel., abs.)",
            f"{_number(train.get('rtol'))}, {_number(train.get('atol'))}",
        ),
        ("Solver step limit", train.get("max_solver_steps")),
        ("Shooting windows", train.get("n_windows")),
        ("Seed", train.get("seed")),
        ("Wall time per cell (s)", grid["fixed"].get("cell_seconds")),
        ("Symbolic regression iterations", distil.get("niterations")),
        ("Expression size limit", distil.get("maxsize")),
        ("Population size", distil.get("population_size")),
        ("Parsimony", distil.get("parsimony")),
        (
            "Operators",
            "$"
            + ", ".join(
                {"*": r"\times", "/": r"\div"}.get(o, o)
                for o in distil.get("binary_operators") or []
            )
            + "$",
        ),
        ("Samples of the field", distil.get("n_samples")),
        (
            "Proposals per cell",
            f"{grid['proposals_per_cell']['n']} (complexity budgets "
            f"{grid['proposals_per_cell']['complexity_budgets'][0]}--"
            f"{grid['proposals_per_cell']['complexity_budgets'][-1]}, "
            "and the distiller's own selection)",
        ),
        ("Judge fit starts", grid["judge"].get("n_starts")),
    ]
    rows += [r"\midrule"] + [
        f"{label} & {value if isinstance(value, str) and '$' in value else _number(value)} \\\\"
        for label, value in fixed
    ]
    launch, code = grid["launch"]["request"], grid["code"]
    return (
        "\n".join(
            [
                header,
                f"% grid3: {grid['n_cells']} cells, the full product of the axes above the rule for each",
                "% benchmark; below it, settings no axis varies, as the trainer and distiller recorded",
                f"% them. Launched on {launch.get('partition')} with {launch.get('cpus-per-task')} cpus, "
                f"{launch.get('mem')}, {launch.get('time')} per cell;",
                f"% code {str(code.get('commit'))[:7]} ({code.get('branch')}). paper/neural_ode_grid.json.",
                r"\begin{tabularx}{\linewidth}{lX}",
                r"\toprule",
                r"Setting & Values \\",
                r"\midrule",
                *rows,
                r"\bottomrule",
                r"\end{tabularx}",
            ]
        )
        + "\n"
    )


def _csv(rows: list[dict], columns) -> str:
    handle = io.StringIO()
    writer = csv.writer(handle, lineterminator="\n")
    writer.writerow(columns)
    for row in rows:
        writer.writerow(
            [
                ";".join(map(str, v))
                if isinstance(v, list)
                else ";".join(f"{k}={x}" for k, x in v.items())
                if isinstance(v, dict)
                else ""
                if v is None
                else v
                for v in (row.get(c) for c in columns)
            ]
        )
    return handle.getvalue()


def export_data(runs: dict | None, grid: dict | None, header: str) -> dict[str, str]:
    files = {}
    if runs is not None:
        files.update(
            {
                "run_configurations.tex": render_cohorts(runs, header),
                "run_budgets.tex": render_budgets(runs, header),
                "run_shared_settings.tex": render_shared(runs, header),
                "run_configurations.csv": _csv(runs["runs"], RUN_COLUMNS),
            }
        )
    if grid is not None:
        axes = ["example", "cell", *grid["axes"]]
        files.update(
            {
                "neural_ode_grid.tex": render_grid(grid, header),
                "neural_ode_grid.csv": _csv(grid["cells"], axes),
            }
        )
    return files
