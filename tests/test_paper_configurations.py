"""The annex tables of what the agent runs and the neural-ODE grid ran with.

Rendered from synthetic inputs, so the tests pin the table logic rather than today's runs: runs
that agree on a setting show it once, runs that differ show every value with its count, the
harness appears under the paper's names, and the grid's small values are written in full.
"""

from petab_harness import paper_configurations as pc


def _run(**kw):
    base = {
        "cohort": "figure2",
        "arm": "O1",
        "example": "crauste",
        "run": "0" * 8,
        "archive_id": "0" * 32,
        "task": "1_0",
        "model": "z-ai/glm-5.3-flash",
        "provider_pin": ["z-ai"],
        "served_by": ["Z.AI"],
        "harness": "gemot",
        "driver": "deepagents",
        "biocurator": "on",
        "biology": "visible",
        "subagents": [],
        "ablation_arm": "baseline",
        "ablations": {},
        "git_sha": "604449e",
        "example_version": 2,
        "prompt_version": "637a3413b6ad",
        "skills_version": None,
        "temperature": 0.0,
        "recursion_limit": 500,
        "max_output_tokens": 131072,
        "criterion": "bic",
        "bic_threshold": 10.0,
        "n_starts": 100,
        "per_start_timeout_s": 60,
        "max_wall_seconds": 13200,
        "fit_workers": 100,
        "budget_source": "slurm log",
        "disabled_tools": [],
        "omnipath_snapshot": None,
        "stop_reason": "completed",
        "log": None,
    }
    return {**base, **kw}


def test_a_setting_the_runs_share_is_shown_once_and_one_they_differ_on_with_counts():
    runs = {"runs": [_run(), _run(git_sha="c1a851f", example="perelson"), _run()]}
    table = pc.render_cohorts(runs, "%")
    assert r"\emph{Figure 2: 3 runs over 2 benchmarks}" in table
    row = next(line for line in table.splitlines() if line.startswith("O1 &"))
    assert r"\texttt{604449e} (2), \texttt{c1a851f} (1)" in row
    assert "GLM-5.3-Flash & curator + biology + gemot &" in row
    assert row.endswith(r"& 3 \\")


def test_what_a_run_was_given_reads_as_figure_3_ladder():
    assert pc.support(_run()) == "curator + biology + gemot"
    assert pc.support(_run(biocurator="off", harness="deepagents")) == "biology + bare agent"
    assert pc.support(_run(biocurator="off", biology="blinded")) == "gemot"


def test_budgets_are_per_benchmark_and_hours_are_hours():
    runs = {
        "runs": [
            _run(),
            _run(example="lucarelli", per_start_timeout_s=1200, max_wall_seconds=120000),
        ]
    }
    table = pc.render_budgets(runs, "%")
    assert r"\emph{Figure 2}" in table
    assert r"crauste & 100 & 60 & 3.67 & 100 \\" in table
    assert r"lucarelli & 100 & 1200 & 33.3 & 100 \\" in table


def test_small_grid_values_are_written_in_full():
    assert pc._number(0.0032) == r"$3.2\times10^{-3}$"
    assert pc._number(0.0003) == r"$3\times10^{-4}$"
    assert pc._number(1.0) == "1" and pc._number(0.03) == "0.03" and pc._number(True) == "yes"


def test_every_table_is_one_balanced_tabular():
    runs = {"runs": [_run(), _run(cohort="figure3", arm="O3", harness="deepagents")]}
    for name, text in pc.export_data(runs, None, "% header").items():
        if name.endswith(".tex"):
            assert text.startswith("% header")
            opened = text.count(r"\begin{tabular}") + text.count(r"\begin{tabularx}")
            closed = text.count(r"\end{tabular}") + text.count(r"\end{tabularx}")
            assert opened == closed == 1


def test_the_reasoning_rows_say_what_was_sent_and_what_the_default_meant():
    """No run sends a reasoning setting, so the table states the providers' own defaults."""
    runs = {
        "runs": [
            _run(reasoning_sent="none", reasoning_runaways=0),
            _run(reasoning_sent="none", reasoning_runaways=2),
            _run(model="openai/gpt-6-astra", reasoning_sent="none", reasoning_runaways=0),
        ],
        "provider_reasoning": {
            "source": "listing",
            "fetched_at": "2026-09-25",
            "models": {
                "z-ai/glm-5.3-flash": {
                    "default_enabled": True,
                    "default_effort": "max",
                    "supported_efforts": ["max", "high", "low"],
                },
                "openai/gpt-6-astra": {
                    "default_enabled": True,
                    "default_effort": "medium",
                    "supported_efforts": ["max", "xhigh", "high", "medium", "low"],
                },
            },
        },
    }
    table = pc.render_shared(runs, "%")
    assert r"Reasoning effort requested & none (provider default) \\" in table
    assert r"Calls re-issued at low effort & 0 (2), 2 (1) \\" in table
    assert r"Default reasoning, GLM-5.3-Flash & on, max effort (of low, high, max) \\" in table
    assert (
        r"Default reasoning, GPT-6-Astra & on, medium effort "
        r"(of low, medium, high, xhigh, max) \\"
    ) in table
    assert "OpenRouter model metadata, 2026-09-25" in table
