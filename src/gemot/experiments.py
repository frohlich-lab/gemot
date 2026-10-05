"""Phoenix-native Experiments + Prompt-registry integration for the model-building demos.

Drives the examples as a Phoenix **Dataset** through ``client.experiments.run_experiment``:
the experiment task reuses the existing cleanup → server → panel → judge pipeline
(:func:`runner.run_scenario_async`) and returns ``result.json``; **evaluators** score the
outcome (BIC pass / ΔBIC / χ² goodness-of-fit p / reduced χ²). Each experiment is one
*batch* over the fixed example set, so batches are directly comparable in the Experiments
UI over time.

**Prompt tracking.** Prompts stay in the repo (``prompts.py`` is the
source of truth). Here we hash their content into a stable ``prompt_version`` and register
each logical prompt into Phoenix's **Prompt registry** (Phoenix versions a prompt when its
content changes), then tag every experiment with the ``prompt_version`` — so an experiment
records exactly which prompts produced it and prompt changes are visible/attributable.
"""

from __future__ import annotations

from typing import Any

from . import models, server_api

# Stable dataset-name prefix. Kept as "...-repair" (the examples' original framing; they are
# now zero-shot BUILDS) on purpose: it is a Phoenix dataset *identifier*, so renaming it would
# orphan every existing dataset/experiment rather than describe them better. User-visible
# descriptions have been updated to "model build"; only this lineage key stays put.
DATASET_PREFIX = "petab-mcp-repair"


# --------------------------------------------------------------------------- prompts
def collect_prompts() -> dict[str, str]:
    """The repo's agent-facing prompts, keyed by a stable registry name.

    The registry itself lives with the text, in :mod:`gemot.prompts` — this is a thin
    flatten of ``PROMPT_GROUPS`` so callers here need not know the grouping.

    It is the FULL surface: the lead's prompt and task, the analyst personas and the
    descriptions the lead picks them by, the biocurator, the panel-facing tool descriptions
    (which REPLACE the server docstrings), the nudges, and the finish gate. It used to be five
    keys, which left the other ~30 model-facing constants outside ``prompt_version`` entirely;
    ``tests/test_prompt_tracking.py`` now fails if a prompt escapes the registry.
    """
    from . import prompts as P

    return P.all_prompts()


# The chat role each prompt plays, so the registry mirrors how the harness actually uses
# them: personas and instruction blocks are system text, the lead's task is a user message,
# and the harness's own messages back to the model (nudges, refusals, tool descriptions) are
# what the model receives mid-conversation. Role is a registration detail — it is NOT part of
# ``prompt_version`` (which hashes prompt text only), so fixing a role does not churn it.
_PROMPT_ROLES = {
    "petab-anonymous-semantics": "system",
    "petab-anonymous-lead": "system",
    "petab-anonymous-task": "user",
    "petab-anonymous-protocol-note": "user",  # appended to the blind task when a problem has one
    "petab-anonymous-analysts": "system",
    "petab-anonymous-tools": "system",
    "petab-active-biology-context": "system",
    "petab-lead-system": "system",
    "petab-lead-task": "user",
    "petab-lead-readout-note": "user",
    # the null arm's single agent (see agent/barebones.py): same two roles as the lead's pair
    "petab-barebones-system": "system",
    "petab-barebones-task": "user",
    "petab-barebones-entities": "user",
    "petab-barebones-bio-tools": "system",
    "petab-analyst-mathematician": "system",
    "petab-analyst-systems-scientist": "system",
    "petab-analyst-signal-engineer": "system",
    "petab-analyst-descriptions": "system",
    "petab-model-editor-system": "system",
    "petab-model-editor-description": "system",
    "petab-biocurator-system": "system",
    "petab-biocurator-description": "system",
    "petab-biocurator-entity-block": "user",
    "petab-tool-descriptions": "system",
    "petab-finish-description": "system",
    "petab-finish-gate": "user",
    "petab-nudges": "user",
    "petab-guards": "user",
    # every `petab-skill-*` entry defaults to "system" via the .get() below —
    # a skill is read as instructions, not as a turn in the conversation.
}


def prompt_version(prompts: dict[str, str] | None = None) -> str:
    """A stable 12-char content hash over all prompts — the harness's *prompt version*.

    Re-exported from :mod:`gemot.prompts`, where it lives beside the text it hashes.
    Kept importable here because every existing caller reaches for it through this module.
    """
    from . import prompts as P

    return P.prompt_version(prompts)


def prompt_group_versions() -> dict[str, str]:
    """Per-group prompt hashes — see :func:`gemot.prompts.prompt_group_versions`."""
    from . import prompts as P

    return P.prompt_group_versions()


def collect_skills() -> dict[str, str]:
    """The server's bundled SKILLS, keyed for the registry.

    Skills are agent-facing instructions, not reference docs: the lead is told to read them
    before authoring (see prompts.DEFAULT_SYSTEM_PROMPT step 0). A change to one therefore
    changes what the agent is told, so they are registered and versioned like the prompts.
    """
    texts = server_api.skills().skill_texts()
    return {f"petab-skill-{name}": text for name, text in texts.items()}


def skills_version() -> str:
    """Content hash over the bundled skills — see :func:`petab_mcp.skills.skills_version`."""
    return server_api.skills().skills_version()


def register_prompts(client, *, model_name: str) -> dict[str, Any]:
    """Mirror the repo prompts AND skills into Phoenix's Prompt registry (idempotent per
    content).

    Phoenix creates a new version of a named prompt only when its content changes, so
    calling this every run is cheap and keeps the registry in lock-step with the code. Skills
    ride here because the panel now READS them — they are instructions, so the registry that
    records "what was the agent told" is incomplete without them.
    Returns ``{"prompt_version", "skills_version", "registered", "errors"}``.
    """
    from phoenix.client.types import PromptVersion

    # Registered together, hashed SEPARATELY: `prompt_version` must mean the same thing here as
    # it does in result.json (prompts.py only), or the two would disagree. A skill entry carries
    # `skills_version` instead.
    skills = collect_skills()
    prompts = {**collect_prompts(), **skills}
    version = prompt_version()
    skills_v = skills_version()
    registered, errors = [], []
    for name, text in prompts.items():
        is_skill = name in skills
        # a skill is instructions the agent READS on demand; a prompt is text it is SENT
        role = "system" if is_skill else _PROMPT_ROLES.get(name, "user")
        stamp = f"skills_version={skills_v}" if is_skill else f"prompt_version={version}"
        kind = "skill" if is_skill else "prompt"
        try:
            client.prompts.create(
                name=name,
                prompt_description=f"petab-mcp model-building harness {kind} ({name}); "
                f"source of truth is the repo. {stamp}.",
                version=PromptVersion(
                    [{"role": role, "content": text}],
                    model_name=model_name,
                    template_format="NONE",  # prompts are literal text, not templates
                    description=stamp,
                ),
                prompt_metadata={"prompt_version": version, "skills_version": skills_v},
            )
            registered.append(name)
        except Exception as exc:  # registry is best-effort — never fail a run over it
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    return {
        "prompt_version": version,
        "skills_version": skills_v,
        "registered": registered,
        "errors": errors,
    }


# --------------------------------------------------------------------------- dataset
def dataset_name(scenario) -> str:
    """Per-example dataset name — each example is its own dataset (and thus its own
    experiment), so runs of different examples are never pooled into one experiment."""
    return f"{DATASET_PREFIX}-{scenario.name}"


def get_or_create_dataset(client, scenario) -> Any:
    """The single-row dataset for one scenario (get-or-create)."""
    name = dataset_name(scenario)
    try:
        return client.datasets.get_dataset(dataset=name)
    except Exception:
        pass
    ref = scenario.reference() or {}
    return client.datasets.create_dataset(
        name=name,
        dataset_description=f"petab-mcp zero-shot model-building example: {scenario.name}.",
        inputs=[{"scenario": scenario.name, "task": scenario.task}],
        outputs=[
            {
                "reference_nll": ref.get("nll"),
                "reference_k": ref.get("k"),
                "reference_n": ref.get("n"),
                "canonical_fix": scenario.fix_summary,
            }
        ],
        metadata=[{"example_version": scenario.version, "bic_threshold": scenario.bic_threshold}],
    )


# --------------------------------------------------------------------------- evaluators
# Plain callables over the task output (the judge's result dict). Phoenix uses the function
# name as the evaluator name; a bool -> pass/fail label, a float -> a recorded score.
def bic_pass(output: dict) -> bool:
    """The gate: did the built model beat the ΔBIC threshold vs the true model?"""
    return bool(output.get("success"))


def delta_bic(output: dict) -> float | None:
    """ΔBIC vs the true model (lower is better; ~0 = indistinguishable from truth)."""
    v = output.get("delta_bic")
    return float(v) if v is not None else None


def chi2_gof_pvalue(output: dict) -> float | None:
    """χ² goodness-of-fit p vs the known noise (>=0.05 = consistent with noise)."""
    v = output.get("chi2_gof_pvalue")
    return float(v) if v is not None else None


def reduced_chi2(output: dict) -> float | None:
    """χ² / dof (~1 = at the noise floor; ≫1 = still mis-specified)."""
    chi2, dof = output.get("chi2"), output.get("chi2_dof")
    return float(chi2) / dof if (chi2 is not None and dof) else None


EVALUATORS = [bic_pass, delta_bic, chi2_gof_pvalue, reduced_chi2]


# --------------------------------------------------------------------------- run
def _run_one_via_subprocess(
    scenario,
    *,
    driver,
    model,
    temperature,
    n_starts,
    recursion_limit,
    base_url,
    wandb=False,
    wandb_project=None,
) -> dict:
    """Run one scenario as a subprocess of its ``run.py`` and return its ``result.json``.

    Reuses the fully-exercised ``run.py`` path so the agent's complete ``model_build_run``
    trace (every LLM + tool span) is captured in the ``petab-mcp-<example>`` project as a
    normal run, with the ``phoenix_trace_id`` coming back in ``result.json`` to link it to
    the experiment. Process isolation also avoids fighting ``run_experiment`` over the global
    OTel tracer provider (in-process this client dispatches sync tasks to a worker thread and
    does not await coroutines, so neither in-process approach captures the agent spans).

    With ``wandb=True`` the subprocess also logs the run to Weights & Biases, so one sweep
    populates the Phoenix experiment AND W&B in parallel (both kept, not either/or).
    """
    import json
    import os
    import subprocess
    import sys

    cmd = [
        sys.executable,
        str(scenario.example_dir / "run.py"),
        "--driver",
        driver,
        "--model",
        model,
        "--temperature",
        str(temperature),
        "--recursion-limit",
        str(recursion_limit),
        "--n-starts",
        str(n_starts),
    ]
    if wandb:
        cmd.append("--wandb")
        if wandb_project:
            cmd += ["--wandb-project", wandb_project]
    env = dict(os.environ)
    env["PHOENIX_COLLECTOR_ENDPOINT"] = base_url  # agent trace -> petab-mcp-<example>
    subprocess.run(cmd, env=env, check=False)
    result_path = scenario.example_dir / "work" / "result.json"
    return json.loads(result_path.read_text()) if result_path.is_file() else {"success": False}


def run_scripted_selftest(scenarios: list, *, n_starts: int = 50) -> list:
    """No-LLM self-test of the pipeline — runs each scenario's canonical scripted build.

    Deliberately does NOT touch Phoenix (no dataset/experiment/prompt registration): the
    scripted driver is a harness self-test, not an evaluation to record.
    """
    import json
    import subprocess
    import sys

    results = []
    for s in scenarios:
        cmd = [
            sys.executable,
            str(s.example_dir / "run.py"),
            "--driver",
            "scripted",
            "--n-starts",
            str(n_starts),
        ]
        rc = subprocess.run(cmd, check=False).returncode
        rp = s.example_dir / "work" / "result.json"
        res = json.loads(rp.read_text()) if rp.is_file() else {"success": rc == 0}
        results.append({"scenario": s.name, **res})
        print(
            f"[scripted] {s.name}: {'PASS' if res.get('success') else 'FAIL'} "
            f"(ΔBIC {res.get('delta_bic')})"
        )
    return results


def run_experiments(
    scenarios: list,
    *,
    base_url: str = "http://localhost:6006",
    driver: str = "deepagents",
    model: str = models.DEFAULT,
    temperature: float = 0.0,
    n_starts: int = 50,
    recursion_limit: int = 250,
    experiment_name: str | None = None,
    dry_run: bool = False,
    wandb: bool = False,
    wandb_project: str | None = None,
) -> list:
    """Run ONE Phoenix Experiment PER example (each over its own single-row dataset), so
    different examples are never pooled into one experiment. Registers the prompts once and
    tags every experiment with model/temperature/git_sha/prompt_version. Returns the list of
    ``RanExperiment``.

    With ``wandb=True`` each run is ALSO logged to Weights & Biases (Phoenix experiment
    logging is kept, not replaced — one sweep populates both).

    The ``scripted`` driver is a no-LLM self-test and is NOT logged to Phoenix — it runs
    :func:`run_scripted_selftest` and returns its result dicts instead.
    """
    if driver == "scripted":
        return run_scripted_selftest(scenarios, n_starts=n_starts)

    from phoenix.client import Client

    from .tracing import phoenix_base_url

    from .runner import _git_sha

    client = Client(base_url=phoenix_base_url(base_url))
    prompts_info = register_prompts(client, model_name=model)  # once; prompts are global
    # Registration is best-effort, but a SILENT best-effort is worse than none: a partly-failed
    # registration leaves the Phoenix registry missing prompts while `prompt_version` still
    # claims to cover them. Print what did not land so the discrepancy is visible.
    if prompts_info["errors"]:
        print(
            f"[prompts] WARNING: {len(prompts_info['errors'])} of "
            f"{len(prompts_info['errors']) + len(prompts_info['registered'])} prompts failed to "
            f"register in Phoenix; prompt_version={prompts_info['prompt_version']} still covers "
            "them, so the registry is INCOMPLETE for this run:"
        )
        for err in prompts_info["errors"]:
            print(f"[prompts]   {err}")
    git_sha = _git_sha(scenarios[0].example_dir) if scenarios else None
    metadata = {
        "driver": driver,
        "model": model,
        "temperature": temperature,
        "git_sha": git_sha,
        "prompt_version": prompts_info["prompt_version"],
        "skills_version": prompts_info["skills_version"],
        "recursion_limit": recursion_limit,
        "n_starts": n_starts,
    }

    ran = []
    for scenario in scenarios:
        dataset = get_or_create_dataset(client, scenario)

        def task(input: dict, _scenario=scenario) -> dict:
            return _run_one_via_subprocess(
                _scenario,
                driver=driver,
                model=model,
                temperature=temperature,
                n_starts=n_starts,
                recursion_limit=recursion_limit,
                base_url=base_url,
                wandb=wandb,
                wandb_project=wandb_project,
            )

        # Default the experiment name to the MODEL only; temperature, driver, git_sha, and
        # prompt_version all ride in experiment_metadata (below), so Phoenix groups every run
        # of a model — across temperatures and repeats — under one experiment identity instead
        # of splitting them into a separate experiment per (temp, sha) cell.
        name = experiment_name or model.split("/")[-1]
        ran.append(
            client.experiments.run_experiment(
                dataset=dataset,
                task=task,
                evaluators=EVALUATORS,
                experiment_name=name,
                experiment_description=f"petab-mcp {scenario.name} zero-shot model build "
                f"(prompt_version={prompts_info['prompt_version']})",
                experiment_metadata=metadata,
                dry_run=dry_run,
            )
        )
    return ran


# --------------------------------------------------------------------------- sweep report
# Read the logged experiments back out for a sweep summary (success rate + ΔBIC across
# model × temperature × example). One row per experiment task-run; each experiment is a
# single (model, temperature, repeat) for one example, so a cell's repeats are just the
# runs that share (example, model, temperature).
_ROW_FIELDS = (
    "example",
    "model",
    "temperature",
    "success",
    "delta_bic",
    # The delta the GATE used, and the criterion it used — `delta_bic` is only the
    # gating number when BIC is that criterion, so aggregating it beside `success`
    # would report one comparison next to a verdict from another.
    "delta_criterion",
    "criterion",
    "chi2_gof_pvalue",
    "stop_reason",
    "n_ops",
    "n_fits",
    "prompt_version",
    "skills_version",
    "git_sha",
)


# --------------------------------------------------------------------------- abort triage
# A run can stop for two very different reasons, and they must NOT be scored the same way:
#
#   (a) an INSTANT infrastructure abort before any modeling work — out-of-credits (HTTP 402),
#       a dropped/timed-out provider or MCP connection, a wedged step, or a harness-side crash
#       (e.g. a scoring float() that raised) — while n_ops==0 && n_fits==0. The run never
#       authored or fit a model, so its "ΔBIC" is just empty-model scoring noise (numbers like
#       1.55e7). These are EXCLUDED from the success/ΔBIC aggregation.
#
#   (b) a LATE abort AFTER real work — e.g. credit exhaustion after 25 fits across 26 committed
#       versions. Such a run did genuine modeling and its best-achieved ΔBIC is a valid PARTIAL
#       result, so it is KEPT (not dropped, not blanket-tagged "infrastructure").
#
# "completed", the self-imposed wall-time budget, and the agent step/recursion limit are genuine
# modeling OUTCOMES and are never treated as infrastructure aborts (a recursion-limit run with no
# commits genuinely failed to build anything — it is a modeling failure, not scoring noise).
_INFRA_ABORT_MARKERS = (
    "402",  # OpenRouter payment-required (out of credits)
    "apistatuserror",
    "apiconnectionerror",
    "apitimeouterror",
    "apierror",
    "internalservererror",
    "ratelimiterror",
    "permissiondeniederror",
    "authenticationerror",
    "readtimeout",
    "readerror",
    "writeerror",
    "connecterror",
    "connecttimeout",
    "remoteprotocolerror",
    "pooltimeout",
    "stalled",  # harness stall watchdog: hung provider call / wedged fit
    "fit call timed out",  # per-call MCP read timeout (clock._wrap_call_timeout). Same
    # event as the "readtimeout" above -- it only stopped being an
    # httpx.ReadTimeout when the timeout moved from the session to
    # the individual call, and it must keep scoring as infra.
    "typeerror",  # e.g. a scoring float() crash before a verdict was reached
)


def _as_count(v: Any) -> int:
    """A work-count field (n_ops / n_fits) as a non-negative int.

    None / NaN / non-numeric -> 0; a real number is floored at 0. ``bool`` is excluded even
    though it subclasses ``int`` (a True/False slipping into a count field is bad data, not a
    count of 1/0), and negatives are clamped so a garbage negative can never read as work done.
    """
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v:  # v != v rejects NaN
        return 0
    return max(0, int(v))


def _did_modeling_work(row: dict[str, Any]) -> bool:
    """Whether a run authored or fit ANY model (n_ops>0 or n_fits>0). ``n_ops`` is the committed
    model-version count (``head_version``); ``n_fits`` the number of fits run."""
    return _as_count(row.get("n_ops")) > 0 or _as_count(row.get("n_fits")) > 0


def _is_infra_abort(stop_reason: Any) -> bool:
    """Whether ``stop_reason`` marks an infrastructure/transport abort (not a genuine outcome)."""
    s = str(stop_reason or "").strip().lower()
    if s in ("", "completed"):
        return False
    # Genuine agent outcomes, not infrastructure: the run hit a harness budget or the graph's step
    # limit because of what it DID, so it counts as a failure rather than being dropped as noise.
    if "wall_time_budget" in s or "token_budget" in s or "recursion" in s:
        return False
    if s.startswith("aborted:"):  # pre-verdict transport abort (runner._write_degraded_result)
        return True
    return any(m in s for m in _INFRA_ABORT_MARKERS)


def is_noop_infra_abort(row: dict[str, Any]) -> bool:
    """A run that aborted on infrastructure BEFORE doing any modeling work — pure scoring noise.

    Only these are excluded from the success/ΔBIC aggregation: an infrastructure/transport stop
    (:func:`_is_infra_abort`) with no modeling work done (:func:`_did_modeling_work` is False).
    A LATE abort that did work is deliberately NOT one of these — its partial result is kept.
    """
    return _is_infra_abort(row.get("stop_reason")) and not _did_modeling_work(row)


def fetch_experiment_rows(base_url: str = "http://localhost:6006") -> list[dict[str, Any]]:
    """One flat row per experiment task-run across every ``petab-mcp-repair-*`` dataset."""
    from phoenix.client import Client

    from .tracing import phoenix_base_url

    client = Client(base_url=phoenix_base_url(base_url))
    rows: list[dict[str, Any]] = []
    for ds_meta in client.datasets.list():
        name = (
            ds_meta.get("name", "") if isinstance(ds_meta, dict) else getattr(ds_meta, "name", "")
        )
        prefix = f"{DATASET_PREFIX}-"
        if not name.startswith(prefix):
            continue
        example = name[len(prefix) :]
        dataset = client.datasets.get_dataset(dataset=name)
        for exp in client.experiments.list(dataset_id=dataset.id):
            exp_id = exp.get("id") if isinstance(exp, dict) else getattr(exp, "id", None)
            if exp_id is None:
                continue
            ran = client.experiments.get_experiment(experiment_id=exp_id)
            md = ran.get("experiment_metadata") or {}
            for tr in ran.get("task_runs") or []:
                out = tr.get("output")
                if not isinstance(out, dict):
                    continue
                rows.append(
                    {
                        "example": example,
                        "model": md.get("model") or out.get("model"),
                        "temperature": md.get("temperature"),
                        "success": out.get("success"),
                        "delta_bic": out.get("delta_bic"),
                        "delta_criterion": out.get("delta_criterion"),
                        "criterion": out.get("criterion"),
                        "chi2_gof_pvalue": out.get("chi2_gof_pvalue"),
                        "stop_reason": out.get("stop_reason"),
                        # Modeling-work signals used to tell an instant infrastructure abort from a
                        # late-abort partial (see summarize_sweep): n_ops = committed model versions;
                        # n_fits = number of fits run. Both zero ⇒ no work was done. head_version is
                        # 0-based and -1 when nothing was fit, so the COUNT is head_version + 1.
                        "n_ops": (
                            None
                            if out.get("head_version") is None
                            else int(out["head_version"]) + 1
                        ),
                        "n_fits": out.get("n_fits"),
                        "prompt_version": md.get("prompt_version"),
                        "git_sha": md.get("git_sha"),
                    }
                )
    return rows


def summarize_sweep(rows: list[dict[str, Any]], *, drop_noop_infra_aborts: bool = True) -> str:
    """Success rate + ΔBIC per (example × model × temperature), plus a model×temp pivot.

    ``drop_noop_infra_aborts`` (default): exclude only the runs that aborted on infrastructure
    BEFORE doing any modeling work — an out-of-credits 402, a dropped/timed-out connection, a
    wedged step, or a harness crash while ``n_ops==0 && n_fits==0`` (:func:`is_noop_infra_abort`).
    Those never built or fit a model, so their empty-model ΔBIC is scoring noise that would
    otherwise pollute the aggregation. Runs that DID modeling work before a late abort (credit
    exhaustion after real fits) are KEPT — their best-achieved ΔBIC is a valid partial result —
    and genuine outcomes (completed, wall-time budget, recursion limit) are always kept. Pass
    ``drop_noop_infra_aborts=False`` (``--include-crashed``) to keep the no-work aborts too.
    """
    import statistics

    if not rows:
        return "(no experiments found — has the sweep produced any runs yet?)"

    excluded = [r for r in rows if is_noop_infra_abort(r)]
    note = ""
    if drop_noop_infra_aborts and excluded:
        rows = [r for r in rows if not is_noop_infra_abort(r)]
        from collections import Counter

        by = Counter((r.get("stop_reason") or "?").split(":")[0].strip() for r in excluded)
        note = (
            f"(excluded {len(excluded)} no-work infrastructure abort(s) "
            f"[n_ops==0 & n_fits==0] — {dict(by)}; late-abort partials are KEPT; "
            f"pass --include-crashed to keep these too)\n"
        )
        if not rows:
            return note + "(no scored runs yet — every run was a no-work infrastructure abort)"

    def _key(r):
        return (r.get("example"), r.get("model"), r.get("temperature"))

    cells: dict[tuple, list[dict]] = {}
    for r in rows:
        cells.setdefault(_key(r), []).append(r)

    def _med(vals):
        vals = [v for v in vals if isinstance(v, (int, float))]
        return statistics.median(vals) if vals else None

    def _fmt(v, spec=""):
        if v is None:
            return "-"
        try:
            return format(v, spec) if spec else str(v)
        except (TypeError, ValueError):
            return str(v)

    lines = [note] if note else []
    lines += ["Per-cell (example × model × temperature):", ""]
    hdr = (
        f"{'example':<16} {'model':<26} {'temp':>4} {'n':>3} {'pass':>5} "
        f"{'ΔBIC med':>9} {'ΔBIC min':>9} {'ΔBIC max':>9} {'chi2p med':>9}"
    )
    lines += [hdr, "-" * len(hdr)]
    for key in sorted(cells, key=lambda k: (str(k[0]), str(k[1]), k[2] is None, str(k[2]))):
        ex, model, temp = key
        group = cells[key]
        n = len(group)
        passes = sum(1 for g in group if g.get("success"))
        # The gate's delta, falling back to canonical BIC for a row written before it existed
        # (they are the same number whenever BIC is the criterion).
        dbs = [
            g.get("delta_criterion", g.get("delta_bic"))
            if g.get("delta_criterion") is not None
            else g.get("delta_bic")
            for g in group
        ]
        lines.append(
            f"{_fmt(ex):<16} {_fmt(model):<26} {_fmt(temp, 'g'):>4} {n:>3} "
            f"{passes}/{n:<3} {_fmt(_med(dbs), '+.3g'):>9} "
            f"{_fmt(min([d for d in dbs if isinstance(d, (int, float))], default=None), '+.3g'):>9} "
            f"{_fmt(max([d for d in dbs if isinstance(d, (int, float))], default=None), '+.3g'):>9} "
            f"{_fmt(_med([g.get('chi2_gof_pvalue') for g in group]), '.2g'):>9}"
        )

    # model × temperature pivot (pooled over examples): pass rate
    lines += ["", "Pass rate by model × temperature (pooled over examples):", ""]
    models = sorted({r.get("model") for r in rows}, key=str)
    temps = sorted({r.get("temperature") for r in rows}, key=lambda t: (t is None, str(t)))
    head = f"{'model':<26}" + "".join(f"{('t=' + _fmt(t, 'g')):>10}" for t in temps)
    lines += [head, "-" * len(head)]
    for model in models:
        cell_strs = []
        for t in temps:
            grp = [r for r in rows if r.get("model") == model and r.get("temperature") == t]
            if grp:
                p = sum(1 for g in grp if g.get("success"))
                cell_strs.append(f"{p}/{len(grp)}")
            else:
                cell_strs.append("-")
        lines.append(f"{_fmt(model):<26}" + "".join(f"{c:>10}" for c in cell_strs))

    total_pass = sum(1 for r in rows if r.get("success"))
    lines += [
        "",
        f"overall: {total_pass}/{len(rows)} runs passed "
        f"({len(cells)} cells, prompt_version={rows[0].get('prompt_version')})",
    ]
    return "\n".join(lines)
