"""Phoenix-native Experiments + Prompt-registry integration for the model-building demos.

Drives the examples as a Phoenix **Dataset** through ``client.experiments.run_experiment``:
the experiment task reuses the existing cleanup → server → panel → judge pipeline
(:func:`runner.run_scenario_async`) and returns ``result.json``; **evaluators** score the
outcome (BIC pass / ΔBIC / χ² goodness-of-fit p / reduced χ²). Each experiment is one
*batch* over the fixed example set, so batches are directly comparable in the Experiments
UI over time.

**Prompt tracking.** Prompts stay in the repo (``scenario.py`` / ``agent.py`` are the
source of truth). Here we hash their content into a stable ``prompt_version`` and register
each logical prompt into Phoenix's **Prompt registry** (Phoenix versions a prompt when its
content changes), then tag every experiment with the ``prompt_version`` — so an experiment
records exactly which prompts produced it and prompt changes are visible/attributable.
"""

from __future__ import annotations

import hashlib
from typing import Any

# Stable dataset-name prefix. Kept as "...-repair" (the examples' original framing; they are
# now zero-shot BUILDS) on purpose: it is a Phoenix dataset *identifier*, so renaming it would
# orphan every existing dataset/experiment rather than describe them better. User-visible
# descriptions have been updated to "model build"; only this lineage key stays put.
DATASET_PREFIX = "petab-mcp-repair"


# --------------------------------------------------------------------------- prompts
def collect_prompts() -> dict[str, str]:
    """The repo's agent-facing prompts, keyed by a stable registry name.

    Source of truth is the code; this just gathers the text so it can be hashed + mirrored
    into Phoenix. The analyst persona constants already include the shared report contract.
    """
    from . import agent as A
    from .scenario import DEFAULT_SYSTEM_PROMPT, DEFAULT_TASK

    return {
        "petab-lead-system": DEFAULT_SYSTEM_PROMPT,
        "petab-lead-task": DEFAULT_TASK,
        "petab-analyst-mathematician": A.MATHEMATICIAN_PROMPT,
        "petab-analyst-systems-scientist": A.SYSTEMS_SCIENTIST_PROMPT,
        "petab-analyst-signal-engineer": A.SIGNAL_ENGINEER_PROMPT,
    }


# The chat role each prompt plays, so the registry mirrors how the harness actually uses
# them (lead system prompt + analyst personas are system instructions; the task is a user
# message). Role is a registration detail — it is NOT part of ``prompt_version`` (which
# hashes prompt text only), so fixing a role does not churn the version.
_PROMPT_ROLES = {
    "petab-lead-system": "system",
    "petab-lead-task": "user",
    "petab-analyst-mathematician": "system",
    "petab-analyst-systems-scientist": "system",
    "petab-analyst-signal-engineer": "system",
}


def prompt_version(prompts: dict[str, str] | None = None) -> str:
    """A stable 12-char content hash over all prompts — the harness's *prompt version*."""
    prompts = prompts if prompts is not None else collect_prompts()
    h = hashlib.sha256()
    for name in sorted(prompts):
        h.update(name.encode("utf-8"))
        h.update(b"\x00")
        h.update(prompts[name].encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:12]


def register_prompts(client, *, model_name: str) -> dict[str, Any]:
    """Mirror the repo prompts into Phoenix's Prompt registry (idempotent per content).

    Phoenix creates a new version of a named prompt only when its content changes, so
    calling this every run is cheap and keeps the registry in lock-step with the code.
    Returns ``{"prompt_version": <hash>, "registered": [...], "errors": [...]}``.
    """
    from phoenix.client.types import PromptVersion

    prompts = collect_prompts()
    version = prompt_version(prompts)
    registered, errors = [], []
    for name, text in prompts.items():
        try:
            client.prompts.create(
                name=name,
                prompt_description=f"petab-mcp model-building harness prompt ({name}); "
                                   f"source of truth is the repo. prompt_version={version}.",
                version=PromptVersion(
                    [{"role": _PROMPT_ROLES.get(name, "user"), "content": text}],
                    model_name=model_name,
                    template_format="NONE",  # prompts are literal text, not templates
                    description=f"prompt_version={version}",
                ),
                prompt_metadata={"prompt_version": version},
            )
            registered.append(name)
        except Exception as exc:  # registry is best-effort — never fail a run over it
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    return {"prompt_version": version, "registered": registered, "errors": errors}


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
        outputs=[{"reference_nll": ref.get("nll"), "reference_k": ref.get("k"),
                  "reference_n": ref.get("n"), "canonical_fix": scenario.fix_summary}],
        metadata=[{"example_version": scenario.version,
                   "bic_threshold": scenario.bic_threshold}],
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
def _run_one_via_subprocess(scenario, *, driver, model, temperature, n_starts,
                            recursion_limit, base_url, wandb=False,
                            wandb_project=None) -> dict:
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

    cmd = [sys.executable, str(scenario.example_dir / "run.py"), "--driver", driver,
           "--model", model, "--temperature", str(temperature),
           "--recursion-limit", str(recursion_limit), "--n-starts", str(n_starts)]
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
        cmd = [sys.executable, str(s.example_dir / "run.py"), "--driver", "scripted",
               "--no-phoenix", "--n-starts", str(n_starts)]
        rc = subprocess.run(cmd, check=False).returncode
        rp = s.example_dir / "work" / "result.json"
        res = json.loads(rp.read_text()) if rp.is_file() else {"success": rc == 0}
        results.append({"scenario": s.name, **res})
        print(f"[scripted] {s.name}: {'PASS' if res.get('success') else 'FAIL'} "
              f"(ΔBIC {res.get('delta_bic')})")
    return results


def run_experiments(
    scenarios: list,
    *,
    base_url: str = "http://localhost:6006",
    driver: str = "deepagents",
    model: str = "deepseek/deepseek-v4-pro",
    temperature: float = 0.0,
    n_starts: int = 50,
    recursion_limit: int = 200,
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

    from .runner import _git_sha

    client = Client(base_url=_phoenix_base_url(base_url))
    prompts_info = register_prompts(client, model_name=model)  # once; prompts are global
    git_sha = _git_sha(scenarios[0].example_dir) if scenarios else None
    metadata = {
        "driver": driver, "model": model, "temperature": temperature,
        "git_sha": git_sha, "prompt_version": prompts_info["prompt_version"],
        "recursion_limit": recursion_limit, "n_starts": n_starts,
    }

    ran = []
    for scenario in scenarios:
        dataset = get_or_create_dataset(client, scenario)

        def task(input: dict, _scenario=scenario) -> dict:
            return _run_one_via_subprocess(
                _scenario, driver=driver, model=model, temperature=temperature,
                n_starts=n_starts, recursion_limit=recursion_limit, base_url=base_url,
                wandb=wandb, wandb_project=wandb_project,
            )

        # Default the experiment name to the MODEL only; temperature, driver, git_sha, and
        # prompt_version all ride in experiment_metadata (below), so Phoenix groups every run
        # of a model — across temperatures and repeats — under one experiment identity instead
        # of splitting them into a separate experiment per (temp, sha) cell.
        name = experiment_name or model.split("/")[-1]
        ran.append(client.experiments.run_experiment(
            dataset=dataset,
            task=task,
            evaluators=EVALUATORS,
            experiment_name=name,
            experiment_description=f"petab-mcp {scenario.name} zero-shot model build "
                                   f"(prompt_version={prompts_info['prompt_version']})",
            experiment_metadata=metadata,
            dry_run=dry_run,
        ))
    return ran


def _phoenix_base_url(url: str) -> str:
    """Normalize to the Phoenix base (strip a trailing OTLP path) for ``Client``."""
    out = url.rstrip("/")
    for suffix in ("/v1/traces", "/v1/otlp"):
        if out.endswith(suffix):
            return out[: -len(suffix)]
    return out


# --------------------------------------------------------------------------- sweep report
# Read the logged experiments back out for a sweep summary (success rate + ΔBIC across
# model × temperature × example). One row per experiment task-run; each experiment is a
# single (model, temperature, repeat) for one example, so a cell's repeats are just the
# runs that share (example, model, temperature).
_ROW_FIELDS = ("example", "model", "temperature", "success", "delta_bic",
               "chi2_gof_pvalue", "stop_reason", "prompt_version", "git_sha")


def fetch_experiment_rows(base_url: str = "http://localhost:6006") -> list[dict[str, Any]]:
    """One flat row per experiment task-run across every ``petab-mcp-repair-*`` dataset."""
    from phoenix.client import Client

    client = Client(base_url=_phoenix_base_url(base_url))
    rows: list[dict[str, Any]] = []
    for ds_meta in client.datasets.list():
        name = ds_meta.get("name", "") if isinstance(ds_meta, dict) else getattr(ds_meta, "name", "")
        prefix = f"{DATASET_PREFIX}-"
        if not name.startswith(prefix):
            continue
        example = name[len(prefix):]
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
                rows.append({
                    "example": example,
                    "model": md.get("model") or out.get("model"),
                    "temperature": md.get("temperature"),
                    "success": out.get("success"),
                    "delta_bic": out.get("delta_bic"),
                    "chi2_gof_pvalue": out.get("chi2_gof_pvalue"),
                    "stop_reason": out.get("stop_reason"),
                    "prompt_version": md.get("prompt_version"),
                    "git_sha": md.get("git_sha"),
                })
    return rows


def summarize_sweep(rows: list[dict[str, Any]], *, completed_only: bool = True) -> str:
    """Success rate + ΔBIC per (example × model × temperature), plus a model×temp pivot.

    ``completed_only`` (default): drop runs whose ``stop_reason != "completed"`` — i.e. the
    agent loop crashed or ran out of steps (APIConnectionError, GraphRecursionError) and the
    judge scored an incomplete model. Those are harness/infra outcomes, not the model
    finishing with a verdict, so they'd otherwise inflate the failure rate.
    """
    import statistics

    if not rows:
        return "(no experiments found — has the sweep produced any runs yet?)"

    crashed = [r for r in rows if (r.get("stop_reason") or "completed") != "completed"]
    note = ""
    if completed_only and crashed:
        rows = [r for r in rows if (r.get("stop_reason") or "completed") == "completed"]
        from collections import Counter
        by = Counter((r.get("stop_reason") or "?").split(":")[0] for r in crashed)
        note = (f"(excluded {len(crashed)} crashed/incomplete run(s) — "
                f"{dict(by)}; pass --include-crashed to keep them)\n")
        if not rows:
            return note + "(no completed runs yet)"

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
    hdr = f"{'example':<16} {'model':<26} {'temp':>4} {'n':>3} {'pass':>5} " \
          f"{'ΔBIC med':>9} {'ΔBIC min':>9} {'ΔBIC max':>9} {'chi2p med':>9}"
    lines += [hdr, "-" * len(hdr)]
    for key in sorted(cells, key=lambda k: (str(k[0]), str(k[1]), k[2] is None, str(k[2]))):
        ex, model, temp = key
        group = cells[key]
        n = len(group)
        passes = sum(1 for g in group if g.get("success"))
        dbs = [g.get("delta_bic") for g in group]
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
    head = f"{'model':<26}" + "".join(f"{('t='+_fmt(t,'g')):>10}" for t in temps)
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
    lines += ["", f"overall: {total_pass}/{len(rows)} runs passed "
                  f"({len(cells)} cells, prompt_version={rows[0].get('prompt_version')})"]
    return "\n".join(lines)
