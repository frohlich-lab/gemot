"""Optional Weights & Biases logging for sweep runs.

Each ``run.py`` invocation (one example × model × temperature × repeat cell) logs **one W&B
run**: ``config`` = the run's provenance (scenario, model, temperature, driver, versions,
and the ``phoenix_trace_id`` so you can jump to the agent trace in a local Phoenix), and the
summary metrics = the judge outcome (pass, ΔBIC, χ², reduced χ², timings, tool calls). W&B's
grouping / tables / parallel-coordinates UI then slices these across the model × temperature
grid.

Best-effort: if ``wandb`` isn't installed or no credentials are configured it prints a hint
and no-ops, so it never breaks a run. Credentials come from ``wandb login`` (``~/.netrc``),
``WANDB_API_KEY``, or W&B's usual resolution; set ``WANDB_MODE=offline`` on an air-gapped
cluster and ``wandb sync`` the runs later.
"""

from __future__ import annotations

from typing import Any

# Provenance -> W&B run config (the sweep axes + reproducibility).
_CONFIG_KEYS = (
    "scenario",
    "model",
    "driver",
    "temperature",
    "recursion_limit",
    "n_starts",
    # `prompt_version` is a run INPUT, like git_sha — config, not a summary metric. Without it
    # a W&B sweep cannot group or filter runs by the prompt text that produced them.
    "example_version",
    "git_sha",
    "prompt_version",
    "skills_version",
    # Same argument, for the arms that change no prompt text and so leave prompt_version equal
    # on both halves of an A/B (gemot.ablations).
    "ablation_arm",
    "ablation_state",
    # Which bar `success` was decided by: Δcriterion against a true model, or an absolute adequacy
    # test where there is none. Group by it before reading `passed` as a rate.
    "judge_mode",
    "phoenix_project",
    "phoenix_trace_id",
    "stop_reason",
)
# Outcome -> W&B summary metrics.
_METRIC_KEYS = (
    "success",
    "delta_bic",
    "bic",
    "bic_reference",
    # The delta that actually DECIDED `success` (canonical `delta_bic` above only does so when
    # BIC is the criterion), and the criterion it was measured under.
    "criterion",
    "criterion_value",
    "criterion_reference",
    "delta_criterion",
    "chi2",
    "chi2_dof",
    "chi2_gof_pvalue",
    "final_nll",
    "baseline_nll",
    "delta",
    "k",
    "n",
    "k_eff",
    "head_version",
    "n_errors",
    "tool_calls",
    "wall_seconds",
    "compute_seconds",
    "llm_seconds",
    "compile_seconds",
    "fit_seconds",
    "n_compiles",
    "n_fits",
)


def log_result(
    result: dict[str, Any], *, project: str = "petab-mcp-sweep", extra: dict[str, Any] | None = None
) -> bool:
    """Log one run's ``result.json`` dict to W&B as a run. Returns True if logged."""
    try:
        import wandb
    except ImportError:
        print("[wandb] not installed; skipping W&B logging (pip install wandb).")
        return False

    config = {k: result.get(k) for k in _CONFIG_KEYS}
    if extra:
        config.update(extra)

    metrics: dict[str, Any] = {k: result.get(k) for k in _METRIC_KEYS}
    # 0/1 so W&B averages give a success RATE when grouped; reduced χ² is derived. Infrastructure
    # noise logs neither: an abort before any modelling work carries `success: False` (the degraded
    # `aborted:` result of `runner.failure` though the judge never ran), and logged as a failure it
    # pulled every group's rate down by that group's aborts (#10). The buckets are the ones
    # `analysis.is_scoreable` leaves out of a sweep's rate, so the two rates agree: a LATE abort
    # that did work stays a failure there and here. A declined verdict (`success: null`) logs
    # neither too.
    from .analysis import stop_class

    noise = stop_class(result) in ("infra_noop", "server_unreachable", "no_result")
    success = None if noise else result.get("success")
    metrics["success"] = success
    metrics["passed"] = None if success is None else (1.0 if success else 0.0)
    # Reuse the one reduced-χ² definition (χ² / dof) instead of re-deriving it here.
    from .experiments import reduced_chi2

    metrics["reduced_chi2"] = reduced_chi2(result)
    # A reference-free run's declared test and verdict, under the history CSV's own column names.
    from .metrics import flatten_adequacy

    metrics.update(flatten_adequacy(result))

    run = None
    try:
        run = wandb.init(
            project=project,
            config=config,
            group=str(result.get("model")),  # a group per model
            job_type=str(result.get("scenario")),  # example as job type
            tags=[
                str(result.get("model")),
                str(result.get("scenario")),
                f"t{result.get('temperature')}",
            ],
            settings=wandb.Settings(silent=True),
        )
        run.log({k: v for k, v in metrics.items() if v is not None})
        print(
            f"[wandb] logged run -> project '{project}' "
            f"({result.get('scenario')} / {result.get('model')} / t{result.get('temperature')})"
        )
        return True
    except Exception as exc:  # never fail the run over telemetry
        print(f"[wandb] logging failed ({type(exc).__name__}: {exc}); continuing.")
        return False
    finally:
        # Always close the run — a failure after init() would otherwise leave it active.
        if run is not None:
            try:
                run.finish()
            except Exception:  # pragma: no cover - best-effort cleanup
                pass
