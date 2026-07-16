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
    "scenario", "model", "driver", "temperature", "recursion_limit", "n_starts",
    "example_version", "git_sha", "phoenix_project", "phoenix_trace_id", "stop_reason",
)
# Outcome -> W&B summary metrics.
_METRIC_KEYS = (
    "success", "delta_bic", "bic", "bic_reference", "chi2", "chi2_dof", "chi2_gof_pvalue",
    "final_nll", "baseline_nll", "delta", "k", "n", "head_version", "n_errors", "tool_calls",
    "wall_seconds", "compute_seconds", "llm_seconds",
    "compile_seconds", "fit_seconds", "n_compiles", "n_fits",
)


def log_result(result: dict[str, Any], *, project: str = "petab-mcp-sweep",
               extra: dict[str, Any] | None = None) -> bool:
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
    # 0/1 so W&B averages give a success RATE when grouped; reduced χ² is derived.
    metrics["passed"] = 1.0 if result.get("success") else 0.0
    chi2, dof = result.get("chi2"), result.get("chi2_dof")
    metrics["reduced_chi2"] = (float(chi2) / dof) if (chi2 is not None and dof) else None

    run = None
    try:
        run = wandb.init(
            project=project,
            config=config,
            group=str(result.get("model")),        # a group per model
            job_type=str(result.get("scenario")),  # example as job type
            tags=[str(result.get("model")), str(result.get("scenario")),
                  f"t{result.get('temperature')}"],
            settings=wandb.Settings(silent=True),
        )
        run.log({k: v for k, v in metrics.items() if v is not None})
        print(f"[wandb] logged run -> project '{project}' "
              f"({result.get('scenario')} / {result.get('model')} / t{result.get('temperature')})")
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
