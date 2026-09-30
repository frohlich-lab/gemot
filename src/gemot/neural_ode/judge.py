"""Score the route's proposals the way every other model is scored: through the server, by the judge.

A proposal is Antimony, and the judge scores PEtab problems the server has fitted. So each
proposal is committed by the scripted driver -- the same ``build_true_model`` call the examples'
canonical builds use -- against a private copy of the example (``problem/`` with its
``reference.json`` and protocol table, ``validation/`` with the held-out set), and the judge writes
the ``result.json`` it writes for a panel run: ΔBIC against the reference, the held-out block,
success under the example's own threshold. Nothing is re-implemented here; the neural route's
models simply enter the existing comparison, under the example's own ``Scenario`` where its
``run.py`` can be imported, so its threshold, version and diagnostics apply.

One run per proposal, each in its own directory, because the runner writes ``work/`` beside
``problem/`` and the server its provenance inside it.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import shutil
from pathlib import Path

#: What the scripted fix tells the server. The judge reads it as any other justification.
JUSTIFICATION = (
    "Rate laws distilled by symbolic regression (PySR) from a neural vector field fitted to "
    "these data (gemot.neural_ode). The named constants are seeded at their regressed "
    "values and refit here; agreement with the field, not with the data, chose the expressions."
)


def score_proposals(
    proposals: Path | str,
    example_dir: Path | str,
    out: Path | str,
    *,
    n_starts: int = 50,
    only: tuple[str, ...] = (),
) -> list[dict]:
    """Run the judge on every proposal in ``proposals`` (the route's ``proposals.json``) for the
    example in ``example_dir``; one row per proposal, also written to ``<out>/scores.json``.
    ``only`` restricts the run to proposals with those tags (``selected``, ``c7``, ...)."""
    from ..runner.cli import run_scenario  # noqa: PLC0415
    from ..scenario import Scenario, build_true_model  # noqa: PLC0415

    document = json.loads(Path(proposals).read_text())
    example_dir, out = Path(example_dir), Path(out)
    template = _example_scenario(example_dir)
    rows = []
    for proposal in document.get("proposals", []):
        record = proposal.get("record") or {}
        tag = str(record.get("tag") or proposal["model_name"])
        if only and tag not in only:
            continue
        run_dir = out / proposal["model_name"]
        _stage_example(example_dir, run_dir)
        fix = _fix_for(proposal["antimony"], build_true_model)
        if template is not None:
            scenario = dataclasses.replace(
                template,
                name=proposal["model_name"],
                problem_dir=run_dir / "problem",
                scripted_fix=fix,
                fix_summary=f"neural route proposal {tag}",
            )
        else:
            scenario = Scenario(
                name=proposal["model_name"],
                problem_dir=run_dir / "problem",
                scripted_fix=fix,
                fix_summary=f"neural route proposal {tag}",
            )
        code = run_scenario(scenario, ["--driver", "scripted", "--n-starts", str(int(n_starts))])
        result_path = run_dir / "work" / "result.json"
        result = json.loads(result_path.read_text()) if result_path.is_file() else {}
        rows.append(
            {
                "model_name": proposal["model_name"],
                "tag": tag,
                "n_constants": proposal.get("n_constants"),
                "complexity_budget": record.get("complexity_budget"),
                "field_r2": record.get("field_r2"),
                "exit_code": code,
                "success": result.get("success"),
                "final_nll": result.get("final_nll"),
                "k": result.get("k"),
                "bic": result.get("bic"),
                "delta_bic": result.get("delta_bic"),
                "validation": result.get("validation"),
                "result": str(result_path),
            }
        )
        print(
            f"[judge] {proposal['model_name']}: success {result.get('success')} ΔBIC {result.get('delta_bic')} "
            f"nll {result.get('final_nll')} k {result.get('k')}"
        )
    out.mkdir(parents=True, exist_ok=True)
    (out / "scores.json").write_text(
        json.dumps(
            {
                "proposals": str(proposals),
                "example": str(example_dir),
                "n_starts": n_starts,
                "scores": rows,
            },
            indent=1,
            default=str,
        )
        + "\n"
    )
    return rows


def _fix_for(antimony: str, build_true_model):
    """The scripted fix: one ``make_changes`` with the proposal's model, as the canonical builds do."""

    async def fix(call, n_starts: int) -> None:
        await build_true_model(
            call, antimony=antimony, justification=JUSTIFICATION, n_starts=n_starts
        )

    return fix


def _stage_example(example_dir: Path, run_dir: Path) -> None:
    """A private copy of the example for one run: the problem with its reference and protocol, the
    held-out set beside it; never the compile cache, an earlier run's provenance or its ``work/``."""
    shutil.copytree(
        example_dir / "problem",
        run_dir / "problem",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns(".petab_mcp", ".amici_build", "__pycache__"),
    )
    if (example_dir / "validation").is_dir():
        shutil.copytree(example_dir / "validation", run_dir / "validation", dirs_exist_ok=True)
    shutil.rmtree(run_dir / "work", ignore_errors=True)


def _example_scenario(example_dir: Path):
    """The example's own ``Scenario`` -- its threshold, version, diagnostics -- from its ``run.py``,
    or ``None`` when there is none to import."""
    run_py = example_dir / "run.py"
    if not run_py.is_file():
        return None
    spec = importlib.util.spec_from_file_location(f"{example_dir.name}_run", run_py)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception:  # noqa: BLE001 -- an example that will not import here scores under the defaults
        return None
    return getattr(module, "SCENARIO", None)
