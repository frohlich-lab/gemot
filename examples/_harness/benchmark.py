"""Shared builder for the zero-shot BENCHMARK examples (real PEtab Benchmark-Models data).

Each benchmark example ships the REAL published experimental data (measurements/observables/
conditions) with its MODEL withheld (an empty SBML → ``describe_problem`` reports
model_initialized: false); the agent must rebuild the mechanism via ``set_model``. The BIC
reference is the TRUE benchmark structure fit to that data with the harness Fides fitter, using
the benchmark's OWN parameter table — so its estimated/fixed split (and thus k) matches what a
candidate produces: the agent introduces the estimated kinetics itself (reconcile creates them
estimated), while the KNOWN fixed constants + observation noise are SHIPPED and inherited.

A model's ``generate_data.py`` only declares its benchmark name, its true model (Antimony), and
which parameter ids ship with the model-less problem; everything else lives here.

Regenerate any example (needs amici + petab + pypesto) with:
    python examples/<name>/generate_data.py

The benchmark PEtab problem is read from $BENCHMARK_MODELS_PETAB (a clone of
Benchmarking-Initiative/Benchmark-Models-PEtab) or /tmp/Benchmark-Models-PEtab; if neither
exists it is shallow-cloned into the example's .gen_build/.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

BENCHMARK_URL = "https://github.com/Benchmarking-Initiative/Benchmark-Models-PEtab"


def benchmark_dir(name: str, scratch: Path) -> Path:
    """Locate the benchmark problem directory, cloning the collection if necessary."""
    candidates: list[Path] = []
    env = os.environ.get("BENCHMARK_MODELS_PETAB")
    if env:
        candidates += [Path(env) / "Benchmark-Models" / name, Path(env) / name, Path(env)]
    candidates += [Path("/tmp/Benchmark-Models-PEtab") / "Benchmark-Models" / name]
    for cand in candidates:
        if (cand / f"{name}.yaml").is_file():
            return cand
    clone = scratch / "Benchmark-Models-PEtab"
    target = clone / "Benchmark-Models" / name
    if not (target / f"{name}.yaml").is_file():
        scratch.mkdir(parents=True, exist_ok=True)
        print(f"[gen] cloning {BENCHMARK_URL} -> {clone}")
        subprocess.run(["git", "clone", "--depth", "1", BENCHMARK_URL, str(clone)], check=True)
    return target


def build_zero_shot_problem(*, name: str, true_ant: str, shipped_param_ids: list[str],
                            example_dir: Path, n_starts: int = 20) -> dict[str, Any]:
    """Write the model-less PEtab problem (+ empty SBML) into ``example_dir/problem`` and the
    true-structure BIC reference, from the real benchmark data. Returns the reference dict."""
    import libsbml
    import petab.v1 as petab

    from _harness.simulate import write_reference

    problem_dir = example_dir / "problem"
    scratch = example_dir / ".gen_build"
    problem_dir.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)

    bm = benchmark_dir(name, scratch)
    print(f"[gen] loading benchmark problem from {bm}")
    prob = petab.Problem.from_yaml(str(bm / f"{name}.yaml"))

    # Copy the REAL benchmark tables verbatim (published experimental data — do NOT simulate).
    prob.measurement_df.to_csv(problem_dir / "measurements.tsv", sep="\t", index=False)
    prob.observable_df.reset_index().to_csv(problem_dir / "observables.tsv", sep="\t", index=False)
    # Drop the informational ``conditionName`` column: reconcile scans condition-table *values* for
    # identifier-like tokens (reconcile.table_referenced_symbols), so a free-text name could keep
    # unrelated symbols alive and interfere with pruning/sync. Other examples drop it too.
    cond_out = prob.condition_df.reset_index()
    cond_out = cond_out.drop(columns=[c for c in ("conditionName",) if c in cond_out.columns])
    cond_out.to_csv(problem_dir / "conditions.tsv", sep="\t", index=False)
    print(f"[gen] copied {len(prob.measurement_df)} real measurements, "
          f"{len(prob.observable_df)} observables, {len(prob.condition_df)} condition(s)")

    # SHIPPED parameter table: only the known fixed constants + estimated observation noise the
    # model-less problem needs. NO estimated kinetics are shipped — the agent introduces those.
    shipped = prob.parameter_df.loc[shipped_param_ids].reset_index()
    keep = [c for c in ["parameterId", "parameterScale", "lowerBound", "upperBound",
                        "nominalValue", "estimate"] if c in shipped.columns]
    shipped[keep].to_csv(problem_dir / "parameters.tsv", sep="\t", index=False)
    print(f"[gen] shipped model-less parameters: {shipped_param_ids}")

    # Ship an EMPTY SBML model so the problem loads model-less.
    doc = libsbml.SBMLDocument(3, 2)
    m = doc.createModel()
    m.setId(name)
    m.setMetaId(name)
    libsbml.writeSBMLToFile(doc, str(problem_dir / "model.xml"))
    (problem_dir / "model.ant").write_text(
        "// Zero-shot problem: ships without a model — the agent builds it via set_model.\n"
        f"model {name}\nend\n")

    (problem_dir / "problem.yaml").write_text(
        "format_version: 1\n"
        "parameter_file: parameters.tsv\n"
        "problems:\n"
        "- sbml_files:\n"
        "  - model.xml\n"
        "  condition_files:\n"
        "  - conditions.tsv\n"
        "  measurement_files:\n"
        "  - measurements.tsv\n"
        "  observable_files:\n"
        "  - observables.tsv\n"
    )

    # BIC reference: fit the TRUE benchmark structure to the REAL data with the benchmark's own
    # parameter table (its estimated/fixed split), so k matches a faithful zero-shot candidate.
    print("[gen] fitting the true benchmark structure for the BIC reference (Fides) ...")
    ref = write_reference(
        true_ant, prob.parameter_df, prob.condition_df, prob.observable_df, prob.measurement_df,
        build_dir=scratch / "truthfit", model_name=f"{name}_truthfit",
        path=problem_dir / "reference.json", n_starts=n_starts,
    )
    print(f"[gen] wrote problem files to {problem_dir}")
    print(f"[gen] zero-shot: shipped model-less; true-model BIC reference {ref}")
    return ref
