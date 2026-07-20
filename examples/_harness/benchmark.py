"""Download the Benchmark-Models-PEtab collection and MATERIALIZE each zero-shot example's problem.

The zero-shot benchmark examples do NOT vendor their PEtab problems in the repo. Instead each
example ships only a small config (its ``run.py``: benchmark name, which parameters ship, the true
reference model, and a ``known_constants`` prompt note). ``examples/setup.py`` downloads the pinned
collection once and calls :func:`materialize` for every example, which writes ``<example>/problem/``
by copying the REAL published tables **verbatim**:

  * measurements / observables — verbatim (petab's ``flatten_timepoint_specific_output_overrides``
    is applied first when the benchmark uses per-measurement observable/noise parameter overrides,
    which AMICI cannot import otherwise);
  * conditions — verbatim, including columns that are model PARAMETERS (per-condition overrides);
    the server keeps those out of the parameter table (reconcile.condition_column_symbols), so no
    rewriting onto species is needed;
  * an EMPTY SBML (so ``describe_problem`` reports model_initialized: false — the agent rebuilds it);
  * a SUBSET ``parameters.tsv`` — only the KNOWN shipped parameters (observation scalings/noise,
    known fixed constants, estimated ICs); the agent introduces the estimated kinetics itself.

``reference.json`` (the BIC bar — the true structure fit to the data) is NOT materialized: it is the
committed answer key, regenerated only via ``setup.py --refit`` (see :func:`write_reference_for`).

The collection is pinned to ``COLLECTION_PIN`` for reproducibility; override its location with
``$BENCHMARK_MODELS_PETAB`` (a local clone) if offline.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

COLLECTION_URL = "https://github.com/Benchmarking-Initiative/Benchmark-Models-PEtab"
# Pinned so the copied tables (and thus the committed reference.json) are reproducible.
COLLECTION_PIN = "1763dfcd66d25bfd97492cfedf6a730472c383d0"

_DEFAULT_CLONE = Path(os.environ.get("BENCHMARK_MODELS_PETAB_CACHE",
                                     Path.home() / ".cache" / "petab_mcp" / "Benchmark-Models-PEtab"))


def download_collection(clone_dir: Path | None = None, pin: str = COLLECTION_PIN) -> Path:
    """Clone (or update) the Benchmark-Models-PEtab collection at the pinned commit and return the
    directory that contains the per-model subdirs (``.../Benchmark-Models``).

    Honours ``$BENCHMARK_MODELS_PETAB`` (an existing clone's ``Benchmark-Models`` dir or its parent)
    so an offline / pre-cloned checkout is used as-is without network access."""
    env = os.environ.get("BENCHMARK_MODELS_PETAB")
    if env:
        p = Path(env)
        return p if p.name == "Benchmark-Models" else p / "Benchmark-Models"

    clone = Path(clone_dir) if clone_dir else _DEFAULT_CLONE
    models = clone / "Benchmark-Models"
    if not (clone / ".git").is_dir():
        clone.parent.mkdir(parents=True, exist_ok=True)
        print(f"[setup] cloning {COLLECTION_URL} -> {clone}")
        subprocess.run(["git", "clone", COLLECTION_URL, str(clone)], check=True)
    # Check out the pin (fetch it first if the local clone predates it).
    have = subprocess.run(["git", "-C", str(clone), "cat-file", "-e", f"{pin}^{{commit}}"]).returncode
    if have != 0:
        subprocess.run(["git", "-C", str(clone), "fetch", "--quiet", "origin", pin], check=True)
    subprocess.run(["git", "-C", str(clone), "checkout", "--quiet", pin], check=True)
    print(f"[setup] collection pinned at {pin[:12]}")
    return models


def benchmark_dir(name: str, collection: Path | None = None) -> Path:
    """Directory of one benchmark model inside the (downloaded) collection."""
    models = collection if collection is not None else download_collection()
    d = models / name
    if not (d / f"{name}.yaml").is_file():
        raise FileNotFoundError(f"benchmark {name!r} not found under {models}")
    return d


def _load_problem(name: str, collection: Path | None):
    """The published petab problem, flattened if it uses per-measurement output overrides."""
    import petab.v1 as petab

    prob = petab.Problem.from_yaml(str(benchmark_dir(name, collection) / f"{name}.yaml"))
    if _needs_flatten(prob):
        print(f"[setup] {name}: flattening per-measurement output overrides (petab)")
        petab.flatten_timepoint_specific_output_overrides(prob)
    return prob


def _needs_flatten(prob) -> bool:
    """True when the measurement table carries per-measurement observable/noise parameter overrides
    that differ within an observable — AMICI cannot import these without flattening."""
    from petab.v1.C import NOISE_PARAMETERS, OBSERVABLE_ID, OBSERVABLE_PARAMETERS

    mdf = prob.measurement_df
    for col in (OBSERVABLE_PARAMETERS, NOISE_PARAMETERS):
        if col not in mdf.columns:
            continue
        # a string override (a parameter id / expression) that varies within one observable needs it
        sub = mdf[[OBSERVABLE_ID, col]].dropna()
        sub = sub[sub[col].apply(lambda v: isinstance(v, str))]
        if not sub.empty and (sub.groupby(OBSERVABLE_ID)[col].nunique() > 1).any():
            return True
    return False


def _write_problem_yaml(problem_dir: Path) -> None:
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


# The model id stamped on the shipped empty SBML. It surfaces to the agent via describe_problem
# (`model_id`), so it must NOT be the benchmark name — `Brannmark_JBC2010` etc. would hand the agent
# the paper (author + journal + year) and defeat the zero-shot rediscovery. A neutral placeholder;
# the agent's own set_model replaces it with its own id anyway.
ANON_MODEL_ID = "zero_shot_problem"


def _write_empty_sbml(problem_dir: Path) -> None:
    import libsbml

    doc = libsbml.SBMLDocument(3, 2)
    m = doc.createModel()
    m.setId(ANON_MODEL_ID)
    m.setMetaId(ANON_MODEL_ID)
    libsbml.writeSBMLToFile(doc, str(problem_dir / "model.xml"))
    (problem_dir / "model.ant").write_text(
        "// Zero-shot problem: ships without a model — the agent builds it via set_model.\n"
        f"model {ANON_MODEL_ID}\nend\n")


def materialize(*, name: str, shipped_param_ids: list[str], example_dir: Path,
                shipped_param_overrides: dict[str, dict] | None = None,
                rename_ids: dict[str, str] | None = None,
                preequilibration: dict[str, str] | None = None,
                collection: Path | None = None) -> dict[str, Any]:
    """Write ``example_dir/problem/`` from the pinned collection — the fast, no-fit path run by
    ``setup.py``. Copies the published measurements/observables/conditions VERBATIM (flattening
    per-measurement output overrides when required), ships an empty SBML + the SUBSET parameter
    table (``shipped_param_ids``) + problem.yaml. Does NOT touch reference.json.

    ``shipped_param_overrides`` applies per-parameter cell overrides to the shipped rows — the
    shipped table is a CURATED subset, not verbatim: e.g. an observation sigma the example ships
    FIXED (``{"sigma": {"estimate": 0}}``) even though the benchmark estimates it, so k counts only
    the kinetics the agent introduces.

    ``rename_ids`` substring-relabels IDs across the observable/measurement/parameter tables so no
    model/paper name reaches the agent — e.g. perelson's PEtab-export observable id
    ``task0_model0_perelson1_V`` -> ``viral_load`` (which also relabels the derived
    ``sd_task0_model0_perelson1_V`` and ``noiseParameter1_...``). A PURE relabel: the data and the
    fit are unchanged, so the committed reference.json is unaffected.

    ``preequilibration`` ``{"preeq_condition_id": ..., "gate_column": ...}`` forces every measurement
    to preequilibrate to a basal steady state. It adds ``gate_column`` to the condition table (1 in
    every real/simulation condition, 0 in a synthesized basal preeq condition), so the model's
    time-dependent stimulus can be switched OFF during preequilibration; the model then RELAXES to
    the basal steady state rather than hard-coding it as closed-form ICs. Needed when the shipped
    data starts at a stimulus-free steady state whose IC↔rate-constant constraint the agent's freely
    built model would otherwise lack (see fiedler)."""
    from petab.v1.C import CONDITION_NAME

    problem_dir = Path(example_dir) / "problem"
    problem_dir.mkdir(parents=True, exist_ok=True)
    prob = _load_problem(name, collection)

    def _relabel(v):
        if not isinstance(v, str):
            return v
        for old, new in (rename_ids or {}).items():
            v = v.replace(old, new)
        return v

    if rename_ids:
        # Relabel author-bearing / export-artifact IDs (see rename_ids). Observable + measurement
        # tables here; the shipped parameter ids are relabelled after subsetting (below), since
        # shipped_param_ids references the ORIGINAL collection ids.
        prob.observable_df.index = prob.observable_df.index.map(_relabel)
        for _col in prob.observable_df.columns:
            prob.observable_df[_col] = prob.observable_df[_col].map(_relabel)
        for _col in prob.measurement_df.columns:
            prob.measurement_df[_col] = prob.measurement_df[_col].map(_relabel)

    if preequilibration:
        from petab.v1.C import PREEQUILIBRATION_CONDITION_ID
        # Every measurement preequilibrates at the synthesized basal condition (added to `cond` below).
        prob.measurement_df[PREEQUILIBRATION_CONDITION_ID] = preequilibration["preeq_condition_id"]

    prob.measurement_df.to_csv(problem_dir / "measurements.tsv", sep="\t", index=False)
    prob.observable_df.reset_index().to_csv(problem_dir / "observables.tsv", sep="\t", index=False)
    # Conditions verbatim (parameter columns included — reconcile keeps them out of the parameter
    # table). Drop only the free-text conditionName so the reconcile symbol scan can't latch onto it.
    cond = prob.condition_df.reset_index()
    cond = cond.drop(columns=[c for c in (CONDITION_NAME,) if c in cond.columns])
    if preequilibration:
        import pandas as pd
        from petab.v1.C import CONDITION_ID
        gate, preeq_id = preequilibration["gate_column"], preequilibration["preeq_condition_id"]
        cond[gate] = 1  # every real (simulation) condition runs with the stimulus ON
        # Synthesized basal preequilibration condition: stimulus OFF (gate=0) and every other input 0.
        preeq_row = {c: (preeq_id if c == CONDITION_ID else 0) for c in cond.columns}
        cond = pd.concat([cond, pd.DataFrame([preeq_row])], ignore_index=True)
    cond.to_csv(problem_dir / "conditions.tsv", sep="\t", index=False)

    shipped = prob.parameter_df.loc[shipped_param_ids].copy()
    # PEtab's nominalValue/bounds are floats, but a shipped SUBSET whose values all happen to be
    # integral makes pandas infer int64 for that column -- then a float override (or a synthesized
    # row's float cell) raises under pandas 2. Coerce them to float first. The materialized table
    # is gitignored, so this cosmetic widening (e.g. 11000 -> 11000.0) never touches a commit.
    for col in ("nominalValue", "lowerBound", "upperBound"):
        if col in shipped.columns:
            shipped[col] = shipped[col].astype(float)
    for pid, cells in (shipped_param_overrides or {}).items():
        for col, val in cells.items():
            shipped.loc[pid, col] = val
    # Synthesizing a new row (a pid not in the subset) upcasts the int estimate column via NaN;
    # restore it so estimate writes as 0/1, not 0.0/1.0.
    if "estimate" in shipped.columns and not shipped["estimate"].isna().any():
        shipped["estimate"] = shipped["estimate"].astype("int64")
    shipped = shipped.reset_index()
    if rename_ids:  # relabel the shipped ids to match the observable/measurement relabel above
        shipped["parameterId"] = shipped["parameterId"].map(_relabel)
    # Keep the published objectivePrior columns VERBATIM (only schwen ships any). A prior can only
    # ride a GIVEN parameter — a shipped one the agent references by name (e.g. schwen's ini_R1 /
    # ini_R2fold receptor ICs). The benchmark's priors on the DISCOVERED kinetics can't be carried:
    # the agent introduces those fresh, reconcile adds them prior-free, and attaching a prior by name
    # would leak structure — so priors on agent-introduced parameters are inherently outside a
    # zero-shot build. Both the reference (derive-via-reconcile) and the candidate load this table,
    # so whatever priors ship here apply to BOTH sides and ΔBIC stays a fair head-to-head.
    keep = [c for c in ["parameterId", "parameterScale", "lowerBound", "upperBound",
                        "nominalValue", "estimate", "objectivePriorType", "objectivePriorParameters"]
            if c in shipped.columns]
    shipped[keep].to_csv(problem_dir / "parameters.tsv", sep="\t", index=False)

    _write_empty_sbml(problem_dir)
    _write_problem_yaml(problem_dir)
    print(f"[setup] {name}: materialized {len(prob.measurement_df)} measurements, "
          f"{len(prob.observable_df)} observables, {len(prob.condition_df)} condition(s); "
          f"shipped params {shipped_param_ids}")
    return {"measurements": len(prob.measurement_df), "observables": len(prob.observable_df),
            "conditions": len(prob.condition_df)}


def write_reference_for(*, name: str, true_ant: str, example_dir: Path,
                        n_starts: int = 50, seed_at_nominal: bool = False,
                        estimate_ics: bool = False, collection: Path | None = None) -> dict[str, Any]:
    """Regenerate the committed ``reference.json`` (the BIC bar): fit the TRUE structure to the real
    data. Run only via ``setup.py --refit`` — slow, and its output is committed.

    The reference parameter table is DERIVED, not hand-written: we load the materialized shipped
    problem, swap in the TRUE model, and run the server's OWN :func:`reconcile` — the exact step the
    candidate's ``set_model`` runs. So the reference optimises over the identical parameter space the
    agent gets (shipped params as shipped; agent-introduced kinetics on the server's default bounds;
    ICs fixed or estimated per ``estimate_ics``), and a faithful rebuild lands at ΔBIC ≈ 0 by
    construction. No per-example nominal/bounds table to maintain — the only inputs are the TRUE
    Antimony, the shipped tables, and the scenario's ``estimate_ics``.
    """
    import petab.v1 as petab
    from petab.v1.models.sbml_model import SbmlModel

    from _harness.simulate import write_reference

    from petab_mcp import reconcile as R

    problem_dir = Path(example_dir) / "problem"
    # The materialized shipped problem (pristine collection tables + shipped parameter subset +
    # empty SBML) — exactly what the server hands the agent. Swap in the true model and reconcile.
    prob = petab.Problem.from_yaml(str(problem_dir / "problem.yaml"))
    prob.model = SbmlModel.from_antimony(true_ant)
    R.reconcile(prob, estimate_ics=estimate_ics)
    # Fit the RECONCILED model (its <sid>_init params etc. now match the reconciled parameter table).
    reconciled_ant = prob.model.to_antimony()
    return write_reference(
        reconciled_ant, prob.parameter_df, prob.condition_df, prob.observable_df,
        prob.measurement_df, build_dir=Path(example_dir) / ".gen_build" / "truthfit",
        model_name=f"{name}_truthfit", path=problem_dir / "reference.json",
        n_starts=n_starts, seed_at_nominal=seed_at_nominal,
    )
