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
  * an EMPTY SBML (so ``describe_problem`` reports model_authored: false — the agent rebuilds it);
  * a SUBSET ``parameters.tsv`` — only the KNOWN shipped parameters (observation scalings/noise,
    known fixed constants, estimated ICs); the agent introduces the estimated kinetics itself.
    A published-FIXED value is shipped only when it is genuine experimental SETUP. One that instead
    encodes a mechanistic VERDICT of the published fit — a link fixed at exactly 0 ("this route does
    not exist"), a ratio fixed at exactly 1 ("these two were tested and found equivalent"), or a rate
    railed onto a bound ("the data cannot identify this") — is deliberately
    WITHHELD, because naming it hands the agent the answer. Withholding alone would inflate the
    candidate's ``k`` (reconcile mints the symbol estimated), so each kind gets a leak-free repair:
    a term fixed at exactly 0 or 1 is PRUNED from the example's ``TRUE_ANT`` (numerically exact, so the pruned
    model IS the effective published model), and a bound-railed one is FREED in the reference via
    ``write_reference_for(free_parameters=...)`` so both sides estimate it. See BENCHMARK_NOTES.

``reference.json`` (the BIC bar — the true structure fit to the data) is NOT materialized: it is the
committed answer key, regenerated only via ``setup.py --refit`` (see :func:`write_reference_for`).

The collection is pinned to ``COLLECTION_PIN`` for reproducibility; override its location with
``$BENCHMARK_MODELS_PETAB`` (a local clone) if offline.
"""

from __future__ import annotations

import math
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
# the agent's own re-author replaces it with its own id anyway.
ANON_MODEL_ID = "zero_shot_problem"


def _write_empty_sbml(problem_dir: Path) -> None:
    import libsbml

    doc = libsbml.SBMLDocument(3, 2)
    m = doc.createModel()
    m.setId(ANON_MODEL_ID)
    m.setMetaId(ANON_MODEL_ID)
    libsbml.writeSBMLToFile(doc, str(problem_dir / "model.xml"))
    (problem_dir / "model.ant").write_text(
        "// Zero-shot problem: ships without a model — the agent builds it via make_changes.\n"
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


def _apply_param_overrides(problem, name: str, param_overrides: dict[str, dict] | None) -> None:
    """Apply ``write_reference_for``'s ``param_overrides`` to a loaded published problem, in place.

    Factored out so BOTH reference stages (the published fit and, when ``free_parameters`` is set,
    the freed fit) are guaranteed to see the same table: fitting them under different tables would
    make the ``min(freed, constrained)`` comparison meaningless, since the two numbers would come
    from different problems.
    """
    if not param_overrides:
        return
    unknown = sorted(set(param_overrides) - set(problem.parameter_df.index))
    if unknown:
        raise KeyError(f"{name}: param_overrides names parameters absent from the published "
                       f"table: {unknown}")
    for col in ("nominalValue", "lowerBound", "upperBound"):
        if col in problem.parameter_df.columns:
            problem.parameter_df[col] = problem.parameter_df[col].astype(float)
    for pid, cells in param_overrides.items():
        for col, val in cells.items():
            problem.parameter_df.loc[pid, col] = val
    if "estimate" in problem.parameter_df.columns:
        problem.parameter_df["estimate"] = problem.parameter_df["estimate"].astype("int64")
    print(f"[setup] {name}: reference is NOT pristine — param_overrides applied to "
          f"{sorted(param_overrides)} (see BENCHMARK_NOTES.md)")


def write_reference_for(*, name: str, example_dir: Path, n_starts: int = 50,
                        n_workers: int | None = None,
                        per_start_timeout: float | None = None,
                        collection: Path | None = None,
                        free_parameters: list[str] | None = None,
                        param_overrides: dict[str, dict] | None = None) -> dict[str, Any]:
    """Regenerate the committed ``reference.json`` (the BIC bar): fit the PRISTINE published PEtab
    problem EXACTLY as it ships. Run only via ``setup.py --refit`` — slow, and its output is committed.

    The reference is the benchmark's OWN model — the shipped SBML + the published parameter table
    (the paper's true free/fixed split, bounds and priors; ``nominalValue`` = the published optimum)
    + the published data — loaded straight from the collection and fit with the server's optimizer.
    It does NOT go through a hand-authored ``TRUE_ANT`` or the server's :func:`reconcile`: the bar is
    the genuine ground truth, independent of the IC / parametrisation conventions the CANDIDATE is
    scored under. ΔBIC then measures how close the agent's rebuild gets to the published fit, rather
    than whether it reproduces a reconcile-mirrored reference (which cancels shared handicaps and can
    hide a mis-scaled reference — see issue #90 / crauste).

    The candidate still fits the SAME data (materialize copies the collection's measurements/
    observables verbatim), so ``n`` matches; ``k`` is the published table's estimated-parameter count.

    ``free_parameters`` FREES parameters the published table froze (``estimate`` 0 -> 1) before
    fitting, so they are estimated on BOTH sides and counted in ``k``. This is for a published
    constant that is not knowledge but an artefact of the published fit — a rate the optimizer drove
    onto a bound and was then frozen there (see ``alkan``: three p53-apoptosis rates railed at their
    lower bound, two checkpoint terms at their upper bound). Such a value cannot be SHIPPED to the
    agent — a rate constant pinned at its floor states "this route is inactive", which is the
    mechanism to rediscover, not experimental setup — so the honest alternative is to let both sides
    estimate it. Freeing it here keeps the candidate's ``k`` equal to the reference's while leaking
    nothing; the parameter is estimated within the PUBLISHED bounds, so the bar stays anchored to the
    published problem's parameter space. Use it only for a demonstrably bound-railed / unidentified
    published constant, and record the choice in BENCHMARK_NOTES.

    AMICI compiles a non-estimated parameter as a model CONSTANT, so a build directory produced
    under a different free/fixed split is silently wrong for this one. The freed fit's build dir is
    therefore keyed to a hash of ``free_parameters``, and the published fit keeps plain
    ``pristine`` (its split never varies) — so editing ``REFERENCE_FREE_PARAMS`` just lands in a
    fresh directory instead of reusing an incompatible model.

    ``per_start_timeout`` sets the Fides per-start wall-clock cap for BOTH stages (they must be fit
    under the same cap, or the min() below would compare a well-converged fit against a truncated
    one). It is a per-example number because the time a start needs to converge is a property of the
    model: the suite spans 0.2 s starts (``perelson``) to 4 h starts (``raimundez``), and a cap that
    binds does not fail loudly — Fides returns its best-so-far, so the reference silently records a
    looser bar and every ΔBIC measured against it looks too good. See ``REFIT_PER_START_TIMEOUT``.

    ``param_overrides`` ``{parameter_id: {column: value}}`` is a deliberate, per-example EXCEPTION to
    "pristine": it edits the published parameter table before the reference is fit. Use it ONLY to
    repair a published NOISE MODEL that ΔBIC cannot be computed against, and only when the SAME
    override is applied to the candidate's table via ``materialize(shipped_param_overrides=...)`` —
    otherwise the two sides are scored under different noise models and ΔBIC is meaningless. It must
    NOT be used to carry the candidate's IC / parametrisation conventions into the reference: that is
    the reconcile-mirroring #90 warns about, and it cancels shared handicaps instead of measuring
    them. Every use needs an entry in ``examples/BENCHMARK_NOTES.md`` (see oliveira).
    """
    import json

    from petab.v1.C import ESTIMATE

    from .simulate import write_reference_from_problem

    problem_dir = Path(example_dir) / "problem"
    build_dir = Path(example_dir) / ".gen_build"
    # The pristine published problem (shipped SBML + full published parameter table + data),
    # flattened if it uses per-measurement output overrides — the same load materialize uses.
    problem = _load_problem(name, collection)
    # param_overrides first: it repairs the published table (a broken noise model), and every
    # subsequent decision -- which rows count as already-estimated, which bounds a freed row gets
    # -- must see the repaired table. It is applied to BOTH reference stages below via the same
    # helper, since fitting them under different tables would make min(freed, constrained)
    # compare two different problems.
    _apply_param_overrides(problem, name, param_overrides)

    if not free_parameters:
        return write_reference_from_problem(
            problem, build_dir=build_dir / "pristine",
            model_name=f"{name}_pristine", path=problem_dir / "reference.json",
            n_starts=n_starts, n_workers=n_workers, per_start_timeout=per_start_timeout,
        )

    missing = [p for p in free_parameters if p not in problem.parameter_df.index]
    if missing:
        raise KeyError(f"{name}: free_parameters not in the published table: {missing}")
    # Freeing an ALREADY-estimated parameter is a no-op that silently degrades the bar: `freed`
    # would equal `problem`, both stages would fit the SAME problem with different random draws,
    # and reporting min() of two independent multistarts lowers the committed nll by pure optimizer
    # luck (at double the refit cost) while printing a "freeing N published-fixed parameter(s)"
    # line about a parameter that was never fixed.
    # PEtab only requires bounds on an ESTIMATED row, so a published FIXED row may leave them
    # blank. Freeing such a row hands pypesto a NaN box: Fides either raises deep inside the
    # optimizer with no mention of REFERENCE_FREE_PARAMS, or samples starts from it and commits a
    # garbage nll. The min(freed, constrained) argument also rests on the constrained optimum being
    # a FEASIBLE point of the freed problem, which needs nominalValue inside the bounds.
    from petab.v1.C import LOWER_BOUND, NOMINAL_VALUE, UPPER_BOUND

    unbounded = []
    for p in free_parameters:
        row = problem.parameter_df.loc[p]
        try:
            lo, hi = float(row[LOWER_BOUND]), float(row[UPPER_BOUND])
            nom = float(row[NOMINAL_VALUE])
        except (KeyError, TypeError, ValueError):
            unbounded.append(f"{p} (unreadable bounds/nominalValue)")
            continue
        if not (math.isfinite(lo) and math.isfinite(hi) and math.isfinite(nom)):
            unbounded.append(f"{p} (non-finite bounds/nominalValue)")
        elif not lo <= nom <= hi:
            unbounded.append(f"{p} (nominalValue {nom} outside [{lo}, {hi}])")
    if unbounded:
        raise ValueError(
            f"{name}: free_parameters need finite PUBLISHED bounds bracketing their nominalValue, "
            f"but {unbounded} do not. Freeing them would fit inside a NaN/degenerate box and the "
            "constrained optimum would not be a feasible point of the freed problem.")
    already = [p for p in free_parameters
               if int(float(problem.parameter_df.loc[p, ESTIMATE])) == 1]
    if already:
        raise ValueError(
            f"{name}: free_parameters must name parameters the published table FIXES, but "
            f"{already} are already estimated. Freeing them would make both reference stages fit "
            "the same problem and take the luckier of two multistarts as the bar.")

    # Fit BOTH the published problem and the freed one, and take the better.
    #
    # Freeing parameters can only ENLARGE the feasible set, so the freed optimum is bounded above by
    # the constrained one: the constrained fit's parameter vector is itself a feasible point of the
    # freed problem (the freed parameters sitting at their published values). A freed fit that scores
    # worse is therefore under-converged, not a better bar — and it is easy to get, because freeing
    # adds dimensions to an already hard multistart (alkan: 20 starts reach 3653.4 in 56-D but only
    # 4002.2 in 65-D, which would hand every candidate a free ~765 BIC win). Reporting the better of
    # the two keeps the bar an ACHIEVABLE point of the freed parameter space and monotone in the
    # freeing, whichever way the optimizer happens to land.
    #
    # Both intermediate fits write to SCRATCH paths, never to the committed reference.json: that
    # file is written exactly once, at the end, from the reconciled result. Writing the constrained
    # fit there first would leave a silently WRONG answer key behind if the freed fit then failed —
    # the constrained nll paired with the UN-freed k, i.e. a k short by len(free_parameters) and a
    # systematic k*ln(n) offset in every future ΔBIC, with no error raised anywhere.
    scratch = build_dir / "_reference_stages"
    scratch.mkdir(parents=True, exist_ok=True)
    ref_fixed = write_reference_from_problem(
        problem, build_dir=build_dir / "pristine",
        model_name=f"{name}_pristine", path=scratch / "published.json",
        n_starts=n_starts, n_workers=n_workers, per_start_timeout=per_start_timeout,
    )
    freed = _load_problem(name, collection)
    _apply_param_overrides(freed, name, param_overrides)  # same table on both stages
    freed.parameter_df.loc[free_parameters, ESTIMATE] = 1
    print(f"[setup] {name}: freeing {len(free_parameters)} published-fixed parameter(s) for the "
          f"reference fit: {sorted(free_parameters)}")
    # Key the freed build directory to the free/fixed SPLIT. AMICI compiles a non-estimated
    # parameter as a model CONSTANT, so a build dir produced under a different split is silently
    # wrong for this one — it either raises ("does not support estimating ...") or, worse, fits a
    # differently-compiled model and returns a slightly different optimum. Hashing the freed set
    # means editing REFERENCE_FREE_PARAMS just lands in a fresh directory. (Stage 1 always fits the
    # PUBLISHED split, so plain `pristine` is always valid for it.)
    import hashlib

    split = hashlib.sha1("\0".join(sorted(free_parameters)).encode()).hexdigest()[:8]
    ref_free = write_reference_from_problem(
        freed, build_dir=build_dir / f"pristine_freed_{split}",
        model_name=f"{name}_pristine_freed", path=scratch / "freed.json",
        n_starts=n_starts, n_workers=n_workers, per_start_timeout=per_start_timeout,
    )

    # k is the FREED count either way — that is the parameter space the candidate is scored in —
    # but every OTHER field has to come from the stage that actually won, and be carried rather
    # than dropped. Rebuilding the dict from just {nll, k, n} silently discarded the
    # `noise_sigma` / `noise_sigma_effective` the sub-call had computed, so a freed example wrote a
    # reference with no sigmas at all and the ground-truth plausibility gate then fell back to
    # `nominalValue` — a starting point, not the truth, for an estimated sigma.
    winner = ref_fixed if ref_fixed["nll"] < ref_free["nll"] else ref_free
    ref = {**winner, "k": ref_free["k"], "n": ref_free["n"]}
    if ref_fixed["nll"] < ref_free["nll"]:
        print(f"[setup] {name}: the freed multistart under-converged "
              f"({ref_free['nll']:.4f} vs {ref_fixed['nll']:.4f} with the same parameters held at "
              f"their published values); keeping the better, achievable point. Raise REFIT_N_STARTS "
              f"if this persists.")
    (problem_dir / "reference.json").write_text(json.dumps(ref, indent=2) + "\n")
    return ref
