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

``reference.json`` (the reference bar — the true structure fit to the data) is NOT materialized: it
is the committed answer key, regenerated only via ``setup.py --refit`` (see
:func:`write_reference_for`).

The collection is pinned to ``COLLECTION_PIN`` for reproducibility; override its location with
``$BENCHMARK_MODELS_PETAB`` (a local clone) if offline.
"""

from __future__ import annotations

import math
import os
import re
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


def _load_problem(name: str, collection: Path | None, flatten: bool = True):
    """The published petab problem — flattened only when AMICI genuinely requires it AND the
    caller asks (the reference fit, which hands the problem to pypesto directly).

    ``materialize`` passes ``flatten=False``: the shipped tables stay in the published compact
    form (bachmann ships 20 observables; flattening would mint 101), and the SERVER flattens a
    copy at its own AMICI boundary when needed (``petab_mcp.fitting.amici_facing_problem``).
    Sharing the server's predicate matters: a looser one — grouping by observable alone — would
    flatten three examples whose overrides only vary ACROSS conditions, which AMICI takes
    natively.
    """
    import petab.v1 as petab

    from petab_mcp.fitting import needs_flattening

    prob = petab.Problem.from_yaml(str(benchmark_dir(name, collection) / f"{name}.yaml"))
    if flatten and needs_flattening(prob):
        print(f"[setup] {name}: flattening timepoint-specific output overrides (petab)")
        petab.flatten_timepoint_specific_output_overrides(prob)
    return prob


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


def _placeholder_re(oid: str) -> "re.Pattern[str]":
    """The observation-parameter placeholders belonging to observable ``oid``.

    PEtab defines a placeholder as ``<observable|noise>ParameterN_<observableId>``, so the
    observable id is a SUFFIX -- which makes both boundaries load-bearing and easy to get wrong
    in only one of the places that need this. Without the trailing ``\\b``, observable ``pAkt``
    matches inside ``observableParameter1_pAkt_tot`` and mangles a different observable's
    placeholder; without the leading one, a longer prefix could match into this id. Every site
    that renames or canonicalises placeholders goes through here so they cannot drift apart.
    """
    return re.compile(rf"\b((?:observable|noise)Parameter\d+)_{re.escape(oid)}\b")


def _merge_equivalent_observables(prob) -> dict[str, str]:
    """Merge observables that are identical up to their own placeholder suffix, returning the
    ``{old_id: new_id}`` mapping (empty when nothing merges).

    Some published encodings mint one observable per DATASET even though the per-dataset
    scaling/noise already rides the measurement table's ``observableParameters`` /
    ``noiseParameters`` columns row by row — raimundez is 79 observables that are 5 formulas.
    PEtab's placeholders (``observableParameterN_<observableId>``) resolve from those columns
    per measurement, so observables whose formulas normalize to the same shape (and share
    transformation + noise distribution) are one observable wearing many names, and the clones
    cost every consumer: the agent re-reads 79 near-identical rows per turn, the blinding needs
    79 spec entries, and a forgotten mapping errors once per clone.

    GUARDED by the same predicate that gates AMICI flattening: a merge is taken only if the
    merged problem still has at most one override per (observable, condition) pair —
    ``petab_mcp.fitting.needs_flattening`` stays False. That refuses exactly the cases where the
    clones carry information: replicate series measured in the SAME condition (bachmann's
    foldA/B/C qRT-PCR replicates, its au1/au2 dose-response gels) keep their own observables.
    Likelihood, n and k are untouched — same rows, same parameters, new labels.

    The merged id is the members' cleaned common prefix (``pEGFR_ID3_...``/``pEGFR_ID6_...`` ->
    ``pEGFR``) when that is unique, else the first member's id.
    """

    from petab.v1.C import (NOISE_DISTRIBUTION, NOISE_FORMULA, OBSERVABLE_ID,
                            OBSERVABLE_NAME, OBSERVABLE_PARAMETERS, OBSERVABLE_TRANSFORMATION)

    from petab_mcp.fitting import needs_flattening

    odf, mdf = prob.observable_df, prob.measurement_df
    if odf is None or mdf is None or OBSERVABLE_PARAMETERS not in mdf.columns:
        return {}

    def norm(oid, s):
        """Strip the observable's own placeholders so two rows differing only in WHICH
        observable they belong to compare equal."""
        return _placeholder_re(oid).sub(r"\1", str(s or ""))

    groups: dict[tuple, list[str]] = {}
    for oid, row in odf.iterrows():
        key = (norm(oid, row.get("observableFormula")), norm(oid, row.get(NOISE_FORMULA)),
               str(row.get(OBSERVABLE_TRANSFORMATION, "")), str(row.get(NOISE_DISTRIBUTION, "")))
        groups.setdefault(key, []).append(str(oid))

    mapping: dict[str, str] = {}
    taken = {str(i) for i in odf.index}
    for members in groups.values():
        if len(members) < 2:
            continue
        import os.path
        prefix = os.path.commonprefix(members).rstrip("_")
        # Cut at the first dataset-ish segment: `pEGFR_ID3_HS746T_FM` and friends share far more
        # than the biological name, and a merged id of `MMET_ID10_HS746T_FM_EGF` (the common
        # prefix of two members) is worse than no merge. Segments like ID3/ID10 mark where the
        # per-dataset suffix starts.
        parts = [s for s in prefix.split("_") if s]
        for i, seg in enumerate(parts):
            if re.fullmatch(r"ID\d*[a-z]?", seg):
                parts = parts[:i]
                break
        prefix = "_".join(parts)
        # Suffix `_obs`: a bare biological name (EGFR, pAKT) is exactly what the agent will call
        # a SPECIES, and petab lint rejects an observable id that shadows a model entity — the
        # e2e probe caught the unsuffixed form breaking every raimundez commit. The published
        # per-dataset ids only avoided this by accident of their suffixes.
        if prefix and not prefix.endswith(("_obs", "_au", "_rel", "_fold", "_count")):
            prefix = f"{prefix}_obs"
        new_id = prefix if (prefix and re.match(r"[A-Za-z_]\w*$", prefix)
                            and (prefix not in taken or prefix in members)) else members[0]
        # trial the merge on COPIES; keep it only if the flattening predicate stays quiet
        trial = {m: new_id for m in members}
        m2 = mdf.copy()
        m2[OBSERVABLE_ID] = m2[OBSERVABLE_ID].map(lambda v: trial.get(str(v), v))

        class _T:  # minimal problem view for the predicate
            measurement_df = m2

        if needs_flattening(_T()):
            continue
        mapping.update({m: new_id for m in members if m != new_id})
        taken.add(new_id)

    if not mapping:
        return {}
    # observable table: keep the FIRST row seen per merged id, placeholders renamed. Dedup on
    # the merged ids directly — checking "did a sibling already map here" against the kept list
    # missed the case where an alias row precedes its group's canonical row, which shipped a
    # duplicate observable id.
    keep, seen_new = [], set()
    for oid in map(str, odf.index):
        new = mapping.get(oid, oid)
        if new in seen_new:
            continue
        seen_new.add(new)
        keep.append(oid)
    new_odf = odf.loc[keep].copy()
    # The kept row is ONE arbitrary member of the group, so any per-dataset label it carries no
    # longer describes the merged observable. raimundez's published observableName cells are the
    # observable ids themselves (`EGFR_ID10_HS746T_FM_EGF_CET`), so keeping the first member's
    # would name the merged row after one arbitrary gel -- misleading, and a dataset-identity
    # leak on any future merging example that is not also blinded (blinding overwrites this cell,
    # which is the only reason it is harmless today). The merged id is the honest label.
    if OBSERVABLE_NAME in new_odf.columns:
        new_odf[OBSERVABLE_NAME] = [mapping.get(oid, oid) for oid in keep]
    new_index = []
    for oid in keep:
        new = mapping.get(oid, oid)
        for col in new_odf.columns:
            v = new_odf.loc[oid, col]
            if isinstance(v, str):
                new_odf.loc[oid, col] = _placeholder_re(oid).sub(rf"\1_{new}", v)
        new_index.append(new)
    new_odf.index = new_index
    new_odf.index.name = odf.index.name
    prob.observable_df = new_odf
    mdf[OBSERVABLE_ID] = mdf[OBSERVABLE_ID].map(lambda v: mapping.get(str(v), v))
    print(f"[setup] merged {len(mapping)} redundant per-dataset observables into "
          f"{len(set(mapping.values()))} (per-row overrides already carried the difference)")
    return mapping


def _collapse_blinding(blinding: dict[str, dict], merged: dict[str, str]) -> dict[str, dict]:
    """Rewrite a blinding spec through the observable merge that just happened.

    When `_merge_equivalent_observables` folds N clone observables onto one id, the spec written
    against the old ids has to follow — and the only thing in it that names an observable id is
    the observation-parameter PLACEHOLDER (`observableParameter1_<oid>`), whose suffix is the
    observable id by PEtab definition.

    So the rewrite is anchored to exactly that shape, in exactly the formula keys. The obvious
    version -- `v.replace(f"_{oid}", f"_{new}")` across every value -- is wrong three ways:

    * it rewrites the `measures` PROSE, mangling any sentence containing the id after an
      underscore, and that prose is what the agent reads;
    * it rewrites an EXPLICIT `readout` symbol, which the run-path check keys on and which is
      deliberately shared across the very observables being merged;
    * it matches SUBSTRINGS, so with both `pAkt` and `pAkt_tot` in the spec, collapsing `pAkt`
      corrupts the longer one's placeholder into `observableParameter1_pAkt_obs_tot`.

    The DERIVED readout is the one exception, and it does have to move. A spec with no `readout`
    key means `readout_<observableId>`, which :func:`_blind_observables` re-derives from the key
    it is handed -- the MERGED id -- so leaving `readout_<oldId>` in the formula makes the two
    disagree and the "must reference readout_<mergedId>" guard fires. Eleven of the blinded
    examples omit the key, so this is not hypothetical, merely unreachable while raimundez is the
    only example that merges. An explicit `readout` is left exactly as written.

    Specs that collapse onto the same merged id must agree; disagreeing ones are a curation
    error, not something to silently pick a winner for.
    """
    collapsed: dict[str, dict] = {}
    for oid, spec in blinding.items():
        new = merged.get(oid, oid)
        ph = _placeholder_re(oid)
        derived = None if spec.get("readout") else (
            re.compile(rf"\breadout_{re.escape(oid)}\b"), f"readout_{new}")
        renamed = dict(spec)
        for key in ("formula", "noise"):
            if isinstance(renamed.get(key), str):
                renamed[key] = ph.sub(rf"\1_{new}", renamed[key])
                if derived is not None:
                    renamed[key] = derived[0].sub(derived[1], renamed[key])
        existing = collapsed.setdefault(new, renamed)
        if existing != renamed:
            raise ValueError(f"observable merge: blinding specs for {oid!r} disagree "
                             f"after collapsing onto {new!r}")
    return collapsed


def _blind_observables(prob, blinding: dict[str, dict]) -> None:
    """Replace each blinded observable's MODEL-CONTENT with a ``readout_<observableId>`` symbol,
    in place, and put the biology-only description in ``observableName``.

    The published ``observableFormula`` is a structural leak: it is written in the REFERENCE
    model's own vocabulary, so shipping it verbatim hands the agent species names (blasi's
    ``x_k5k12`` acetylation states), complex composition (fujita's ``pAkt + pAkt_S6``) and even
    stoichiometry (sneyd's ``(0.9*IPR_A + 0.1*IPR_O)^4`` — four subunits, two conducting states,
    their conductances). Worse than informational: the model-less SEED declares one species per
    formula identifier, so the leak becomes v0's species list. Discovering the measurement mapping
    is part of the modelling task, so the formula the agent sees references a symbol IT must
    define (an assignment rule, ``readout_x := <its own expression>``; the ``set_assignment`` edit
    op), and ``observableName`` says WHAT was measured in assay/biology terms only.

    WHAT ``measures`` MAY SAY. The line is ASSAY SELECTIVITY vs MODEL STRUCTURE, and it has to
    be drawn deliberately, because both mistakes are expensive:

    * KEEP assay selectivity -- what the assay binds and what it does not. "TOTAL X", "selective
      for the phosphorylated form", "the antibody binds every X molecule", "a cytoplasmic
      lysate", "(active or not)". Without it the readout is not merely unblinded, it is
      UNDERDETERMINED: raimundez's reference counts each EGFR chain, so its dimeric species
      carry a factor 2, and only "binds every EGFR molecule" makes those weights derivable.
      Relative weights are not absorbable by the shipped scaling.
    * REMOVE model structure -- the SPECIES the readout sums over and where they live.
      "in whatever dimer/complex and compartment", "surface plus internalized", "summed over
      whatever cell-cycle phase". That is the mapping, and choosing it is the task.

    The failure mode in between is the dangerous one: a description that is neither leaky nor
    accurate. Replacing schwen's "surface plus internalized pool" with "the total receptor pool"
    removed the leak and silently renamed the quantity -- the reference readout is the OCCUPIED
    receptor, and total receptor is conserved by that model, so the description asked for a
    constant. Check every reworded description against the example's ``TRUE_ANT`` readout.

    ``blinding`` is ``{observableId: {"formula": <blinded formula>, "measures": <description>}}``,
    curated per example in its ``run.py`` (``OBSERVABLE_BLINDING``). Two optional keys:

    * ``"readout"`` names the readout symbol when the derived ``readout_<observableId>`` is wrong
      — the flattened benchmarks (isensee, bachmann, fiedler, ...) carry MANY observables per
      biological readout (one per gel/dataset, distinct scalings), and deriving per-observable
      symbols there would demand a hundred-odd identical mappings from the agent and seed as many
      junk species into v0. A shared ``readout_pErk`` across every pErk gel is one mapping, which
      is the biology. Must still start with ``readout_`` (the run-path check keys on the prefix).
    * ``"noise"`` rewrites ``noiseFormula`` the same way, for the examples whose noise is
      PROPORTIONAL to the signal (armistead's ``noiseParameter1_Cer_obs*Cer``): the sigma
      STRUCTURE is kept exactly — the readout equals the expression it replaces — and only the
      reference-model vocabulary goes. Without it those examples were unblindable (the guard
      below refuses a noise formula referencing a blinded-away symbol). The blinded formula is given
    EXPLICITLY, not derived: which part of a formula is assay scaling (kept — fujita's
    ``observableParameter1_pAkt_tot * readout_pAkt_tot``) and which is model content (blinded) is
    a judgement, and deriving it here would hide that judgement inside a heuristic.

    Guards, each a leak or a break this function refuses to ship:

    * an unknown observable id — a silently unblinded observable defeats the point;
    * a blinded formula that does not reference ``readout_<observableId>`` — nothing for the
      agent to define;
    * a surviving model-content symbol: every non-readout identifier in the blinded formula must
      already appear in the ORIGINAL formula (keeps scalings/offsets, and makes smuggling a new
      hint in impossible) and must not be one the blinding was supposed to remove — i.e. the
      original's identifiers minus the kept ones must all be gone;
    * a ``noiseFormula`` that references a blinded-away symbol — the leak would just move into
      the sigma column. Not hit by blasi/sneyd/fujita (placeholder sigmas), but the first
      proportional-noise example would leak silently without the check.
    """
    from petab.v1.C import NOISE_FORMULA, OBSERVABLE_FORMULA, OBSERVABLE_NAME

    from petab_mcp.reconcile import formula_identifiers

    odf = prob.observable_df
    # What counts as a MODEL entity: anything the published formulas name that is not a
    # parameter-table row (the same rule `reconcile.seed_model_antimony` uses to decide what the
    # seed declares as a species).
    pdf = getattr(prob, "parameter_df", None)
    table_rows = {str(p) for p in getattr(pdf, "index", [])}
    seed_entities = set()
    for value in odf[OBSERVABLE_FORMULA]:
        seed_entities |= formula_identifiers(value)
    seed_entities -= table_rows
    if not table_rows:
        # No parameter table to distinguish a shipped scaling from a model species — the check
        # would flag every kept symbol. Skip it rather than refuse (materialize always has one;
        # this is the hand-built-problem path).
        seed_entities = set()
    seed_entities = {i for i in seed_entities
                     if not re.fullmatch(r"(?:observable|noise)Parameter\d+_.*", i)}
    # The noise branch needs the same notion of "model entity", widened by the symbols only the
    # SIGMA column names: a species appearing nowhere but a proportional noise formula is still
    # reference-model content, and checking it against the observable-side set alone would miss it.
    noise_seed_entities = set(seed_entities)
    if table_rows and NOISE_FORMULA in odf.columns:
        for value in odf[NOISE_FORMULA]:
            noise_seed_entities |= formula_identifiers(value)
        noise_seed_entities -= table_rows
        # An OBSERVABLE ID in a noise formula is PEtab's relative-error idiom
        # (`noiseParameter1_a * a`, where petab substitutes the simulated observable), not a
        # model entity -- refusing it would reject a formula that leaks nothing.
        noise_seed_entities -= {str(i) for i in odf.index}
        noise_seed_entities = {i for i in noise_seed_entities
                               if not re.fullmatch(r"(?:observable|noise)Parameter\d+_.*", i)}
    # Once, before the loop: a shipped-but-empty observableName column is all-NaN float64, and
    # writing a string into it raises under pandas 2 rather than upcasting.
    if OBSERVABLE_NAME not in odf.columns:
        odf[OBSERVABLE_NAME] = ""
    elif odf[OBSERVABLE_NAME].dtype != object:
        odf[OBSERVABLE_NAME] = odf[OBSERVABLE_NAME].astype(object)
    for obs_id, spec in blinding.items():
        if obs_id not in odf.index:
            raise ValueError(f"observable blinding: no observable {obs_id!r} "
                             f"(have: {sorted(map(str, odf.index))})")
        readout = str(spec.get("readout") or f"readout_{obs_id}")
        if not readout.startswith("readout_"):
            raise ValueError(f"observable blinding: {obs_id!r}: readout symbol {readout!r} must "
                             "start with 'readout_' — the commit-time check keys on the prefix")
        blinded = str(spec["formula"])
        original = str(odf.loc[obs_id, OBSERVABLE_FORMULA])
        blinded_ids = formula_identifiers(blinded)
        original_ids = formula_identifiers(original)
        if readout not in blinded_ids:
            raise ValueError(f"observable blinding: {obs_id!r}: the blinded formula must "
                             f"reference {readout!r}, got {blinded!r}")
        # every readout_* is agent-defined (a template may reference more than one -- schwen's
        # ELISA form carries the intact-insulin AND fragments readouts); the commit-time check
        # keys on the prefix, so none of them is a "kept" reference-model symbol.
        kept = {i for i in blinded_ids if not i.startswith("readout_")}
        if not kept <= original_ids:
            raise ValueError(f"observable blinding: {obs_id!r}: {sorted(kept - original_ids)} "
                             "do not appear in the published formula — only existing "
                             "scaling/offset symbols may be kept")
        # The guard above stops a NEW symbol being smuggled in; this one stops an old one being
        # LEFT IN. Without it a spec of `scale * (readout_x + pAkt)` passed every check — `pAkt`
        # is in the published formula, so "kept <= original" holds — and the species leaked into
        # the shipped table AND the v0 seed, which is the whole failure this function prevents.
        # A kept symbol must be a shipped observation parameter (a scaling/offset the assay
        # gives), never a model entity.
        removed = original_ids - kept
        model_side = {i for i in kept if i in seed_entities}
        if model_side:
            raise ValueError(
                f"observable blinding: {obs_id!r}: {sorted(model_side)} are reference-model "
                "entities and must not survive the blinding — only shipped scaling/offset "
                "parameters may be kept")
        if "noise" in spec:
            if NOISE_FORMULA not in odf.columns:
                raise ValueError(f"observable blinding: {obs_id!r}: a 'noise' key was given but "
                                 "the observable table has no noiseFormula column to blind")
            noise_blinded = str(spec["noise"])
            noise_original_ids = formula_identifiers(odf.loc[obs_id, NOISE_FORMULA])
            # Same rule as the observable side, and for the same two reasons: EVERY `readout_*`
            # is agent-defined, not just the one derived from this observable id (a proportional
            # sigma may reference a shared readout the formula also uses), and a kept symbol must
            # be a shipped scaling — never a model entity. Subtracting only `{readout}` refused
            # legal multi-readout specs; omitting the model-side check let the leak the observable
            # branch prevents walk straight back in through the sigma column, which is precisely
            # what the guard below this block exists to stop.
            noise_kept = {i for i in formula_identifiers(noise_blinded)
                          if not i.startswith("readout_")}
            if not noise_kept <= noise_original_ids:
                raise ValueError(
                    f"observable blinding: {obs_id!r}: noise {sorted(noise_kept - noise_original_ids)} "
                    "do not appear in the published noiseFormula — only existing symbols may be kept")
            noise_model_side = {i for i in noise_kept if i in noise_seed_entities}
            if noise_model_side:
                raise ValueError(
                    f"observable blinding: {obs_id!r}: noise {sorted(noise_model_side)} are "
                    "reference-model entities and must not survive the blinding — only shipped "
                    "scaling/offset parameters may be kept")
            odf.loc[obs_id, NOISE_FORMULA] = noise_blinded
        if NOISE_FORMULA in odf.columns:
            noise_ids = formula_identifiers(odf.loc[obs_id, NOISE_FORMULA])
            # Two ways a sigma column leaks, and `removed` only catches the first:
            #  * it keeps a symbol the observable side just blinded away;
            #  * it names a model entity the observable formula never mentioned at all, so the
            #    entity is not in `removed` and the check waved it through -- shipping the
            #    reference-model species name straight to the agent via list_observables.
            leaked = (noise_ids & removed) | (noise_ids & noise_seed_entities)
            if leaked:
                raise ValueError(
                    f"observable blinding: {obs_id!r}: noiseFormula references "
                    f"{sorted(leaked)}, which are reference-model entities the blinding must "
                    "not ship — the leak would move into sigma. Blind the noise formula too "
                    "(the 'noise' key), or keep the symbol.")
        odf.loc[obs_id, OBSERVABLE_FORMULA] = blinded
        odf.loc[obs_id, OBSERVABLE_NAME] = str(spec["measures"])


#: Where an example's HELD-OUT measurements land, relative to its example dir. A sibling of
#: ``problem/``, deliberately NOT inside it: the runner copies ``problem/`` wholesale into ``work/``
#: and points the MCP server at it, so a file placed there is one directory listing away from the
#: agent. Nothing about the current server would expose it — but "the held-out data is not in the
#: tree the server is given" is a property worth having structurally rather than by audit.
VALIDATION_DIR = "validation"

#: The held-out measurement table inside :data:`VALIDATION_DIR`.
VALIDATION_MEASUREMENTS = "measurements.tsv"


def validation_measurements_path(example_dir: Path) -> Path:
    """Where :func:`materialize` writes this example's held-out measurements (may not exist)."""
    return Path(example_dir) / VALIDATION_DIR / VALIDATION_MEASUREMENTS


def _validation_measurement_df(*, name: str, validation_yaml: str, collection: Path | None,
                               published, training_measurement_df):
    """The held-out measurement table for ``name``, in the PUBLISHED vocabulary, fully checked.

    ``validation_yaml`` is a path relative to the benchmark's directory in the collection — fujita's
    ``test_data/...yaml``, the published problem for the EGF pulse + ramp protocols. That yaml points
    at the SAME model, condition, observable and parameter files and differs only in its measurement
    files, which is why a held-out score is comparable: the model is re-tested, not re-specified.
    CHECKED here rather than trusted.

    ``published`` is the PRISTINE training problem — before ``materialize`` relabels, merges or
    blinds it — because that is what the held-out yaml must match; ``training_measurement_df`` is
    the training table in the same vocabulary, for the disjointness check.

    Returned in the PUBLISHED vocabulary: ``rename_ids`` and the observable merge are applied by the
    CALLER (only ``materialize`` has the map, and the reference side must not apply them at all).
    Blinding is never applied — it only touches the observable table.

    Also the one place the FLATTENING hazard can be caught, because deciding it needs the
    UNFLATTENED problem, which is what ``published`` is on both call paths; by score time the
    reference's copy is already flattened and the verdict is unrecoverable.
    """
    import petab.v1 as petab

    bdir = benchmark_dir(name, collection)
    path = bdir / validation_yaml
    if not path.is_file():
        raise FileNotFoundError(
            f"{name}: validation_yaml {validation_yaml!r} does not exist under {bdir}")
    vprob = petab.Problem.from_yaml(str(path))

    # The held-out problem must differ from the training one ONLY in its measurements. Compared on
    # the tables that define the estimation problem; the visualization table is deliberately not
    # compared (it describes plots, and the collection's held-out variants do carry their own).
    for label, mine, theirs in (("observable", published.observable_df, vprob.observable_df),
                                ("condition", published.condition_df, vprob.condition_df),
                                ("parameter", published.parameter_df, vprob.parameter_df)):
        if theirs is None or not mine.equals(theirs):
            raise ValueError(
                f"{name}: the validation problem's {label} table differs from the published "
                f"training one. A held-out set must re-use the SAME problem (same model, same "
                f"observables, same conditions, same parameters) and change only the measurements "
                f"— otherwise the two scores are not comparable.")

    vmeas = vprob.measurement_df
    if vmeas is None or vmeas.empty:
        raise ValueError(f"{name}: validation_yaml {validation_yaml!r} lists no measurements")
    _check_held_out_rows(name, vprob, published, training_measurement_df)
    return vmeas


def _check_held_out_rows(name: str, vprob, published, training_measurement_df) -> None:
    """The checks every held-out set passes at SETUP, whichever kind it is: petab's own lint on both
    measurement tables, the flattening refusal, and row-level disjointness from the training data.
    """
    import pandas as pd
    from petab.v1.C import OBSERVABLE_ID, SIMULATION_CONDITION_ID, TIME
    from petab.v1.lint import check_measurement_df

    from petab_mcp.fitting import needs_flattening

    vmeas = vprob.measurement_df
    # petab's own measurement-table lint on BOTH operands (required columns, observable ids, override
    # counts, numeric measurements) before the disjointness check below indexes them.
    for label, frame, odf in (("validation", vmeas, vprob.observable_df),
                              ("published training", training_measurement_df,
                               published.observable_df)):
        try:
            check_measurement_df(frame, odf)
        except (AssertionError, ValueError) as exc:
            raise ValueError(
                f"{name}: the {label} measurement table fails petab's lint: {exc}") from exc

    # Flattening: refused at SETUP, where `published` is still unflattened (see the docstring).
    for label, prob_ in (("published training", published), ("validation", vprob)):
        if needs_flattening(prob_):
            raise ValueError(
                f"{name}: the {label} measurement table needs petab's timepoint-specific "
                f"flattening, which mints one observable per distinct override — so the training "
                f"and held-out tables would flatten to different observable sets and their "
                f"likelihoods could not be compared. Held-out scoring is not supported for this "
                f"problem; drop the example's held-out declaration.")

    # HELD OUT means held out: a row the model was fitted to cannot also test it. Checked at row
    # level rather than by condition, because a held-out set may legitimately withhold later
    # timepoints of a condition the training set also uses.
    # TIME is compared as a FLOAT, not stringified: the two tables are parsed from different TSVs,
    # so the same instant can arrive as 1 in one and 1.0 in the other, and `.astype(str)` would
    # make those different keys — letting a genuinely in-sample row past the check.
    def _keys(frame):
        ids = frame[[OBSERVABLE_ID, SIMULATION_CONDITION_ID]].astype(str)
        times = pd.to_numeric(frame[TIME], errors="coerce").astype(float)
        return set(zip(ids[OBSERVABLE_ID], ids[SIMULATION_CONDITION_ID], times))

    shared = _keys(vmeas) & _keys(training_measurement_df)
    if shared:
        raise ValueError(
            f"{name}: {len(shared)} held-out measurement row(s) are also in the TRAINING table "
            f"(e.g. {sorted(shared)[:3]}); the score would be partly in-sample")


def _transfer_problem(*, name: str, transfer_yaml: str, collection: Path | None, published,
                      training_measurement_df):
    """The collection's second problem on the same mechanism, in the published vocabulary.

    ``transfer_yaml`` is relative to the benchmark directory. Its own condition and parameter
    tables accompany the measurements, for :func:`materialize` and the reference refit.
    The SBML must match, the condition columns (the model's inputs; ``conditionName`` is free
    text) must agree, and every observable it defines must be one the training problem defines,
    the same way — a readout the training problem never had is refused rather than dropped, so
    what is scored is never a silent subset of what the paper measured. Then
    :func:`_check_held_out_rows` checks disjointness, lint and flattening.
    """
    import hashlib

    import petab.v1 as petab
    from petab.v1.C import CONDITION_NAME
    from petab.v1.yaml import load_yaml

    from petab_mcp.validation_data import _reject_untransferable_table

    bdir = benchmark_dir(name, collection)
    path = bdir / transfer_yaml
    if not path.is_file():
        raise FileNotFoundError(
            f"{name}: transfer_yaml {transfer_yaml!r} does not exist under {bdir}")
    vprob = petab.Problem.from_yaml(str(path))

    def _sbml_digest(yaml_path: Path) -> list[str]:
        cfg = load_yaml(str(yaml_path))
        return sorted(hashlib.sha1((yaml_path.parent / f).read_bytes()).hexdigest()
                      for prob_ in cfg["problems"] for f in prob_.get("sbml_files", []))

    if _sbml_digest(path) != _sbml_digest(bdir / f"{name}.yaml"):
        raise ValueError(
            f"{name}: the transfer problem's SBML differs from the training problem's. A transfer "
            f"set is the SAME mechanism on a different system; a variant that also changed the "
            f"model would be refit as a different structure from the one the candidate is compared "
            f"to.")
    mine = sorted(set(map(str, published.condition_df.columns)) - {CONDITION_NAME})
    theirs = sorted(set(map(str, vprob.condition_df.columns)) - {CONDITION_NAME})
    if mine != theirs:
        raise ValueError(
            f"{name}: the transfer problem's condition table has columns {theirs} where the "
            f"training one has {mine}; a condition column is a model input, so the two describe "
            f"different models.")

    # Validate against the complete published problem, before cropping unshared readouts. The
    # zero-shot shipped table omits kinetics, and a candidate may name its own rates differently.
    _reject_untransferable_table(vprob.parameter_df, published.parameter_df, vprob)

    unknown = sorted(set(map(str, vprob.observable_df.index))
                     - set(map(str, published.observable_df.index)))
    if unknown:
        raise ValueError(
            f"{name}: the transfer problem defines observable(s) {unknown} that the training "
            f"problem does not; a model built for the training problem has no readout to score "
            f"them on, and dropping them would score a subset of what was measured without saying "
            f"so. Ship a held-out problem that defines only the training problem's observables.")
    for oid in vprob.observable_df.index:
        mine = {k: str(v) for k, v in published.observable_df.loc[oid].items()}
        theirs = {k: str(v) for k, v in vprob.observable_df.loc[oid].items()}
        if mine != theirs:
            raise ValueError(
                f"{name}: observable {oid!r} is defined differently in the transfer problem "
                f"({theirs}) and the training problem ({mine}); the two would score different "
                f"quantities under one name.")
    vprob.visualization_df = None
    if vprob.measurement_df.empty:
        raise ValueError(f"{name}: transfer_yaml {transfer_yaml!r} lists no measurements")
    _check_held_out_rows(name, vprob, published, training_measurement_df)
    return vprob

def _held_out_source(*, name: str, validation_yaml: str | None, validation_table: str | None,
                     example_dir: Path, collection: Path | None, published,
                     transfer_yaml: str | None = None):
    """The held-out set (:class:`petab_mcp.validation_data.HeldOut`) from whichever source the
    example declares, in the PUBLISHED vocabulary, or ``None``.

    Every call site goes through here so the sources cannot disagree: there is one held-out set
    per example, so declaring two sources is a contradiction, not a merge. A set with a parameter
    table — a ``transfer_yaml``, or a curated table with a ``parameters.tsv`` beside it — is a
    TRANSFER set; the others are fixed-θ.
    """
    from petab_mcp.validation_data import HeldOut

    declared = {"VALIDATION_YAML": validation_yaml, "VALIDATION_TABLE": validation_table,
                "TRANSFER_VALIDATION_YAML": transfer_yaml}
    if sum(1 for v in declared.values() if v) > 1:
        raise ValueError(
            f"{name}: declares BOTH of {sorted(k for k, v in declared.items() if v)}; the held-out "
            f"sources are mutually exclusive — an example has one held-out set, so pick the source "
            f"it comes from")
    if validation_yaml:
        return HeldOut(measurement_df=_validation_measurement_df(
            name=name, validation_yaml=validation_yaml, collection=collection,
            published=published, training_measurement_df=published.measurement_df))
    if transfer_yaml:
        vprob = _transfer_problem(
            name=name, transfer_yaml=transfer_yaml, collection=collection, published=published,
            training_measurement_df=published.measurement_df)
        return HeldOut(measurement_df=vprob.measurement_df, condition_df=vprob.condition_df,
                       parameter_df=vprob.parameter_df)
    if validation_table:
        return _curated_held_out(
            name=name, validation_table=validation_table, example_dir=example_dir,
            published=published)
    return None


def _curated_held_out(*, name: str, validation_table: str, example_dir: Path, published):
    """The held-out set an example SHIPS, in the published vocabulary, fully checked.

    ``validation_table`` is a PEtab measurement TSV committed under the example directory, for
    held-out data the collection does not export (raimundez's Fig 5D long-term kinetics, curated
    from the paper's own archive). There is no second problem to compare against as there is for
    ``validation_yaml``: the table is written against the PUBLISHED problem, so ``published``'s
    observable table is the one it has to satisfy, and every check :func:`_check_held_out_rows`
    makes applies unchanged. Two optional files beside it say what kind of set it is
    (:class:`petab_mcp.validation_data.HeldOut`):

    * no ``parameters.tsv`` — FIXED-θ: every condition the table names must be one the model was
      told about — a published condition, or a row of a ``conditions.tsv`` giving NEW VALUES of
      the published inputs (raimundez's knockdown level), under the rules the scorer applies at
      score time (:func:`petab_mcp.validation_data._with_held_out_conditions`);
    * a ``parameters.tsv`` — TRANSFER: the same mechanism on a different system (lucarelli's other
      cell types), on which both structures are REFIT. The file holds only what differs from the
      published parameter table — new rows, or published rows with another value, bound or
      ``estimate`` — and may be header-only; the published table is the rest, pruned to what the
      held-out tables reference, and checked as a collection transfer problem is
      (:func:`petab_mcp.validation_data._reject_untransferable_table`). A ``conditions.tsv`` then
      REPLACES the training condition table, on the same columns, and a ``units.tsv``
      (``conditionId``, ``unitId``) names the systems the set is refit on, one at a time
      (:attr:`petab_mcp.validation_data.HeldOut.units`); without it the set is one system.
    """
    import copy

    import petab.v1 as petab
    from petab.v1.C import (CONDITION_NAME, OBSERVABLE_ID, PREEQUILIBRATION_CONDITION_ID,
                            SIMULATION_CONDITION_ID)
    from petab.v1.parameters import get_valid_parameters_for_parameter_table

    from petab_mcp.validation_data import (HELD_OUT_CONDITIONS, HELD_OUT_PARAMETERS,
                                           HELD_OUT_UNITS, HeldOut, _refit_units,
                                           _reject_untransferable_table, _with_held_out_conditions,
                                           read_units)

    path = Path(example_dir) / validation_table
    if not path.is_file():
        raise FileNotFoundError(
            f"{name}: validation_table {validation_table!r} does not exist under {example_dir}")
    vmeas = petab.get_measurement_df(str(path))
    if vmeas.empty:
        raise ValueError(f"{name}: validation_table {validation_table!r} lists no measurements")
    cpath = path.with_name(HELD_OUT_CONDITIONS)
    vcond = petab.get_condition_df(str(cpath)) if cpath.is_file() else None
    ppath = path.with_name(HELD_OUT_PARAMETERS)
    vpars = petab.get_parameter_df(str(ppath)) if ppath.is_file() else None
    upath = path.with_name(HELD_OUT_UNITS)
    units = read_units(upath) if upath.is_file() else None
    if units is not None and vpars is None:
        raise ValueError(f"{name}: {upath.relative_to(example_dir)} names refit units, but without "
                         f"a {HELD_OUT_PARAMETERS} beside it the set is scored at the fitted "
                         f"point, where nothing is refit")
    if units is not None and isinstance(_refit_units(vmeas, units), str):
        raise ValueError(f"{name}: {upath.relative_to(example_dir)}: {_refit_units(vmeas, units)}")

    vprob = copy.copy(published)
    vprob.measurement_df = vmeas
    vprob.visualization_df = None
    if vpars is None:
        rejected = _with_held_out_conditions(vprob, vcond)
        if rejected:
            raise ValueError(f"{name}: {cpath.relative_to(example_dir)}: {rejected}")
    else:
        system_differs = False
        if vcond is not None:
            mine = sorted(set(map(str, published.condition_df.columns)) - {CONDITION_NAME})
            theirs = sorted(set(map(str, vcond.columns)) - {CONDITION_NAME})
            if mine != theirs:
                raise ValueError(
                    f"{name}: {cpath.relative_to(example_dir)} has columns {theirs} against the "
                    f"published {mine}; a condition column is a model input, so the two tables "
                    f"describe different models")
            vprob.condition_df = vcond
            # A row with input values no published row has is another system; published rows
            # under new ids are not one by themselves.
            known = {tuple(map(str, r)) for r in published.condition_df[mine].itertuples(index=False)}
            system_differs = any(tuple(map(str, r)) not in known
                                 for r in vcond[mine].itertuples(index=False))
        # Only the observables the set measures: a sigma no held-out row informs is not a
        # parameter of the refit, and a delta row for it names nothing the set uses.
        measured = set(vmeas[OBSERVABLE_ID].astype(str))
        vprob.observable_df = published.observable_df.loc[
            [o for o in published.observable_df.index if str(o) in measured]]
        full = _overlay_parameters(published.parameter_df, vpars)
        try:
            # On the unpruned overlay: a published row the held-out tables retire must not read as
            # a difference, and the identity check sees every column (`system_differs` is the
            # other legitimate source of difference).
            _reject_untransferable_table(full, published.parameter_df, vprob,
                                         system_differs=system_differs)
        except ValueError as exc:
            raise ValueError(f"{name}: {ppath.relative_to(example_dir)}: {exc}") from exc
        allowed = get_valid_parameters_for_parameter_table(
            model=vprob.model, condition_df=vprob.condition_df, observable_df=vprob.observable_df,
            measurement_df=vmeas)
        unused = sorted(str(pid) for pid in vpars.index if pid not in allowed)
        if unused:
            raise ValueError(
                f"{name}: {ppath.relative_to(example_dir)}: row(s) {unused} name nothing the "
                f"held-out problem uses — not a parameter of the published model, not a "
                f"placeholder of the observables it measures or of its measurement table")
        vprob.parameter_df = full.loc[[pid for pid in full.index if pid in allowed]]
    # Lint before indexing condition ids, so a malformed curated table gets the same
    # actionable setup error as a table loaded through the yaml path.
    _check_held_out_rows(name, vprob, published, published.measurement_df)

    named = set(vmeas[SIMULATION_CONDITION_ID].astype(str))
    if PREEQUILIBRATION_CONDITION_ID in vmeas.columns:
        named |= set(vmeas[PREEQUILIBRATION_CONDITION_ID].dropna().astype(str))
    unknown = sorted(named - set(vprob.condition_df.index.astype(str)))
    if unknown:
        raise ValueError(
            f"{name}: the curated held-out table names condition(s) neither the published problem "
            f"nor a {HELD_OUT_CONDITIONS} beside it defines: {unknown}. Scoring at the fitted "
            f"point requires the model to have been told what the condition is: a new value of "
            f"the published inputs goes in that file; a new input makes this a TRANSFER set, not "
            f"a fixed-θ one")

    return HeldOut(measurement_df=vmeas, condition_df=vcond,
                   parameter_df=None if vpars is None else vprob.parameter_df, units=units)


def _overlay_parameters(published_df, delta_df):
    """The published parameter table with a curated set's own rows laid over it: a row the
    published table has takes the delta's filled cells, a new row is appended as given."""
    import pandas as pd
    from petab.v1.C import ESTIMATE

    full = published_df.copy()
    for pid, row in delta_df.iterrows():
        cells = row.dropna()
        if pid in full.index:
            for col, val in cells.items():
                full.loc[pid, col] = val
        else:
            full = pd.concat([full, pd.DataFrame([cells], index=pd.Index([pid],
                                                                          name=full.index.name))])
    if ESTIMATE in full.columns and not full[ESTIMATE].isna().any():
        full[ESTIMATE] = full[ESTIMATE].astype(float).astype(int)
    return full


def _held_out_problem(published, held_out):
    """The held-out PROBLEM a curated transfer set describes — the published model under the
    set's tables — for the reference refit (:func:`_transfer_reference`)."""
    import copy

    from petab.v1.C import OBSERVABLE_ID

    vprob = copy.copy(published)
    vprob.measurement_df = held_out.measurement_df
    if held_out.condition_df is not None:
        vprob.condition_df = held_out.condition_df
    measured = set(held_out.measurement_df[OBSERVABLE_ID].astype(str))
    vprob.observable_df = published.observable_df.loc[
        [o for o in published.observable_df.index if str(o) in measured]]
    vprob.parameter_df = held_out.parameter_df
    vprob.visualization_df = None
    return vprob


def materialize(*, name: str, shipped_param_ids: list[str], example_dir: Path,
                shipped_param_overrides: dict[str, dict] | None = None,
                rename_ids: dict[str, str] | None = None,
                preequilibration: dict[str, str] | None = None,
                observable_blinding: dict[str, dict] | None = None,
                validation_yaml: str | None = None,
                transfer_yaml: str | None = None,
                validation_table: str | None = None,
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
    built model would otherwise lack (see fiedler).

    ``validation_yaml`` names a published PEtab problem inside the benchmark's own directory whose
    measurements are HELD OUT — data from the same experiment under conditions the training set does
    not cover (fujita's ``test_data/`` EGF pulse + ramp protocols). Its measurement table is written
    to ``<example_dir>/validation/measurements.tsv``, OUTSIDE ``problem/`` so it never enters the
    tree the MCP server is pointed at, and the judge scores the finished model against it
    report-only (see :mod:`petab_mcp.validation_data`). Everything else about the example is
    unchanged: the agent is handed exactly the same problem, so runs with and without this are
    directly comparable and an example's ``Scenario.version`` does not move for it.

    ``validation_table`` is the same thing for a set the collection does not export: a measurement
    TSV the EXAMPLE ships (raimundez's ``validation_source/measurements.tsv``), read and checked
    here and written to the same place. Mutually exclusive with ``validation_yaml`` — one held-out
    file, one source of it. Fixed-θ, or a transfer set when a ``parameters.tsv`` sits beside the
    table (see :func:`_curated_held_out`).

    ``transfer_yaml`` is the other kind of held-out set (:func:`_transfer_problem`): the same
    mechanism on a different system, with its own condition and parameter tables, on which both
    the candidate and the reference are REFIT. Its measurements land in the same file; its condition
    and parameter tables are written beside them (``conditions.tsv``, ``parameters.tsv``), and the
    parameter table's presence is what tells the server to refit rather than evaluate
    (:class:`petab_mcp.validation_data.HeldOut`). One held-out set per example: the two are
    mutually exclusive."""
    from petab.v1.C import CONDITION_NAME

    problem_dir = Path(example_dir) / "problem"
    problem_dir.mkdir(parents=True, exist_ok=True)
    prob = _load_problem(name, collection, flatten=False)

    # BEFORE any rewriting: `prob` is still exactly as the collection ships it, which is what the
    # held-out yaml has to be checked against — so it serves as BOTH `published` and the training
    # table, and nothing is re-parsed. (Handing `_validation_measurement_df` a second
    # freshly-loaded copy cost another full SBML + four-TSV parse for tables identical to these.)
    # `prob` is mutated in place from here on.
    held_out = _held_out_source(
        name=name, validation_yaml=validation_yaml, validation_table=validation_table,
        transfer_yaml=transfer_yaml, example_dir=example_dir, collection=collection,
        published=prob)
    validation_meas = held_out.measurement_df if held_out is not None else None
    if preequilibration and held_out is not None and held_out.parameter_df is not None:
        raise ValueError(f"{name}: a transfer set is not supported together with PREEQUILIBRATION "
                         f"(the synthesized gate column would have to be added to the held-out "
                         f"condition table too)")

    def _relabel(v):
        if not isinstance(v, str):
            return v
        for old, new in (rename_ids or {}).items():
            v = v.replace(old, new)
        return v

    if rename_ids:
        condition_tables = [prob.condition_df]
        if held_out is not None and held_out.condition_df is not None:
            condition_tables.append(held_out.condition_df)
        for frame in condition_tables:
            frame.index = frame.index.map(_relabel)
            frame.columns = frame.columns.map(_relabel)
            for column in frame.columns:
                frame[column] = frame[column].map(_relabel)
        # Relabel author-bearing / export-artifact IDs (see rename_ids). Observable + measurement
        # tables here; the shipped parameter ids are relabelled after subsetting (below), since
        # shipped_param_ids references the ORIGINAL collection ids.
        prob.observable_df.index = prob.observable_df.index.map(_relabel)
        for _col in prob.observable_df.columns:
            prob.observable_df[_col] = prob.observable_df[_col].map(_relabel)
        for _col in prob.measurement_df.columns:
            prob.measurement_df[_col] = prob.measurement_df[_col].map(_relabel)
        if validation_meas is not None:
            # The held-out table speaks the SAME vocabulary as the shipped one or it references
            # observables/parameters that no longer exist. Same rewrite, same map.
            for _col in validation_meas.columns:
                validation_meas[_col] = validation_meas[_col].map(_relabel)

    merged = _merge_equivalent_observables(prob)
    if merged and validation_meas is not None:
        # Same fold, same table. `_merge_equivalent_observables` rewrites the TRAINING measurement
        # table's observableId in place; a held-out table left unfolded would reference the retired
        # per-dataset ids and fail the observable check in `validation_data._table_mismatch`.
        from petab.v1.C import OBSERVABLE_ID as _OID
        validation_meas[_OID] = validation_meas[_OID].map(lambda v: merged.get(str(v), v))
    if merged and observable_blinding:
        observable_blinding = _collapse_blinding(observable_blinding, merged)

    if observable_blinding:
        # After rename_ids: the blinding spec is written in the RELABELLED vocabulary — the ids
        # the agent will actually see.
        _blind_observables(prob, observable_blinding)

    if preequilibration:
        from petab.v1.C import PREEQUILIBRATION_CONDITION_ID
        # Every measurement preequilibrates at the synthesized basal condition (added to `cond` below).
        prob.measurement_df[PREEQUILIBRATION_CONDITION_ID] = preequilibration["preeq_condition_id"]
        if validation_meas is not None:
            # EVERY measurement means the held-out ones too. Left off, the training data would be
            # fitted from the basal steady state while the held-out data was predicted from the
            # model's declared t=0 — and a model built to RELAX to that steady state (which is the
            # whole point of this knob, see fiedler) predicts from the wrong initial state. The
            # two scores would then describe different simulations and `delta_nll` would be a
            # silent, arbitrarily wrong number. The synthesized condition is in the shipped
            # condition table either way, so nothing else has to change.
            validation_meas[PREEQUILIBRATION_CONDITION_ID] = \
                preequilibration["preeq_condition_id"]

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
    out = {"measurements": len(prob.measurement_df), "observables": len(prob.observable_df),
           "conditions": len(prob.condition_df)}
    print(f"[setup] {name}: materialized {len(prob.measurement_df)} measurements, "
          f"{len(prob.observable_df)} observables, {len(prob.condition_df)} condition(s); "
          f"shipped params {shipped_param_ids}")
    from petab_mcp.validation_data import HELD_OUT_CONDITIONS, HELD_OUT_PARAMETERS, HELD_OUT_UNITS

    vpath = validation_measurements_path(example_dir)
    cpath = vpath.with_name(HELD_OUT_CONDITIONS)
    # Files an earlier definition of the example left behind: the directory is gitignored, a
    # leftover parameters.tsv would flip the set's mode (validation_data.read_held_out) and a
    # leftover set would score an example that declares none.
    for stale in (vpath, cpath, vpath.with_name(HELD_OUT_PARAMETERS),
                  vpath.with_name(HELD_OUT_UNITS)):
        stale.unlink(missing_ok=True)
    if validation_meas is not None:
        from petab.v1.C import SIMULATION_CONDITION_ID

        vpath.parent.mkdir(parents=True, exist_ok=True)
        validation_meas.to_csv(vpath, sep="\t", index=False)
        out["validation_measurements"] = len(validation_meas)
        # The set's own condition table, in the shape of the conditions.tsv written above (no
        # conditionName, the preequilibration gate ON): rows the server appends for a fixed-θ set,
        # the table it swaps in for a transfer set.
        if held_out.condition_df is not None:
            hcond = held_out.condition_df.reset_index()
            hcond = hcond.drop(columns=[c for c in (CONDITION_NAME,) if c in hcond.columns])
            if preequilibration:
                hcond[preequilibration["gate_column"]] = 1
            hcond.to_csv(cpath, sep="\t", index=False)
            out["validation_conditions"] = len(hcond)
            print(f"[setup] {name}: held-out set brings {len(hcond)} condition row(s) of its "
                  f"own -> {cpath.name}")
        # Disjointness is enforced at ROW level (`_check_held_out_rows`), so a held-out condition
        # may legitimately also appear in training with other timepoints — hence "condition(s)",
        # with no claim that they are absent from the training data.
        print(f"[setup] {name}: held-out validation set -> {vpath.name}: "
              f"{len(validation_meas)} measurement(s) over "
              f"{validation_meas[SIMULATION_CONDITION_ID].nunique()} condition(s), none of them a "
              f"training row (from {validation_yaml or validation_table or transfer_yaml})")
        if held_out.parameter_df is not None:
            # The set's own parameter table, ids relabelled with the shipped ones; its presence is
            # what tells the server to refit (validation_data.HeldOut). Which of its rows the
            # candidate's refit takes is the scorer's rule (transfer_view), not a setup filter.
            hpars = held_out.parameter_df.reset_index()
            hpars["parameterId"] = hpars["parameterId"].map(_relabel)
            hpars.to_csv(vpath.with_name(HELD_OUT_PARAMETERS), sep="\t", index=False)
            print(f"[setup] {name}: transfer set: {len(hpars)}-row parameter table -> both sides "
                  f"are REFIT on it")
        if held_out.units:
            import pandas as pd
            pd.DataFrame({"conditionId": [_relabel(c) for c in held_out.units],
                          "unitId": list(held_out.units.values())}).to_csv(
                vpath.with_name(HELD_OUT_UNITS), sep="\t", index=False)
            print(f"[setup] {name}: refit in {len(set(held_out.units.values()))} unit(s) "
                  f"({HELD_OUT_UNITS})")
    return out


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


def charge_hardcoded_parameters(ref: dict[str, Any],
                                hardcoded: dict[str, str] | None) -> dict[str, Any]:
    """Price the published model's FITTED literals into the reference's ``k`` (see
    ``write_reference_for``): one parameter per literal, the literals recorded on the result as
    ``hardcoded_constants`` so a reference whose ``k`` exceeds its estimated-row count is
    auditable. Pure, so the charge is the same whichever fit stage won."""
    if not hardcoded:
        return ref
    return {**ref, "k": int(ref["k"]) + len(hardcoded),
            "hardcoded_constants": sorted(hardcoded)}


def write_reference_for(*, name: str, example_dir: Path, n_starts: int = 50,
                        n_workers: int | None = None,
                        per_start_timeout: float | None = None,
                        collection: Path | None = None,
                        free_parameters: list[str] | None = None,
                        param_overrides: dict[str, dict] | None = None,
                        hardcoded_parameters: dict[str, str] | None = None,
                        validation_yaml: str | None = None,
                        transfer_yaml: str | None = None,
                        validation_table: str | None = None,
                        ) -> dict[str, Any]:
    """Regenerate the committed ``reference.json`` (the reference bar): fit the PRISTINE published
    PEtab problem EXACTLY as it ships. Run only via ``setup.py --refit`` — slow, and its output is
    committed.

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

    ``hardcoded_parameters`` ``{literal: why}`` names numeric literals in the PUBLISHED model that
    are FITTED values, not knowledge — a degree of freedom the published fit spent and then wrote
    into the math. Each adds one to the reference's ``k`` and is recorded as ``hardcoded_constants``
    in reference.json, so the bar pays for them exactly as the server prices a candidate's literals
    (``petab_mcp.literals``, #180). The published fit is unchanged; only its count is. It is a
    per-example judgment, not a scan: a unit conversion (raimundez's ``0.161`` ng/mL -> nM) or a
    measured total (boehm's ``207.6``) is knowledge and is NOT charged — a candidate absorbs the one
    into its kinetics for free and is shipped the other as a fixed row. See fujita's ``68190``: the
    author's fitted ``pro_EGFR`` (BioModels BIOMD0000000262), inlined by the collection.

    ``validation_yaml`` additionally records the reference's HELD-OUT score under ``validation`` —
    the predictive bar. It is the SAME argument ``materialize`` takes, re-resolved here from the
    collection rather than read off the materialized copy, so the reference is never scored against
    a stale ``validation/measurements.tsv`` left behind by an earlier definition of the example.

    ``transfer_yaml`` records the reference structure's REFIT on the transfer set instead
    (:func:`_transfer_reference`) — the bar for a candidate's own refit there. Same ``mode`` in the
    block as the server's payload carries, so the judge can refuse to subtract across kinds.
    ``validation_table`` does the same for a curated held-out set, and for the same reason: the
    reference is scored against the table the example SHIPS, re-read here rather than off the
    materialized copy.

    """
    import json

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

    # The held-out table in the PUBLISHED vocabulary — this side fits the pristine collection
    # problem, whose observable ids are the original ones, so the relabel/merge that
    # `materialize` applies for the AGENT's copy must NOT be applied here. Loaded unflattened for
    # the same reason `materialize` compares unflattened tables: flattening is an AMICI detail
    # applied downstream, and the yaml's own tables are what the identity check is about.
    held_out = None
    if validation_yaml or validation_table or transfer_yaml:
        pristine = _load_problem(name, collection, flatten=False)
        held_out = _held_out_source(
            name=name, validation_yaml=validation_yaml, validation_table=validation_table,
            transfer_yaml=transfer_yaml, example_dir=example_dir, collection=collection,
            published=pristine)
    transfer = held_out is not None and held_out.parameter_df is not None
    # A fixed-θ set is scored at the vector the training refit lands on, inside it; a transfer set
    # is REFIT from the published point after it (`_transfer_reference`), so the refit gets none.
    fixed = None if transfer else held_out

    if not free_parameters:
        ref = write_reference_from_problem(
            problem, build_dir=build_dir / "pristine",
            model_name=f"{name}_pristine", path=problem_dir / "reference.json",
            n_starts=n_starts, n_workers=n_workers, per_start_timeout=per_start_timeout,
            held_out=fixed,
        )
    else:
        ref = _write_freed_reference(
            name=name, problem=problem, problem_dir=problem_dir, build_dir=build_dir,
            collection=collection, free_parameters=free_parameters,
            param_overrides=param_overrides, held_out=fixed,
            n_starts=n_starts, n_workers=n_workers, per_start_timeout=per_start_timeout)
    # After both arms, so a freed reference pays for the published literals too, and rewritten
    # because charging changes the `k` the arm above already wrote to reference.json — as does the
    # transfer block below.
    if hardcoded_parameters:
        ref = charge_hardcoded_parameters(ref, hardcoded_parameters)
    if transfer:
        # The collection's second problem is re-read as a problem (its observable set is its own);
        # a curated set is the published problem under the set's tables.
        vprob = (_transfer_problem(
                     name=name, transfer_yaml=transfer_yaml, collection=collection,
                     published=pristine, training_measurement_df=pristine.measurement_df)
                 if transfer_yaml else _held_out_problem(pristine, held_out))
        ref["validation"] = _transfer_reference(
            name=name, vprob=vprob, param_overrides=param_overrides, build_dir=build_dir,
            n_starts=n_starts, units=held_out.units)
    if hardcoded_parameters or transfer:
        (problem_dir / "reference.json").write_text(json.dumps(ref, indent=2) + "\n")
    return ref


def _transfer_reference(*, name: str, vprob, param_overrides: dict[str, dict] | None,
                        build_dir: Path, n_starts: int, units=None) -> dict[str, Any]:
    """Refit the reference through the candidate's solver and budget, from the published point.

    Apply the training reference's parameter overrides so both sides use the same noise model.
    Key the AMICI build by the estimate split, as :func:`_write_freed_reference` does.
    A failed refit is recorded under ``validation`` while preserving the training result.
    """
    import hashlib

    from petab.v1.C import ESTIMATE

    from petab_mcp import validation_data

    _apply_param_overrides(vprob, name, param_overrides)
    vprob.parameter_df = vprob.parameter_df.drop(
        columns=[c for c in ("objectivePriorType", "objectivePriorParameters")
                 if c in vprob.parameter_df.columns])
    estimated = sorted(str(p) for p in vprob.parameter_df.index
                       if int(float(vprob.parameter_df.loc[p, ESTIMATE])) == 1)
    split = hashlib.sha1("\0".join(estimated).encode()).hexdigest()[:8]
    print(f"[setup] {name}: refitting the reference structure on the transfer set "
          f"({len(vprob.measurement_df)} rows, {len(estimated)} estimated parameters) ...")
    try:
        out = validation_data.refit_problem(
            vprob, build_dir=build_dir / f"pristine_transfer_{split}",
            model_name=f"{name}_pristine_transfer", n_starts=n_starts, units=units)
    except Exception as exc:  # noqa: BLE001 - report-only; preserve the training result
        out = {"status": "error", "mode": validation_data.MODE_TRANSFER,
               "reason": f"{type(exc).__name__}: {exc}"}
        print(f"[setup] {name}: could not refit the reference on the transfer set "
              f"({type(exc).__name__}: {exc}); the nll/k/n bar is still written, but no transfer "
              f"bar — re-run --refit once the cause is fixed")
    else:
        print(f"[setup] {name}: transfer reference: status={out.get('status')} "
              f"nll={out.get('nll')} k={out.get('k')} n={out.get('n')}")
    return {**out, "mode": validation_data.MODE_TRANSFER}


def _write_freed_reference(*, name: str, problem, problem_dir: Path, build_dir: Path,
                           collection: Path | None, free_parameters: list[str],
                           param_overrides: dict[str, dict] | None, held_out,
                           n_starts: int, n_workers: int | None,
                           per_start_timeout: float | None) -> dict[str, Any]:
    """The ``REFERENCE_FREE_PARAMS`` arm of :func:`write_reference_for` (alkan, bachmann,
    raimundez): fit the published split and the freed one, keep the better, write the result."""
    import json

    from petab.v1.C import ESTIMATE

    from .simulate import write_reference_from_problem

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
        held_out=held_out,
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
        held_out=held_out,
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
