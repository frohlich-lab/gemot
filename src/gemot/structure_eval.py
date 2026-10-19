"""Structure-recovery eval: compare a model's interaction structure to the reference's,
projected onto the shared observables so it works across models with *different* species.

Motivation: BIC scores fit quality; it says nothing about whether the agent recovered the
right *coupling structure*. Candidate and reference are fit to the same observables but have
their own, non-corresponding internal species, so the raw state Jacobian isn't comparable.
The observables are shared and identically named — project onto them:

1. Build each model's symbolic ODE model with **AMICI** (``SbmlImporter._build_ode_model`` —
   symbolic only, no C++ compile), and read the *structural* nonzero patterns AMICI already
   derives: the state Jacobian ``∂ẋ/∂x`` — boolean-assembled from ``dxdotdx_explicit`` and
   ``dxdotdw · dwdx`` (so flux/assignment-rule dependencies are folded in correctly, without
   re-parsing SBML ourselves) — and ``dydx`` (which states each observable reads).
2. Reduce to observable→observable coupling: ``y_b`` couples into ``y_a`` iff some state read
   by ``y_b`` can REACH (transitively, in the interaction graph) some state read by ``y_a``.
   Transitive reachability marginalizes hidden species out entirely, so two models with the
   same observable-level topology match regardless of how many latent states each threads
   between observables — that is what makes the comparison meaningful across species sets.
3. Score with the Jaccard similarity of the two off-diagonal coupling edge sets.

This is a standalone, **post-hoc** analyzer: point it at a finished run's archived
``model.ant`` (or a live model string) + the problem's observables, and the example's true
structure, and it reports the Jaccard. It does not depend on the live MCP server, so it can be
shimmed onto already-completed sweeps. Building the symbolic model is real (sympy) CPU, so run
batches on a compute node.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from . import paths

# observableParameter1_x / noiseParameter1_x placeholders in observable/noise formulas are
# resolved from the measurement table at real PEtab-import time; for a standalone structural
# build they'd be undefined symbols. They never affect which STATES an observable reads, so we
# neutralize them to 1 before handing the formula to AMICI.
_PLACEHOLDER = re.compile(r"\b(?:observable|noise)Parameter\d+_\w+\b")


def _nonzero(mat, i: int, j: int) -> bool:
    """Structural nonzero test for a sympy matrix entry: an entry counts as present unless it
    is provably identically zero (``is_zero is True``). Unknown (``None``) counts as present."""
    return mat[i, j].is_zero is not True


def _build_demodel(sbml_str: str, observable_formulas: dict[str, str]):
    """AMICI symbolic ``DEModel`` for an SBML string + observable formulas (no compile)."""
    from amici.importers.sbml import SbmlImporter
    from amici.importers.utils import MeasurementChannel

    channels = [
        MeasurementChannel(id_=str(oid), formula=_PLACEHOLDER.sub("1", str(formula)), sigma="1.0")
        for oid, formula in observable_formulas.items()
    ]
    return SbmlImporter(sbml_str, from_file=False)._build_ode_model(observation_model=channels)


def _state_successors(de) -> dict[str, set[str]]:
    """Directed state influence graph ``{state_j: {states it influences}}`` — i.e. state ``j``
    influences ``ẋ_i`` (structurally ``∂ẋ_i/∂x_j ≠ 0``).

    The state Jacobian is ``∂ẋ/∂x = dxdotdx_explicit + dxdotdw · (I + dwdw + dwdw² + …) · dwdx``
    because a flux/expression ``w`` may depend on *other* ``w``'s (nested assignment rules /
    repeated expressions), so the state's influence can route through an arbitrarily long
    ``w → w → … → w → x`` chain. Rather than sum matrix powers, we build ONE directed
    dependency graph over nodes = states ∪ fluxes and take reachability — which captures the
    full ``dwdw`` recursion for free:

      * ``x_j → w_k``   iff ``dwdx[k, j] ≠ 0``           (w_k reads state x_j)
      * ``w_k' → w_k``  iff ``dwdw[k, k'] ≠ 0``          (w_k reads flux w_k')
      * ``w_k → x_i``   iff ``dxdotdw[i, k] ≠ 0``        (ẋ_i uses flux w_k)
      * ``x_j → x_i``   iff ``dxdotdx_explicit[i, j] ≠ 0`` (explicit, w-independent)

    ``x_j`` influences ``x_i`` iff node ``x_i`` is reachable from node ``x_j`` in that graph.
    """
    states = [str(s) for s in de.sym("x")]
    n = len(states)
    expl = de.eq("dxdotdx_explicit")  # (n, n)
    dxdotdw = de.eq("dxdotdw")  # (n, nw)
    dwdx = de.eq("dwdx")  # (nw, n)
    dwdw = de.eq("dwdw")  # (nw, nw)
    nw = dxdotdw.cols

    # nodes: ("x", i) and ("w", k). adjacency = "influences" (src -> dst).
    adj: dict[tuple, set[tuple]] = {}

    def add(src, dst):
        adj.setdefault(src, set()).add(dst)

    for i in range(n):
        for j in range(n):
            if _nonzero(expl, i, j):
                add(("x", j), ("x", i))
    for k in range(nw):
        for j in range(n):
            if _nonzero(dwdx, k, j):
                add(("x", j), ("w", k))
        for kp in range(nw):
            if _nonzero(dwdw, k, kp):
                add(("w", kp), ("w", k))
        for i in range(n):
            if _nonzero(dxdotdw, i, k):
                add(("w", k), ("x", i))

    succ: dict[str, set[str]] = {}
    for j in range(n):
        seen, stack = set(), [("x", j)]
        reached_states = set()
        while stack:
            node = stack.pop()
            for nxt in adj.get(node, ()):
                if nxt not in seen:
                    seen.add(nxt)
                    if nxt[0] == "x":
                        reached_states.add(states[nxt[1]])
                    stack.append(nxt)
        succ[states[j]] = reached_states
    return succ


def _observable_states(de) -> dict[str, set[str]]:
    """Each observable → the set of model states it reads, from AMICI's ``dydx``.

    The layout is resolved from the matrix SHAPE rather than assumed. Every other equation this
    module reads follows AMICI's ``∂a/∂b`` → rows-are-``a`` convention (``dxdotdw`` is
    ``nx × nw``, ``dwdx`` is ``nw × nx``), under which ``dydx`` would be ``ny × nx`` — but it was
    indexed here as ``nx × ny``. On a model with ``nx == ny`` the two are indistinguishable and a
    wrong guess silently transposes the couplings; otherwise it is an index error. Checking the
    shape is exact in both cases, and keeps working if AMICI's storage order changes.
    """
    states = [str(s) for s in de.sym("x")]
    obs = [str(o) for o in de.sym("y")]
    dydx = de.eq("dydx")
    shape = (dydx.rows, dydx.cols)
    if shape == (len(obs), len(states)):

        def reads(o: int, i: int) -> bool:
            return _nonzero(dydx, o, i)
    elif shape == (len(states), len(obs)):

        def reads(o: int, i: int) -> bool:
            return _nonzero(dydx, i, o)
    else:
        raise ValueError(
            f"dydx has shape {shape}, which matches neither (n_observables, n_states) "
            f"{(len(obs), len(states))} nor its transpose"
        )
    return {obs[o]: {states[i] for i in range(len(states)) if reads(o, i)} for o in range(len(obs))}


def observable_coupling(de) -> list[list[str]]:
    """Off-diagonal observable coupling edges (sorted ``[[a, b], ...]``): ``y_b`` structurally
    drives ``y_a`` (some state read by b reaches — or IS — a state read by a). Self-coupling
    excluded; observables reading no state contribute nothing."""
    succ = _state_successors(de)  # succ[j] = states reachable FROM j (transitive, excl. self)
    obs_states = _observable_states(de)
    # states y_b's dynamics can drive: its own read states plus everything they reach
    reach = {b: set(s).union(*(succ.get(x, set()) for x in s)) for b, s in obs_states.items() if s}
    out = []
    for a in sorted(obs_states):
        sa = obs_states[a]
        if not sa:
            continue
        for b in sorted(obs_states):
            if a != b and obs_states.get(b) and (reach[b] & sa):
                out.append([a, b])
    return sorted(out)


def coupling_from_sbml(sbml_str: str, observable_formulas: dict[str, str]) -> list[list[str]]:
    """Convenience: SBML string + {observable_id: formula} → coupling edge list."""
    return observable_coupling(_build_demodel(sbml_str, observable_formulas))


def coupling_jaccard(
    cand: list[list[str]] | None,
    ref: list[list[str]] | None,
    n_observables: int,
) -> float | None:
    """Jaccard of two coupling edge sets. ``None`` when < 2 observables (no off-diagonal to
    compare). Two empty sets over ≥ 2 observables AGREE (no cross-coupling) → 1.0."""
    if not n_observables or n_observables < 2:
        return None
    c = {tuple(e) for e in (cand or [])}
    r = {tuple(e) for e in (ref or [])}
    union = c | r
    return 1.0 if not union else len(c & r) / len(union)


# --------------------------------------------------------------------------- post-hoc driver
# Runs the eval over ALREADY-COMPLETED runs (their archived model.ant), so it can be shimmed
# onto sweeps that finished before this eval existed. Building the symbolic model is real CPU
# (sympy) — run a whole-archive batch on a compute node (srun/sbatch), not the login node.
_EXAMPLES_ROOT = paths.examples_root()


def observable_formulas(example: str) -> dict[str, str]:
    """{observable_id: formula} from an example's shipped ``problem/observables.tsv``."""
    import pandas as pd
    from petab.v2.C import OBSERVABLE_FORMULA, OBSERVABLE_ID

    df = pd.read_csv(_EXAMPLES_ROOT / example / "problem" / "observables.tsv", sep="\t")
    return {str(r[OBSERVABLE_ID]): str(r[OBSERVABLE_FORMULA]) for _, r in df.iterrows()}


def _true_antimony(example: str) -> str:
    """An example's withheld true model, as Antimony.

    Two conventions are in use and both are searched: the three synthetic demos define
    ``TRUTH_ANT`` in ``generate_data.py`` (they simulate their own data from it), while the
    thirteen benchmark examples define ``TRUE_ANT`` in ``run.py`` (their data is published, so
    they have no generate_data.py at all). Looking only for ``TRUE_ANT`` in ``generate_data.py``
    — the original behaviour — matched neither group.

    Importing the module offline is fine: this is an analysis tool, not the agent, so seeing the
    true structure is the point. Both file kinds guard their entry point behind ``__main__``.
    """
    import importlib.util

    for filename in ("generate_data.py", "run.py"):
        path = _EXAMPLES_ROOT / example / filename
        if not path.is_file():
            continue
        spec = importlib.util.spec_from_file_location(f"_truth_{example}_{path.stem}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for attr in ("TRUTH_ANT", "TRUE_ANT"):
            if ant := getattr(mod, attr, None):
                return ant
    raise ValueError(f"{example}: no TRUTH_ANT/TRUE_ANT in generate_data.py or run.py")


def reference_coupling(example: str) -> tuple[list[list[str]], int]:
    """The (cached) reference observable-coupling for an example, from its withheld true model.
    Returns ``(edges, n_observables)``."""
    from petab.v2.models.sbml_model import SbmlModel

    obs = observable_formulas(example)
    sbml = SbmlModel.from_antimony(_true_antimony(example), model_id=example).to_sbml_str()
    return coupling_from_sbml(sbml, obs), len(obs)


def evaluate_model_ant(
    example: str, model_ant: str, ref: tuple[list[list[str]], int] | None = None
) -> dict[str, Any]:
    """Score a candidate ``model.ant`` for ``example`` against the reference structure. Returns
    ``{jaccard, n_observables, candidate_coupling, reference_coupling}`` (jaccard ``None`` when
    < 2 observables). ``ref`` may be passed pre-computed to avoid recomputing it per run."""
    from petab.v2.models.sbml_model import SbmlModel

    ref_edges, n_obs = ref if ref is not None else reference_coupling(example)
    obs = observable_formulas(example)
    sbml = SbmlModel.from_antimony(model_ant, model_id=example).to_sbml_str()
    cand = coupling_from_sbml(sbml, obs)
    return {
        "jaccard": coupling_jaccard(cand, ref_edges, n_obs),
        "n_observables": n_obs,
        "candidate_coupling": cand,
        "reference_coupling": ref_edges,
    }


def _final_model_ant(run_dir: Path) -> str | None:
    """The HEAD (final) version's ``model.ant`` in an archived run, or None if unavailable."""
    try:
        result = json.loads((run_dir / "result.json").read_text())
    except Exception:
        return None
    head = result.get("head_version")
    vdirs = []
    vroot = run_dir / ".petab_mcp" / "versions"
    if head is not None and (vroot / f"v{head}" / "model.ant").is_file():
        vdirs = [vroot / f"v{head}"]
    elif vroot.is_dir():  # fall back to the highest-numbered version that has a model
        vdirs = sorted(
            (d for d in vroot.glob("v*") if (d / "model.ant").is_file() and d.name[1:].isdigit()),
            key=lambda d: int(d.name[1:]),
            reverse=True,
        )
    for vd in vdirs:
        try:
            return (vd / "model.ant").read_text()
        except OSError:  # unreadable/missing despite the is_file() prefilter — try the next
            continue
    return None


def batch(
    archive_root: str | Path = _EXAMPLES_ROOT / "_runs_archive", examples: set[str] | None = None
) -> list[dict[str, Any]]:
    """Evaluate every archived run (optionally filtered to ``examples``). Reference coupling is
    computed once per example and reused. Returns a list of per-run result dicts."""
    archive_root = Path(archive_root)
    ref_cache: dict[str, tuple[list[list[str]], int]] = {}
    rows: list[dict[str, Any]] = []
    for run_dir in sorted(archive_root.glob("*/")):
        rj = run_dir / "result.json"
        if not rj.is_file():
            continue
        try:
            result = json.loads(rj.read_text())
        except Exception as exc:
            rows.append(
                {"run": run_dir.name, "example": "", "error": f"{type(exc).__name__}: {exc}"}
            )
            continue
        ex = result.get("scenario")
        if not ex or (examples and ex not in examples):
            continue
        # A reference-free example has no true structure to recover, so nothing to score against:
        # skipped rather than reported as an ERROR row per run.
        if result.get("judge_mode") == "reference_free":
            continue
        model_ant = _final_model_ant(run_dir)
        if model_ant is None:
            continue
        try:
            if ex not in ref_cache:
                ref_cache[ex] = reference_coupling(ex)
            ev = evaluate_model_ant(ex, model_ant, ref=ref_cache[ex])
        except Exception as exc:  # noqa: BLE001 - one bad run must not abort the batch
            rows.append(
                {"run": run_dir.name, "example": ex, "error": f"{type(exc).__name__}: {exc}"}
            )
            continue
        rows.append(
            {
                "run": run_dir.name,
                "example": ex,
                "success": result.get("success"),
                "delta_bic": result.get("delta_bic"),
                "jaccard": ev["jaccard"],
                "n_observables": ev["n_observables"],
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        description="Post-hoc structure-recovery (Jacobian-sparsity "
        "observable-coupling Jaccard) over archived runs."
    )
    p.add_argument("--archive", default=str(_EXAMPLES_ROOT / "_runs_archive"))
    p.add_argument("--examples", nargs="*", default=None, help="restrict to these examples")
    args = p.parse_args(argv)
    rows = batch(args.archive, set(args.examples) if args.examples else None)
    print(f"{'run':<34}{'example':<11}{'ok':>5}{'dBIC':>9}{'n_obs':>6}{'jaccard':>9}")
    for r in rows:
        if "error" in r:
            print(f"{r['run']:<34}{r['example']:<11}  ERROR: {r['error'][:50]}")
            continue
        db = f"{r['delta_bic']:+.1f}" if isinstance(r.get("delta_bic"), (int, float)) else "—"
        jac = f"{r['jaccard']:.2f}" if r.get("jaccard") is not None else "N/A"
        print(
            f"{r['run']:<34}{r['example']:<11}{str(r.get('success')):>5}{db:>9}"
            f"{r.get('n_observables', 0):>6}{jac:>9}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
