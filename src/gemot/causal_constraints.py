"""The causal_constraints arm's record: what the constraints cost the agent, and what the judged
model does with them.

The server checks every change against the problem's causal constraints and refuses one whose model
breaks a claim (frohlich-lab/PEtab-MCP#134, ``petab_mcp.constraints``), so every committed version
makes them all, the judged one included. Two things are left to measure, post hoc:

* **The cost.** The refusals per constraint (tallied by the runner's tool clock,
  ``runner.clock._ToolClock.by_constraint``), the first version that makes them all, and how many
  ``make_changes`` calls it took to get there.
* **The behaviour at the fit** (decision 9 on #134). An indirect constraint holds wherever the
  server's trace moves its form as claimed (``petab_mcp.tracing``), whatever else the trace also
  finds; which way the model moves its object is the fit's to say, not its structure's, once it has
  feedback, and isensee's forskolin claim turns with an estimated factor. So on the judged model, at its fitted
  values, each signed indirect constraint's subject is perturbed (its starting amount raised by
  half, or seeded where it starts at 0, as the #134 prototype measured the reference signs) and the
  object's amount compared with the gain norms of Fröhlich et al., MSB 2023
  (doi:10.15252/msb.202210988): the ratio of L∞ norms, and of L1 norms over log-time, perturbed
  over baseline, so that 1 is no effect. Each reads as the sign the constraint claims, the
  opposite, or none.

Which species ARE a constraint's subject and object is the server's to say, not this module's: the
judged version is built again from its statements record with the two added as ``amounts``
(``petab_mcp.operations.statements_to_antimony``), so each is read exactly as a readout reads it,
and its species are found in the stored model by their PySB pattern, which both models carry as
the species' name. The stored model is what is simulated, through :mod:`gemot.simulate`, over each
of the problem's experiments that PEtab v1 can state: one period from time 0, or a pre-equilibration
at ``-inf`` followed by one (#74: perica_egf's every experiment is pre-equilibrated, so the check
had read none of them).

In a pre-equilibrated experiment the subject is perturbed across both periods, deliberately not in
the main one alone. Its starting amount is where the pre-equilibration starts, so the cell rests in
the perturbed state and is stimulated from there, as an over-expression would be; raising a species
at time 0 instead is a pulse no experiment gives, and one PEtab v1 cannot state, since a condition
sets a species to a value and not to a multiple of the steady state it reached. Both periods are
reported: the gain norms over the main period, and the gain where the pre-equilibration left the
object, beside the subject's own there (near 1 where the model's turnover returned it to its steady
state, which no perturbation of a start can move).

Report-only: nothing here gates a run, and a constraint whose behaviour cannot be read records why
instead. Import-light at module scope, like the analysis it reads the store with.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

#: The server's refusal of a model that breaks a constraint (``petab_mcp.constraints.enforce``):
#: a refusal like any other in ``edit_rejects.by_cause``, never an infrastructure failure, and
#: tallied per constraint as well (:func:`broken_constraints`).
CONSTRAINT_VIOLATED = "constraint_violated"

#: Each constraint such a refusal names, as ``constraints.enforce`` writes it: "`<id>` (<text>):
#: <why>", the first after "constraints: " and the rest after "; ". The id is the constraint
#: reader's (``constraints._ID``).
_BROKEN = re.compile(r"(?:constraints: |; )`([A-Za-z_][A-Za-z0-9_]*)` \(")

#: How far the subject's starting amount is raised, and the seed where it starts at 0, as a
#: fraction of the model's largest fitted starting amount (at least 1): the #134 prototype's
#: perturbation (``behaviour.py``), which measured the reference signs the structural check was
#: scored against.
RAISE = 1.5
SEED_FRACTION = 0.1

#: A gain within this of 1 is no effect: 100x the integrator's relative tolerance, and the
#: prototype's threshold.
NO_EFFECT = 1e-6

#: The two norms, as ``result.json`` keys them.
NORMS = ("linf", "l1_logt")

#: Why a constraint's behaviour is not read on a problem none of whose experiments PEtab v1 can
#: state: a fact about the problem, which a sweep must tell apart from a check that crashed.
NO_EXPERIMENT = "no supported experiment"

#: Each experiment is simulated at time 0 and at points log-spaced over its measured span's last
#: three decades, so the L∞ norm sees the start and the L1 norm over log-time the whole span.
_GRID_POINTS = 101
_GRID_DECADES = 3

#: Names this module adds to a model. An agent's cannot clash: no amount or parameter it may write
#: starts with `__`.
_PREFIX = "__gemot_"


# ------------------------------------------------------------------------------------- the cost
def broken_constraints(error: str) -> list[str]:
    """The ids of the constraints a ``constraint_violated`` refusal names, each once."""
    return list(dict.fromkeys(_BROKEN.findall(error)))


def first_satisfying_version(constraints: Any, run_dir: str | Path) -> int | None:
    """The first stored version that makes every constraint, or ``None`` when none does.

    A version built from statements keeps the verdicts it was committed under
    (``petab_mcp.constraints.kept``), and on the arm every commit is checked under the run's
    constraints, so those say; re-checking would trace each indirect constraint on a rebuilt
    model, which the server already did. One not built from statements claims nothing, so it makes
    only the negative constraints, which ``constraints.check`` reads with no model."""
    from petab_mcp import constraints as constraints_mod

    from .analysis import iter_versions, read_version_json, store_dir

    for meta in iter_versions(run_dir):
        version = int(meta["version"])
        record = read_version_json(
            store_dir(run_dir) / "versions" / f"v{version}", "statements.json"
        )
        if record:
            verdicts = constraints_mod.kept(constraints, record)
            if len(verdicts) < len(constraints.items):
                continue  # checked under another set: it says nothing of this one
        else:
            verdicts = constraints_mod.check(constraints, None)
        if all(v.ok for v in verdicts.values()):
            return version
    return None


# -------------------------------------------------------------------------------- the behaviour
def claim(constraint: Any) -> dict | str:
    """What a constraint's behaviour is read on, ``{claim, subject, objects}``, or why it is not.

    Only a signed indirect constraint is read: a direct one is a row's claim, which the server
    checked exactly, and neither ``not`` nor ``does not affect`` claims a sign. The reading is the
    server's (``petab_mcp.constraints.Constraint``): the subject, and each form it must move with
    the way it must move it (for ``activates``, a declared inactive form falling where an active
    one rises), each agent as a pattern in the notation."""
    from petab_mcp import notation

    if constraint.direct:
        return "a direct claim, which the server checks exactly"
    if constraint.negated or any(form.way is None for form in constraint.forms):
        return "claims no sign"

    def pattern(agent) -> str:
        return notation.agent_text(agent.to_json())

    return {
        "claim": constraint.claim,
        "subject": pattern(constraint.subject),
        "objects": [(pattern(form.agent), form.way) for form in constraint.forms],
    }


def gains(baseline: list, perturbed: list) -> dict[str, float]:
    """MSB 2023's gain norms of a perturbation: ``linf``, the ratio of the object's L∞ norms, and
    ``l1_logt``, of its L1 norms over log-time, perturbed over baseline, so 1 is no effect.

    Each argument is a list of ``(times, amounts)``, one per experiment, pooled: the largest amount
    over all of them, and the integrals summed. A baseline norm of 0 gives ``inf`` where the
    perturbed one is not."""
    import numpy as np

    def norms(trajectories) -> tuple[float, float]:
        top = l1 = 0.0
        for times, values in trajectories:
            t, y = np.asarray(times, float), np.abs(np.asarray(values, float))
            top = max(top, float(y.max(initial=0.0)))
            later = t > 0
            if later.sum() > 1:
                l1 += float(np.trapezoid(y[later], np.log(t[later])))
        return top, l1

    (b_inf, b_l1), (p_inf, p_l1) = norms(baseline), norms(perturbed)
    return {"linf": _ratio(p_inf, b_inf), "l1_logt": _ratio(p_l1, b_l1)}


def _ratio(perturbed: float, baseline: float) -> float:
    return perturbed / baseline if baseline > 0 else (math.inf if perturbed > 0 else 1.0)


def verdict(gain: float) -> str:
    """``as_claimed``, ``opposite`` or ``none``: whether a gain says the subject raised what the
    constraint claims it raises (its forms, each counted the way it must move), lowered it, or
    left it."""
    if abs(gain - 1.0) <= NO_EFFECT:
        return "none"
    return "as_claimed" if gain > 1 else "opposite"


def behaviour(
    run_dir: str | Path,
    version: int,
    constraints: Any,
    *,
    entity_map: str | Path | None,
    build_dir: str | Path,
) -> dict[str, dict]:
    """Each constraint's behaviour on stored ``version`` of the run whose working copy is
    ``run_dir``: for a signed indirect one, its two gains (:func:`gains`), each with its
    :func:`verdict`, how its subject was perturbed and on which experiments, and on a
    pre-equilibrated problem its ``pre_equilibration`` gain as well; for any other, or one that
    could not be read, ``{"skipped": why}`` (:data:`NO_EXPERIMENT` where no experiment can be
    simulated, with the ones skipped).

    ``entity_map`` is the one the server was started with, which the rebuild reads the rows'
    agents against; ``build_dir`` holds the compiled model."""
    claims = {c.id: claim(c) for c in constraints.items}
    out: dict[str, dict] = {cid: {"skipped": c} for cid, c in claims.items() if isinstance(c, str)}
    read = {cid: c for cid, c in claims.items() if not isinstance(c, str)}
    if read:
        try:
            judged = _Judged(Path(run_dir), version, entity_map)
            plan, unread = judged.plan(read)
            out |= {cid: {"skipped": why} for cid, why in unread.items()}
            if plan:
                out |= judged.simulate(plan, Path(build_dir))
        except Exception as exc:  # noqa: BLE001 - report-only: a metric must never fail the judge
            why = f"{type(exc).__name__}: {exc}"
            out |= {cid: {"skipped": why} for cid in read if cid not in out}
    return dict(sorted(out.items()))


def _seeds_named(record: dict | None) -> dict | None:
    """``record`` with the ``starts`` of one stored before PEtab-MCP#184 read as its ``seeds``.
    The server takes no alias, so an archived run's version would otherwise rebuild with none of
    the seeds it declared."""
    if not record or "seeds" in record or "starts" not in record:
        return record
    return {**{k: v for k, v in record.items() if k != "starts"}, "seeds": record["starts"]}


class _Judged:
    """The judged version: its stored model, its fitted values, and the problem it was fitted to."""

    def __init__(self, run_dir: Path, version: int, entity_map: str | Path | None) -> None:
        from petab.v1.models.sbml_model import SbmlModel
        from petab_mcp.problem import WorkingProblem
        from petab_mcp.provenance import statements_record

        from .analysis import read_version_json, store_dir

        self.run_dir, self.version, self.entity_map = run_dir, version, entity_map
        self.vdir = store_dir(run_dir) / "versions" / f"v{version}"
        # As the server reads it now: a record stored before rows and their bindings were given
        # apart (PEtab-MCP#135) reads as the same rows and bindings, and one stored before the
        # species at t = 0 were named seeds (PEtab-MCP#184) reads its `starts` as its seeds.
        self.record = _seeds_named(
            statements_record(read_version_json(self.vdir, "statements.json"))
        )
        if not self.record:
            raise ValueError(f"version {version} was not built from statements")
        self.sbml = SbmlModel.from_antimony(
            (self.vdir / "model.ant").read_text(encoding="utf-8")
        ).sbml_model
        self.problem = WorkingProblem.from_yaml(run_dir / "problem.yaml")

    # -------------------------------------------------------------------------------- reading
    def rebuilt(self, amounts: dict[str, dict]):
        """The version's model built again from its record with ``amounts`` added, as SBML: the
        server's ``statement_edits`` path (``set_amount``), on a stand-in for the session that
        built it."""
        from types import SimpleNamespace

        from petab.v1.models.sbml_model import SbmlModel
        from petab_mcp import operations, protocols, seeding

        rows, bindings, seeds, readouts, given, _ = operations.apply_statement_edits(
            self.record,
            [{"op": "set_amount", "name": name, "agent": agent} for name, agent in amounts.items()],
        )
        # The run's written mechanisms, which the session held (`PetabSession.mechanisms`): each
        # row built from one stores its definition, and each call the ones it wrote.
        written = [row["mechanism"] for row in self.record["rows"] if row.get("mechanism")]
        written += list(self.record.get("mechanisms") or [])
        session = SimpleNamespace(
            mechanisms={(d["statement_type"], d["policy"]): d for d in written},
            entities=seeding.load_entity_map(str(self.entity_map)) if self.entity_map else {},
            inputs=protocols.load(self.run_dir, self.problem),
            problem=self.problem,
            shipped_parameter_df=self.problem.parameter_df.copy(deep=True),
        )
        antimony, *_ = operations.statements_to_antimony(
            session, rows, bindings, None, seeds, readouts, given
        )
        return SbmlModel.from_antimony(antimony).sbml_model

    def plan(self, read: dict[str, dict]) -> tuple[dict[str, dict], dict[str, str]]:
        """``(plan, unread)``: for each constraint read, its object as a formula over the stored
        model's species, the species holding its subject and those a seed goes to; for each that
        could not be, why. One rebuild serves them all, unless it fails: then each is rebuilt on
        its own, so a failure is charged to its constraint alone."""

        def amounts(i: int, c: dict) -> dict[str, dict]:
            return {f"{_PREFIX}s{i}": c["subject"]} | {
                f"{_PREFIX}o{i}_{j}": agent for j, (agent, _) in enumerate(c["objects"])
            }

        import sympy

        indexed = list(enumerate(read.items()))
        try:
            shared = self.rebuilt({k: v for i, (_, c) in indexed for k, v in amounts(i, c).items()})
        except Exception:  # noqa: BLE001 - attributed per constraint below
            shared = None
        stored = {sp.getName(): sp.getId() for sp in self.sbml.getListOfSpecies()}
        plan, unread = {}, {}
        for i, (cid, c) in indexed:
            try:
                rebuilt = shared if shared is not None else self.rebuilt(amounts(i, c))
                subject = _amount(rebuilt, f"{_PREFIX}s{i}")
                obj = sum(
                    way * _amount(rebuilt, f"{_PREFIX}o{i}_{j}")
                    for j, (_, way) in enumerate(c["objects"])
                )
                renamed, names = {}, {}
                for sym in subject.free_symbols | obj.free_symbols:
                    species = rebuilt.getSpecies(str(sym))
                    if species is None:
                        if self.sbml.getParameter(str(sym)) is None:
                            raise ValueError(f"the stored model has no parameter {sym}")
                        continue
                    if species.getName() not in stored:
                        raise ValueError(f"the stored model has no species {species.getName()!r}")
                    names[str(sym)] = species.getName()
                    renamed[sym] = sympy.Symbol(stored[species.getName()])
            except Exception as exc:  # noqa: BLE001 - this constraint's, recorded as why
                unread[cid] = f"{type(exc).__name__}: {exc}"
                continue
            held = [str(s) for s in subject.free_symbols if str(s) in names]
            # A seed goes to the subject's fewest-molecule form: free, in the state it is written.
            # A species' name is its PySB pattern, one `Monomer(...)` per molecule.
            size = {sid: names[sid].count("(") for sid in held}
            plan[cid] = {
                "claim": c["claim"],
                # What the subject must raise: each form, counted the way it must move it.
                "object": str(obj.xreplace(renamed)),
                "subject": {stored[names[sid]]: names[sid] for sid in held},
                "seeded": sorted(
                    stored[names[sid]] for sid in held if size[sid] == min(size.values())
                ),
            }
        return plan, unread

    # ----------------------------------------------------------------------------- simulating
    def simulate(self, plan: dict[str, dict], build_dir: Path) -> dict[str, dict]:
        """Simulate the baseline and each constraint's perturbations, in one compile."""
        import libsbml
        import numpy as np
        import pandas as pd
        from petab.v1.C import (
            OBSERVABLE_ID,
            PREEQUILIBRATION_CONDITION_ID,
            SIMULATION,
            SIMULATION_CONDITION_ID,
            TIME,
        )
        from petab.v1.models.sbml_model import SbmlModel

        from .simulate import measurement_skeleton, simulate_true

        experiments, skipped = self.experiments()
        if not experiments:
            return {cid: {"skipped": NO_EXPERIMENT, "experiments_skipped": skipped} for cid in plan}
        values = _fitted(self.vdir, self.sbml)
        seed = SEED_FRACTION * max(
            [1.0, *(v for k, v in values.items() if k.endswith("_init") and math.isfinite(v))]
        )
        document = libsbml.readSBMLFromString(
            libsbml.writeSBMLToString(self.sbml.getSBMLDocument())
        )
        sbml = document.getModel()
        # What the conditions set that the model holds: a target the agent's model never reads
        # has no effect, and a PEtab v1 condition may name only the model's own ids.
        known = {
            x.getId()
            for x in (
                *sbml.getListOfParameters(),
                *sbml.getListOfSpecies(),
                *sbml.getListOfCompartments(),
            )
        }
        for eid, (pre, main) in experiments.items():
            main = {t: v for t, v in main.items() if t in known}
            if pre is not None:
                pre = {t: v for t, v in pre.items() if t in known}
                # A v1 condition states every target, and v2's changes persist, so the main period
                # restates what the pre-equilibration set; but not a species, which v1 would then
                # reset rather than carry on from the steady state.
                main = {t: v for t, v in pre.items() if sbml.getSpecies(t) is None} | main
            experiments[eid] = (pre, main)
        set_by_conditions = {
            t for periods in experiments.values() for p in periods for t in p or {}
        }
        observables, rule_held = {}, {}
        for i, (cid, p) in enumerate(plan.items()):
            scale, offset = f"{_PREFIX}scale{i}", f"{_PREFIX}seed{i}"
            for name, value in ((scale, 1.0), (offset, 0.0)):
                par = sbml.createParameter()
                par.setId(name)
                par.setValue(value)
                par.setConstant(True)
            share = 1.0 / len(p["seeded"])
            for sid in p["subject"]:
                _perturbable(
                    sbml,
                    sid,
                    scale,
                    f"{share!r} * {offset}" if sid in p["seeded"] else None,
                    set_by_conditions=sid in set_by_conditions,
                )
            rule_held[cid] = any(sbml.getAssignmentRuleByVariable(s) for s in p["subject"])
            observables[f"{_PREFIX}y{i}"] = p["object"]
            observables[f"{_PREFIX}x{i}"] = " + ".join(p["subject"])

        def perturbed(targets: dict, kind: str, i: int, p: dict) -> dict:
            # A period's targets with constraint i's subject raised or seeded: its parameter set,
            # and each species of it that the period sets moved too.
            held = [s for s in p["subject"] if s in targets]
            if kind == "raise":
                return {
                    **targets,
                    f"{_PREFIX}scale{i}": RAISE,
                    **{s: RAISE * targets[s] for s in held},
                }
            share = 1.0 / len(p["seeded"])
            return {
                **targets,
                f"{_PREFIX}seed{i}": seed,
                **{s: targets[s] or share * seed for s in held if s in p["seeded"]},
            }

        conditions, times, preequilibration = {}, {}, {}
        for eid, (pre, main) in experiments.items():
            grid = _grid(self.span(eid))
            variants = {"base": (pre, main)}
            for i, p in enumerate(plan.values()):
                for kind in ("raise", "seed"):
                    variants[f"{kind}{i}"] = tuple(
                        None if period is None else perturbed(period, kind, i, p)
                        for period in (pre, main)
                    )
            for variant, (first, last) in variants.items():
                name = f"{eid}{_PREFIX}{variant}"
                conditions[name], times[name] = last, grid
                if first is not None:
                    preequilibration[name] = f"{name}_pre"
                    conditions[f"{name}_pre"] = first
        columns = sorted({k for c in conditions.values() for k in c})

        def own(target: str) -> float:
            # What a condition that does not set `target` leaves it at: a parameter's fitted value
            # or the model's, a compartment's size, a species its own start (NaN, in PEtab v1).
            if (par := sbml.getParameter(target)) is not None:
                return values.get(target, par.getValue())
            if (comp := sbml.getCompartment(target)) is not None:
                return comp.getSize()
            return np.nan

        condition_df = pd.DataFrame(
            [
                {"conditionId": c, **{k: v[k] if k in v else own(k) for k in columns}}
                for c, v in conditions.items()
            ]
        )
        observable_df = pd.DataFrame(
            [
                {"observableId": o, "observableFormula": f, "noiseFormula": "1"}
                for o, f in observables.items()
            ]
        )
        parameter_df = pd.DataFrame(
            [
                {
                    "parameterId": k,
                    "parameterScale": "lin",
                    "lowerBound": min(v, 0.0),
                    "upperBound": 10 * max(v, 1.0),
                    "nominalValue": v,
                    "estimate": 1,
                }
                for k, v in values.items()
                if k not in columns
            ]
        )
        skeleton = measurement_skeleton(times, list(observables))
        if preequilibration:
            # "" on a row that is not, which petab's lint reads as none; a NaN it counts as a
            # time-point-specific mapping. (AMICI's v1 import still fails on that mix where a
            # condition sets a species: no problem here mixes them.)
            skeleton[PREEQUILIBRATION_CONDITION_ID] = (
                skeleton[SIMULATION_CONDITION_ID].map(preequilibration).fillna("")
            )
        sim = simulate_true(
            SbmlModel(sbml_model=sbml).to_antimony(),
            parameter_df,
            condition_df,
            observable_df,
            skeleton,
            build_dir=build_dir / f"v{self.version}",
            model_name=f"gemot_constraints_v{self.version}",
        )

        def trajectory(condition: str, observable: str) -> tuple[list, list]:
            rows = sim[
                (sim[SIMULATION_CONDITION_ID] == condition) & (sim[OBSERVABLE_ID] == observable)
            ].sort_values(TIME)
            return rows[TIME].tolist(), rows[SIMULATION].tolist()

        def at_start(condition: str, observable: str) -> tuple[list, list]:
            times, amounts = trajectory(condition, observable)
            return times[:1], amounts[:1]

        out = {}
        for i, (cid, p) in enumerate(plan.items()):
            x, y = f"{_PREFIX}x{i}", f"{_PREFIX}y{i}"
            base_y, pert_y, rested, how = [], [], [], set()
            for eid, (pre, _) in experiments.items():
                base = f"{eid}{_PREFIX}base"
                # Raised, unless that leaves the subject as it was at time 0: one starting at 0.
                raised = gains([at_start(base, x)], [at_start(f"{eid}{_PREFIX}raise{i}", x)])
                variant = "raise" if rule_held[cid] or verdict(raised["linf"]) != "none" else "seed"
                how.add({"raise": "raised", "seed": "seeded"}[variant])
                perturbed_run = f"{eid}{_PREFIX}{variant}{i}"
                base_y.append(trajectory(base, y))
                pert_y.append(trajectory(perturbed_run, y))
                if pre is not None:
                    rested.append((base, perturbed_run))
            g = gains(base_y, pert_y)
            out[cid] = {
                "claim": p["claim"],
                **{norm: {"gain": g[norm], "verdict": verdict(g[norm])} for norm in NORMS},
                "subject_species": sorted(p["subject"].values()),
                "perturbation": sorted(how),
                "raise": RAISE,
                "seed": seed,
                "experiments": sorted(experiments),
                "experiments_skipped": skipped,
            }
            if rested:
                # Where the pre-equilibration left the object, and the subject: time 0's L∞ gain.
                obj, subject = (
                    gains([at_start(b, o) for b, _ in rested], [at_start(q, o) for _, q in rested])
                    for o in (y, x)
                )
                out[cid]["pre_equilibration"] = {
                    "gain": obj["linf"],
                    "verdict": verdict(obj["linf"]),
                    "subject_gain": subject["linf"],
                }
        return out

    def experiments(
        self,
    ) -> tuple[dict[str, tuple[dict[str, float] | None, dict[str, float]]], list[str]]:
        """``({experiment: (pre, main)}, skipped)``: each measured experiment PEtab v1 can state,
        with the targets each period's conditions set, all of a period's rows together. That is one
        period from time 0 (``pre`` is ``None``), or one at ``-inf`` and one at 0, which is v1's
        pre-equilibration. The rest are skipped, by id (another start, a later period, a value that
        is an expression). Measurements with no experiment are the experiment ``""``, which sets
        nothing."""
        from petab.v2.C import TIME_PREEQUILIBRATION

        edf, cdf = self.problem.experiment_df, self.problem.condition_df
        changes = {}
        if cdf is not None and len(cdf):
            changes = {
                str(c): dict(zip(sub["targetId"], sub["targetValue"]))
                for c, sub in cdf.groupby("conditionId", sort=False)
            }
        out, skipped = {}, []
        for eid in sorted(set(self.problem.measurement_df["experimentId"])):
            if not eid or edf is None:
                out[eid] = (None, {})
                continue
            periods = {}
            for t, rows in edf[edf["experimentId"] == eid].groupby("time"):
                periods[float(t)] = targets = {}
                for cids in rows["conditionId"].dropna():
                    for c in str(cids).split(";"):
                        targets |= changes.get(c.strip(), {})
            if list(periods) not in ([0.0], [TIME_PREEQUILIBRATION, 0.0]) or not all(
                _number(v) for targets in periods.values() for v in targets.values()
            ):
                skipped.append(eid)
                continue
            *pre, main = ({str(t): float(v) for t, v in p.items()} for p in periods.values())
            out[eid] = (pre[0] if pre else None, main)
        return out, skipped

    def span(self, experiment: str) -> float:
        """The last time ``experiment`` is measured at."""
        m = self.problem.measurement_df
        return float(m.loc[m["experimentId"] == experiment, "time"].max())


def _amount(sbml, name: str):
    """Amount ``name`` of model ``sbml`` as an expression over its species (and parameters): the
    species itself where the server named one the amount is exactly, else its assignment rule,
    through any other rule it reads, but never through a species' own rule (an input's protocol:
    the species is what the amount counts). Not always a sum: over the blocks of a monomer the
    reduction split, an amount is their product (``petab_mcp.reduction.amount``)."""
    import libsbml
    import sympy

    if sbml.getSpecies(name) is not None:
        return sympy.Symbol(name)
    rules = {
        r.getVariable(): r
        for r in sbml.getListOfRules()
        if r.isAssignment() and sbml.getSpecies(r.getVariable()) is None
    }
    if name not in rules:
        raise ValueError(f"the rebuilt model has no amount {name}")

    def expr(var: str):
        text = libsbml.formulaToL3String(rules[var].getMath())
        # Every name a Symbol, so a species named `S` or `E` is not sympy's; a call (`exp(`) is not.
        names = {s: sympy.Symbol(s) for s in re.findall(r"[A-Za-z_]\w*(?!\w|\s*\()", text)}
        e = sympy.sympify(text, locals=names)
        return e.xreplace({s: expr(str(s)) for s in e.free_symbols if str(s) in rules})

    e = expr(name)
    if not any(sbml.getSpecies(str(s)) is not None for s in e.free_symbols):
        raise ValueError(f"amount {name} reads no species")
    return e


def _perturbable(
    sbml, species: str, scale: str, offset: str | None, *, set_by_conditions: bool
) -> None:
    """Make ``species``' start ``scale * <its own>``, plus ``offset`` where the seed goes to it;
    for one an assignment rule holds (an input's protocol), the rule ``scale * <its own>``. One the
    conditions set starts at what they set, so it is perturbed through them instead."""
    import libsbml

    rule = sbml.getAssignmentRuleByVariable(species)
    if rule is not None:
        rule.setMath(
            libsbml.parseL3Formula(f"{scale} * ({libsbml.formulaToL3String(rule.getMath())})")
        )
        return
    if set_by_conditions:
        return
    assignment = sbml.getInitialAssignmentBySymbol(species)
    if assignment is None:
        sp = sbml.getSpecies(species)
        start = sp.getInitialAmount() if sp.isSetInitialAmount() else sp.getInitialConcentration()
        assignment = sbml.createInitialAssignment()
        assignment.setSymbol(species)
        assignment.setMath(libsbml.parseL3Formula(repr(float(start or 0.0))))
    own = libsbml.formulaToL3String(assignment.getMath())
    formula = f"{scale} * ({own})" + (f" + {offset}" if offset else "")
    assignment.setMath(libsbml.parseL3Formula(formula))


def _fitted(vdir: Path, sbml) -> dict[str, float]:
    """The version's fitted value of each parameter of its model (its ``parameters.tsv``)."""
    import pandas as pd

    table = pd.read_csv(vdir / "parameters.tsv", sep="\t", index_col=0)
    model = {p.getId() for p in sbml.getListOfParameters()}
    return {
        str(pid): float(value)
        for pid, value in table["nominalValue"].items()
        if str(pid) in model and _number(value)
    }


def _number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _grid(end: float) -> list[float]:
    import numpy as np

    if not end > 0:
        return [0.0]
    return [0.0, *np.geomspace(end * 10.0**-_GRID_DECADES, end, _GRID_POINTS).tolist()]


# --------------------------------------------------------------------------------- the record
def record(
    run_dir: str | Path,
    constraints_file: str | Path,
    clock: Any,
    version: int | None,
    *,
    entity_map: str | Path | None,
    build_dir: str | Path,
) -> dict[str, Any]:
    """The ``causal_constraints`` block of ``result.json``: the constraints the run was held to,
    what they cost (the clock's refusals, :func:`first_satisfying_version` and the ``make_changes``
    calls it took), and the behaviour of the judged ``version`` (:func:`behaviour`)."""
    from petab_mcp import constraints as constraints_mod

    from .analysis import read_version_json, store_dir

    constraints = constraints_mod.read(constraints_file)
    by = dict(clock.by_constraint)
    first = first_satisfying_version(constraints, run_dir)
    built = first is not None and bool(
        read_version_json(store_dir(run_dir) / "versions" / f"v{first}", "statements.json")
    )
    return {
        "file": str(constraints_file),
        "constraints": constraints.texts(),
        "refusals": clock.reject_causes.get(CONSTRAINT_VIOLATED, 0),
        # Every constraint, 0 included, so a sweep can average per constraint.
        "refusals_by_constraint": {c.id: by.pop(c.id, 0) for c in constraints.items} | by,
        "first_satisfying_version": first,
        # The calls up to and including the one that committed it. Every version built from
        # statements was a make_changes the constraints let through, so the first such is the
        # run's first commit; a version built otherwise (the shipped seed) took none, and makes
        # only a set of negative constraints.
        "edits_to_first_satisfying": (
            None if first is None else clock.first_commit_edit if built else 0
        ),
        "behaviour": (
            behaviour(run_dir, version, constraints, entity_map=entity_map, build_dir=build_dir)
            if version is not None and version >= 0
            else None
        ),
    }
