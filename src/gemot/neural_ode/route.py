"""Propose ODE models by training a neural vector field and distilling it into rate laws.

:func:`propose` takes a PEtab v1 problem directory and a :class:`Config` and returns an
:class:`Outcome`: Antimony proposals with the record of how each was made, or one
:class:`Refusal` saying why nothing was proposed. It fits nothing itself and reports no
likelihood; the distilled model is scored downstream by refitting it on the data, and the
field's own fit is a search statistic.

Where this interpreter has no JAX stack the route runs whole in :attr:`Config.interpreter` as
``python -m gemot.neural_ode.worker`` and reads one JSON payload back. The parent owns
the hard timeout; the worker gets an earlier scientific deadline with grace to write its payload,
so a finished route is never killed reporting.

The architecture is the caller's: one training and one distillation at the settings given. Two
ladders used to choose it here -- climbing training nll for the latent count, and taking the
widest field the measurements afford -- and both criteria were refuted by measurement (see
:mod:`.tune`), so they are gone rather than retuned. Where an architecture must be chosen, the
way that works is a small grid ranked by the DISTILLED model's BIC.

The distiller's ceiling
pays for its constants. A rung the clock cannot afford is censored, not evidence against it.

Nothing above function scope imports JAX, AMICI or petab; :func:`_modules` fetches the three
modules that do, so a test can stand them in.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import math
import os
import subprocess
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path

from .config import Config

METHOD = "neural_ode_symbolic"
#: Neighbouring finite timepoints further apart than this mark everything above the gap as settled.
SETTLED_GAP_RATIO = 1e3
_WORKER_GRACE_FRACTION, _WORKER_GRACE_CAP = 0.10, 600.0
_DISTIL_RESERVE_FRACTION, _DISTIL_RESERVE_MIN, _DISTIL_RESERVE_MAX = 0.15, 60.0, 1800.0
_REPEAT_DISTIL_GRACE = 60.0


class RouteError(Exception):
    """A malformed outcome or a problem directory the route cannot read."""


class LadderFailure(Exception):
    """A ladder produced no usable result; the message is the refusal's reason."""


@dataclass(frozen=True)
class Proposal:
    """Antimony over ``state_names``; ``n_constants`` is the distiller's count, a start for the
    refit's own. No likelihood: the distillation's fit is to the learned field, not the data."""

    antimony: str
    state_names: tuple[str, ...]
    model_name: str
    n_constants: int
    record: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.antimony.strip():
            raise RouteError(f"{self.model_name}: an empty Antimony string is not a proposal")

    def as_dict(self) -> dict:
        return {
            "antimony": self.antimony,
            "state_names": list(self.state_names),
            "model_name": self.model_name,
            "n_constants": self.n_constants,
            "record": dict(self.record),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Proposal":
        return cls(
            d["antimony"],
            tuple(d["state_names"]),
            d["model_name"],
            int(d["n_constants"]),
            dict(d.get("record") or {}),
        )


@dataclass(frozen=True)
class Refusal:
    reason: str


@dataclass(frozen=True)
class Outcome:
    """Proposals or a refusal, exactly one of the two."""

    problem: str
    proposals: tuple[Proposal, ...] = ()
    refusal: Refusal | None = None

    def __post_init__(self) -> None:
        if bool(self.proposals) == (self.refusal is not None):
            raise RouteError(f"{self.problem}: an Outcome carries proposals OR a refusal")

    def as_dict(self) -> dict:
        return {
            "method": METHOD,
            "problem": self.problem,
            "proposals": [p.as_dict() for p in self.proposals],
            "refusal": None if self.refusal is None else {"reason": self.refusal.reason},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Outcome":
        refusal = d.get("refusal")
        return cls(
            d["problem"],
            tuple(Proposal.from_dict(p) for p in d.get("proposals", ())),
            None if refusal is None else Refusal(refusal["reason"]),
        )


def problem_name(problem_dir: Path | str) -> str:
    problem_dir = Path(problem_dir)
    return problem_dir.parent.name if problem_dir.name == "problem" else problem_dir.name


def refuse_settled(yaml_path: Path) -> str | None:
    """``None`` for trajectory data, else why: every measurement at steady state (``inf``, or above
    a :data:`SETTLED_GAP_RATIO` gap in the time grid) leaves no trajectory to fit a field to. The
    route would not fail on it; it would fit, and the fit would be meaningless."""
    import yaml  # noqa: PLC0415

    document = yaml.safe_load(yaml_path.read_text()) or {}
    files = list(document.get("measurement_files") or [])
    for entry in document.get("problems") or []:
        files.extend(entry.get("measurement_files") or [])
    times: list[float] = []
    for name in files:
        with (yaml_path.parent / name).open(newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if not reader.fieldnames or "time" not in reader.fieldnames:
                raise RouteError(f"{name} has no 'time' column")
            times.extend(float(row["time"]) for row in reader)
    if not times:
        return f"{yaml_path}: the measurement tables are empty; nothing to train on"
    positive = sorted({t for t in times if not math.isinf(t) and t > 0})
    floor = None
    if len(positive) >= 2:
        ratio, upper = max(
            ((positive[i + 1] / positive[i], positive[i + 1]) for i in range(len(positive) - 1)),
            key=lambda g: g[0],
        )
        floor = upper if ratio > SETTLED_GAP_RATIO else None
    if all(math.isinf(t) or (floor is not None and t >= floor) for t in times):
        return (
            f"all {len(times)} measurements are at steady state; there is no trajectory for a "
            "neural vector field to fit"
        )
    return None


def _modules():
    """``(problem, train, distil)``: the three modules that import petab, AMICI and JAX. Raises
    :class:`ImportError` where that stack is absent, which is the signal to delegate."""
    from . import distil, problem, train  # noqa: PLC0415

    return problem, train, distil


def _worker_seconds(limit: float) -> float:
    """``limit`` less reporting grace: the difference is grace, not compute."""
    value = float(limit)
    if not math.isfinite(value) or value <= 0.0:
        raise RouteError(f"the time limit must be finite and positive, got {limit}")
    return value - min(_WORKER_GRACE_CAP, _WORKER_GRACE_FRACTION * value)


def _distil_reserve(seconds: float | None) -> float:
    """Seconds withheld from the latent ladder for the mandatory first distillation."""
    if seconds is None:
        return 0.0
    value = float(seconds)
    if not math.isfinite(value) or value <= 0.0:
        raise RouteError(f"the clock must be finite and positive, got {seconds}")
    return min(
        0.5 * value,
        max(_DISTIL_RESERVE_MIN, min(_DISTIL_RESERVE_MAX, _DISTIL_RESERVE_FRACTION * value)),
    )


def _package_path(scratch: Path) -> Path:
    """A directory holding only links to this interpreter's installed ``gemot`` and ``petab_mcp``,
    which the worker imports but its own interpreter need not have."""
    import importlib.util

    shim = scratch / "package_path"
    shim.mkdir(parents=True, exist_ok=True)
    for name in ("gemot", "petab_mcp"):
        link = shim / name
        link.unlink(missing_ok=True)  # a reused work dir may hold a link to an earlier install
        link.symlink_to(Path(importlib.util.find_spec(name).origin).parent)
    return shim


def _run_in_interpreter(
    problem_dir: Path,
    problem: str,
    config: Config,
    *,
    work_dir,
    seconds: float | None,
    local_reason: str,
) -> Outcome | None:
    """Run the whole route in :attr:`Config.interpreter`; ``None`` when none is named."""
    if not config.interpreter:
        return None
    if not Path(config.interpreter).is_file():
        return Outcome(
            problem,
            refusal=Refusal(
                f"interpreter {config.interpreter!r} is not a file; not attempted. Locally: {local_reason}"
            ),
        )
    scratch = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="neural_ode_"))
    scratch.mkdir(parents=True, exist_ok=True)
    payload = scratch / "neural_proposals.json"
    env = dict(os.environ)
    # This harness and the server it imports, not the rest of this interpreter's packages.
    paths = [str(_package_path(scratch))]
    env["PYTHONPATH"] = os.pathsep.join(
        paths + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    if config.threads and int(config.threads) > 1:  # unset, the inherited one-thread pinning stands
        threads = str(int(config.threads))
        for name in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        ):
            env[name] = threads
        env["XLA_FLAGS"] = (
            env.get("XLA_FLAGS", "") + " --xla_cpu_multi_thread_eigen=true"
            f" --xla_force_host_platform_device_count={threads}"
        ).strip()
    command = [
        config.interpreter,
        "-m",
        "gemot.neural_ode.worker",
        str(problem_dir),
        "--out",
        str(payload),
        "--config",
        str(config.dump(scratch / "config.json")),
        "--work",
        str(scratch / "work"),
    ]
    if seconds is not None:
        command += ["--seconds", str(_worker_seconds(seconds))]
    try:
        completed = subprocess.run(
            command, env=env, capture_output=True, text=True, timeout=seconds
        )
    except subprocess.TimeoutExpired:
        return Outcome(
            problem,
            refusal=Refusal(
                f"the JAX worker was killed after {round(seconds)}s; a partial training is not a proposal"
            ),
        )
    if not payload.is_file():
        tail = " | ".join((completed.stderr or completed.stdout or "").strip().splitlines()[-8:])
        return Outcome(
            problem,
            refusal=Refusal(
                f"the JAX worker exited {completed.returncode} without a payload. Last output: {tail}"
            ),
        )
    document = json.loads(payload.read_text())
    if not document.get("ok"):
        return Outcome(
            problem, refusal=Refusal(f"the JAX worker reported: {document.get('reason')}")
        )
    outcome = Outcome.from_dict(document["outcome"])
    stamp = {"jax_interpreter": document.get("interpreter"), "ran_out_of_process": True}
    return replace(
        outcome,
        proposals=tuple(replace(p, record={**p.record, **stamp}) for p in outcome.proposals),
    )


def propose(
    problem_dir: Path | str,
    config: Config | None = None,
    *,
    work_dir=None,
    seconds: float | None = None,
    train_config=None,
    distil_config=None,
) -> Outcome:
    """Train a neural ODE on a PEtab v1 problem, distil it, return the distilled models.

    ``seconds`` overrides :attr:`Config.seconds` as the clock for the whole route. Refuses on
    settled data, an absent yaml, a problem the reader cannot make consistent, an absent JAX
    stack with no interpreter configured, a solver that fails on the first rung, an absent PySR,
    or an empty distillation.
    """
    config = config or Config()
    problem_dir = Path(problem_dir)
    problem = problem_name(problem_dir)
    seconds = config.seconds if seconds is None else seconds
    yaml_path = problem_dir / "problem.yaml"
    try:
        if not yaml_path.is_file():
            raise RouteError(f"no problem.yaml in {problem_dir}")
        reason = refuse_settled(yaml_path)
    except (RouteError, ValueError) as exc:
        reason = str(exc)
    if reason is not None:
        return Outcome(problem, refusal=Refusal(reason))

    try:
        P, T, D = _modules()
    except ImportError as exc:
        unavailable = f"the JAX stack is not importable here: {exc}"
        delegated = _run_in_interpreter(
            problem_dir,
            problem,
            config,
            work_dir=work_dir,
            seconds=seconds,
            local_reason=unavailable,
        )
        return (
            delegated if delegated is not None else Outcome(problem, refusal=Refusal(unavailable))
        )

    try:
        node_problem = P.load(yaml_path, name=problem, dosed=tuple(config.dosed))
    except P.ProblemError as exc:
        return Outcome(problem, refusal=Refusal(str(exc)))
    train_config = config.train_config(T, train_config)
    distil_config = config.distil_config(D, distil_config)
    scratch = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="neural_ode_"))
    try:
        result, distilled, policy = _default_field(
            T, D, node_problem, train_config, distil_config, scratch
        )
    except LadderFailure as failure:
        return Outcome(problem, refusal=Refusal(str(failure)))

    entries = _entries(distilled, config.keep_candidates)
    if not entries:
        return Outcome(problem, refusal=Refusal("distillation produced no Antimony model"))
    common = {
        "method": METHOD,
        "train_config": result.config.key(),
        "train_settings": dataclasses.asdict(result.config),
        "distil_config": distilled.config.key(),
        "distil_settings": dataclasses.asdict(distilled.config),
        "tuning": policy,
        "field_nll": float(result.nll),
        "n_network_weights": int(result.n_network_weights),
        "n_ode_parameters": int(result.n_ode_parameters),
        "stage_objectives": list(result.stage_objectives),
        "training_notes": list(result.notes),
        "problem_warnings": list(node_problem.warnings),
        "warnings": list(distilled.warnings),
        "field_r2_is_not_a_likelihood": "agreement with the trained field, not the measurements",
    }
    proposals = tuple(
        Proposal(
            antimony,
            tuple(distilled.state_names),
            f"{problem}_node_{tag}",
            int(n_constants),
            _json_native(
                {
                    **common,
                    "tag": tag,
                    "complexity_budget": budget,
                    "field_r2": field_r2,
                    "expressions": [str(e) for e in expressions],
                }
            ),
        )
        for tag, budget, antimony, expressions, n_constants, field_r2 in entries
    )
    return Outcome(problem, proposals=proposals)


def _json_native(value):
    """The record as the JSON it crosses the process boundary as, so it reads back equal."""
    return json.loads(json.dumps(value, default=str))


def _default_field(T, D, node_problem, train_config, distil_config, scratch):
    """One training and one distillation at the given settings; ``(result, distilled, None)``."""
    result = T.train(node_problem, Path(scratch) / train_config.key(), train_config)
    if result.solver_failed:
        raise LadderFailure(f"the solver failed after {result.steps_run} steps; no field to distil")
    # The FITTED field, kept. Staging writes the network's initial weights and nothing writes them
    # again, so without this the trained field dies with the worker and only its distillations and
    # the scalars in the record survive -- a sweep cannot re-distil a field at other settings, or
    # re-score it, without retraining it. Beside the staged directory, never inside it: the bundle
    # is a copy of that directory and a destination within it would copy into itself.
    T.save_bundle(result, Path(scratch) / "field" / train_config.key())
    distilled = D.distil(node_problem, result, distil_config)
    if not distilled.pysr_available:
        raise LadderFailure(
            f"PySR is not available; placeholders are not a proposal. Warnings: "
            f"{list(distilled.warnings)}"
        )
    return result, distilled, None


def _entries(distilled, keep_candidates: bool):
    """``(tag, complexity_budget, antimony, expressions, n_constants, field_r2)``: the selected
    set first, then every point of the complexity front if kept."""
    entries = []
    if distilled.antimony:
        entries.append(
            (
                "selected",
                None,
                distilled.antimony,
                distilled.expressions,
                distilled.n_constants,
                distilled.field_r2,
            )
        )
    if keep_candidates:
        entries += [
            (
                f"c{c.complexity_budget}",
                c.complexity_budget,
                c.antimony,
                c.expressions,
                c.n_constants,
                c.field_r2,
            )
            for c in distilled.candidates
            if c.antimony
        ]
    return entries
