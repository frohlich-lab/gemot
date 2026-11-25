"""The adversarial_prior arm's input: the finished run whose model the agent is told to assume wrong.

The arm (``PETAB_ADVERSARIAL_PRIOR``, :mod:`gemot.ablations`) asks whether a panel that settles on
one mechanism finds a different one when told the first is wrong. WHETHER the agent is shown a prior
is the arm; WHICH run it is shown is data. ``PETAB_PRIOR_RUN`` names that run, as an archived run's
directory (``examples/_runs_archive/<run_id>/``, :mod:`gemot.explorer.archive`), and
:func:`provenance` records it in ``result.json`` as ``prior_run``. A boolean could not say which.

The agent sees the prior's SCORED version (``result.json``'s ``scored_version``), in the text that
version was written in: its INDRA statements, else the Antimony the agent wrote
(``model_source.ant``), else the server's (``model.ant``). Its criterion value goes with it, unless
the progress_signals arm is withholding such numbers.

:func:`resolve` REFUSES, before the runner wipes or launches anything, rather than running something
other than what the record says:

* the arm on with no prior, or a prior named with the arm off;
* a prior from another problem, or another version of it, whose model answers different data;
* a prior with no scored version;
* a prior run under any other arm setting. A model built with the reference handoff would carry it
  into a run without one, and an INDRA run's model is statements the Antimony route cannot take. The
  prior's own ``adversarial_prior`` setting is not compared, so priors can be chained;
* the biology_blind arm, whose lead reads its own task (``prompts.BIOLOGY_BLIND_TASK``) and whose
  anonymous identifiers are not known to match between two runs.

Deliberately NOT enforced: that the agent's model differs from the prior. Whether the prompt alone
moves it is the arm's question, and how far it moved is read after the fact (for instance with
:mod:`gemot.structure_eval`'s observable coupling of the two models). Enforcement, if it comes, is
an editing constraint and so belongs to the server.

Kept free of ``petab_mcp`` like :mod:`gemot.scenario`, which imports it.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from . import ablations

if TYPE_CHECKING:
    from .scenario import Scenario

#: A scored version's model files, most to least the agent's own text. Only a model built from
#: statements has ``statements.json``; ``model_source.ant`` is absent from archives older than it.
_MODEL_FILES = ("statements.json", "model_source.ant", "model.ant")


@dataclass(frozen=True)
class PriorRun:
    """The prior as the agent is shown it, and the fields :func:`provenance` records."""

    path: Path
    scored_version: int
    model_file: str
    model: str
    criterion: str | None
    criterion_value: float | None

    @property
    def run_id(self) -> str:
        return self.path.name

    @property
    def model_sha256(self) -> str:
        return hashlib.sha256(self.model.encode()).hexdigest()


def _named() -> str | None:
    return os.environ.get("PETAB_PRIOR_RUN", "").strip() or None


def load(path: Path, scenario: Scenario) -> PriorRun:
    """The archived run at ``path`` as a prior for ``scenario``, or an error naming why it is not one."""
    path = Path(path).expanduser().resolve()
    result_json = path / "result.json"
    if not result_json.is_file():
        raise FileNotFoundError(
            f"PETAB_PRIOR_RUN={path} holds no result.json; name an archived run's directory "
            "(examples/_runs_archive/<run_id>/)"
        )
    result = json.loads(result_json.read_text())
    if result.get("scenario") != scenario.name:
        raise ValueError(
            f"the prior run is of {result.get('scenario')!r}, not {scenario.name!r}: its model "
            "answers different data"
        )
    if result.get("example_version") != scenario.version:
        raise ValueError(
            f"the prior run is of {scenario.name} version {result.get('example_version')!r}, not "
            f"{scenario.version!r}: its model answers different data"
        )
    theirs = result.get("ablations") or {}
    differs = sorted(
        name
        for name, on in ablations.state().items()
        if name != "adversarial_prior" and theirs.get(name) != on
    )
    if differs:
        raise ValueError(
            f"the prior run's arm differs from this run's on {differs} (prior "
            f"{ {n: theirs.get(n) for n in differs} }): the model it hands over was built under "
            "another configuration"
        )
    version = result.get("scored_version")
    if not isinstance(version, int) or isinstance(version, bool) or version < 0:
        raise ValueError(f"the prior run scored no version (scored_version={version!r})")
    vdir = path / ".petab_mcp" / "versions" / f"v{version}"
    found = next((vdir / name for name in _MODEL_FILES if (vdir / name).is_file()), None)
    if found is None:
        raise FileNotFoundError(f"the prior run's scored version has no model file in {vdir}")
    value = result.get("criterion_value")
    finite = (
        isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    )
    return PriorRun(
        path=path,
        scored_version=version,
        model_file=found.name,
        model=found.read_text(),
        criterion=result.get("criterion"),
        criterion_value=float(value) if finite else None,
    )


def resolve(scenario: Scenario) -> PriorRun | None:
    """This run's prior, ``None`` off the arm; raises where the arm cannot mean what it says (see the
    module docstring). The runner calls it before anything is wiped or launched."""
    named = _named()
    if not ablations.enabled("adversarial_prior"):
        if named is not None:
            raise ValueError(
                "PETAB_PRIOR_RUN is set but the adversarial_prior arm is off "
                "(PETAB_ADVERSARIAL_PRIOR=1 turns it on): the prior would be recorded and never shown"
            )
        return None
    if named is None:
        raise ValueError(
            "the adversarial_prior arm is on but PETAB_PRIOR_RUN names no prior run: the arm is "
            "refused rather than run as the baseline"
        )
    if ablations.enabled("biology_blind"):
        raise ValueError("the adversarial_prior arm does not run with biology_blind on")
    return load(Path(named), scenario)


def note(scenario: Scenario) -> str:
    """What the arm appends to the agent's task: ``""`` off it, else
    ``prompts.ADVERSARIAL_PRIOR_NOTE_TMPL`` filled with the prior's model and score."""
    prior = resolve(scenario)
    if prior is None:
        return ""
    from . import prompts

    score = ""
    if prior.criterion_value is not None and ablations.enabled("progress_signals"):
        score = f" (its {prior.criterion or 'criterion'} was {prior.criterion_value:.1f})"
    return prompts.ADVERSARIAL_PRIOR_NOTE_TMPL.format(score=score, model=prior.model.strip())


def provenance(scenario: Scenario) -> dict | None:
    """``result.json``'s ``prior_run``: ``None`` off the arm, else which prior was shown. Never
    raises, because it is recorded by the handler that writes a failed run's result too; a prior
    that does not resolve is recorded with the ``error`` that refused it."""
    named = _named()
    if named is None and not ablations.enabled("adversarial_prior"):
        return None
    try:
        prior = resolve(scenario)
    except Exception as exc:  # recorded, not raised; see the docstring
        return {"path": named, "error": f"{type(exc).__name__}: {exc}"}
    assert prior is not None  # resolve raises for a prior named with the arm off
    return {
        "run_id": prior.run_id,
        "path": str(prior.path),
        "scored_version": prior.scored_version,
        "model_file": prior.model_file,
        "model_sha256": prior.model_sha256,
        "criterion": prior.criterion,
        "criterion_value": prior.criterion_value,
    }
