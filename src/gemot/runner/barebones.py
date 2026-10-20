"""The NULL-arm pipeline: stage the problem on disk, run the barebones agent, judge what it left.

Inverts :mod:`.main`: the agent works on the PEtab files first, and the server is started
afterwards on the ``problem.yaml`` it delivered — with ``score_shipped_model`` on, so the same
read tools and the same :func:`.judge._judge` score the delivered model at its delivered
parameter values. Same ``result.json`` shape and archive (plus a ``delivered/`` copy).

Between the agent and the judge: :func:`sandbox_preflight` (before any LLM call),
:func:`stage_workspace` (moves the problem into ``work/agent/`` — the only directory the agent
sees, so the harness's own ``run_live.json``/logs/result stay outside it — and hashes the design
tables), :func:`barebones_system_prompt` (the instructions for the problem's PEtab version,
naming the design tables the check hashes), :func:`barebones_task` (the same known-constants note
and entity names the panel gets),
and :func:`check_delivery` (design tables untouched, shipped parameter rows intact, petab lint,
and the parameters the server's reconcile would estimate, which the judge counts).
A tampered or unloadable delivery is recorded as a loss (``unscoreable_reason``), not an abort.
Reasons are in ``docs/RATIONALE.md`` (the ``#203`` entries).

The workspace holds what a modeller is handed and nothing else: ``problem.yaml``, the files it
lists, and ``inputs.tsv`` when the problem ships one. That is an allowlist, and must stay one. The
denylist it replaced (``reference.json`` and the ``model.ant`` seed) let everything else in the
problem directory through, and PEtabAgentBench tracks ``validation_v1_baseline.json`` for
perelson, crauste, fujita, lucarelli and raimundez: the reference's held-out score, and the
reference model's parameters by name with their fitted values. The seed stays
out because the judge's server would load it instead of the delivered SBML. What was withheld is
recorded in ``result.json`` as ``workspace_withheld``.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import os
import re
import shutil
import time
from pathlib import Path

from .. import ablations
from ..agent.sandbox import SANDBOX_HOME_DIR, SANDBOX_TMP_DIR, sandbox_preflight
from ..bio_sessions import open_bio_sessions
from ..explorer.archive import archive_run
from ..fit_budget import FIT_N_WORKERS, fit_time_budget
from ..agent.tools import ALLOWED_TOOLS
from ..prompts import (
    BAREBONES_ENTITIES_TMPL,
    BAREBONES_PETAB_FORMATS,
    BAREBONES_SYSTEM_PROMPT_TMPL,
    BAREBONES_TASK,
)
from ..scenario import Baseline, Scenario
from .clock import _make_call, _wrap_call_timeout, _wrap_tool_clock
from .endpoints import BIO_MCP_URLS
from .failure import _should_reraise, _summarize_exc, _write_degraded_result
from .judge import _judge
from .main import _petab_server, cleanup
from .provenance import _base_meta, _now_iso, _write_live_marker

#: The agent's workspace is THIS subdirectory of ``work/``, holding only the problem files. The
#: harness's own files — ``run_live.json`` (which carries the reference score), ``result.json``,
#: the server logs — are written to ``work/`` itself and so never enter the sandbox mount.
AGENT_SUBDIR = "agent"

#: Delivered files copied beside the archived result, size-capped.
_ARCHIVE_FILE_CAP_BYTES = 2_000_000
_ARCHIVE_SUFFIXES = (
    ".yaml",
    ".yml",
    ".tsv",
    ".xml",
    ".sbml",
    ".ant",
    ".py",
    ".sh",
    ".md",
    ".txt",
    ".json",
    ".log",
    ".csv",
)

#: Kept back from the END of the wall budget for compiling and scoring the delivery: the agent stops
#: this much early, and the server gets exactly this long to compile + score before it listens. So
#: the scoring ends inside ``--max-wall-seconds`` or is recorded as ``delivered_model_unscoreable``,
#: and the margin a sweep keeps below Slurm's ``--time`` stays the judge's. When the scoring ran in
#: that margin, a slow compile was killed with no ``result.json`` and the run dropped (#8).
_SCORE_RESERVE_S = 1200.0

#: The scoring's limit for a run with no wall budget, which has no outer limit to finish inside:
#: this only cuts a hung compile.
_SCORE_UNBUDGETED_S = 3600.0


#: The tables that make up the experimental design, which the agent may not touch. A v1 problem
#: has no experiment table.
_DESIGN_ROLES = ("measurement", "condition", "experiment", "observable")

#: The other tables a ``problem.yaml`` may list: staged with the problem, not held to their bytes.
_OTHER_ROLES = ("mapping", "visualization")


# --------------------------------------------------------------------------- staging
def _design_files(work: Path) -> dict[str, list[str]]:
    """The design-table and parameter-table file names ``problem.yaml`` lists, by role: from its
    first ``problems`` entry in PEtab v1, from the top level in v2."""
    import yaml
    from petab.versions import get_major_version

    doc = yaml.safe_load((work / "problem.yaml").read_text(encoding="utf-8")) or {}
    if get_major_version(doc) == 1:
        problems = doc.get("problems") or [{}]
        listed = problems[0] if problems else {}
        parameters = [doc["parameter_file"]] if doc.get("parameter_file") else []
    else:
        listed = doc
        parameters = list(doc.get("parameter_files") or [])
    return {
        **{role: list(listed.get(f"{role}_files") or []) for role in _DESIGN_ROLES},
        "parameter": parameters,
    }


def _staged_files(work: Path) -> set[str]:
    """What the agent is handed: ``problem.yaml``, every file it lists (the tables
    :func:`_design_files` reads, the model -- v1's ``sbml_files``, the ``location`` of each of v2's
    ``model_files`` -- and any mapping or visualization table), and ``inputs.tsv`` (the problem's
    input protocols) when it ships one.

    Separate from :func:`_design_files`, which also reads the YAML the agent DELIVERS: a model
    listing the agent wrote malformed must reach the linter, not read as a tampered design."""
    import yaml
    from petab.versions import get_major_version
    from petab_mcp.protocols import FILENAME as INPUTS_FILENAME

    doc = yaml.safe_load((work / "problem.yaml").read_text(encoding="utf-8")) or {}
    if get_major_version(doc) == 1:
        listed = (doc.get("problems") or [{}])[0]
        models = list(listed.get("sbml_files") or [])
    else:
        listed = doc
        models = [spec["location"] for spec in (doc.get("model_files") or {}).values()]
    staged = {"problem.yaml", *models}
    for role in _OTHER_ROLES:
        staged.update(listed.get(f"{role}_files") or [])
    for names in _design_files(work).values():
        staged.update(names)
    if (work / INPUTS_FILENAME).is_file():
        staged.add(INPUTS_FILENAME)
    return staged


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stage_workspace(work: Path) -> dict:
    """Move the problem from ``work/`` into ``work/agent/`` (the agent's workspace), delete the rest
    of ``work/``, and return what the prompt and the delivery check need afterwards: ``agent_dir``,
    the ``withheld`` names, the problem's ``petab_version`` (its YAML's major version), the
    design-table hashes, the shipped parameter rows' ``(estimate, nominalValue)``, and the shipped
    parameter table as the server reads it (``shipped_parameter_df``, for :func:`_reconciled`).

    An allowlist (:func:`_staged_files`): whatever else the problem directory holds is withheld."""
    from petab.versions import get_major_version
    from petab_mcp.problem import WorkingProblem

    handed = _staged_files(work)
    agent_dir = work / AGENT_SUBDIR
    agent_dir.mkdir()
    for name in sorted(handed):
        (agent_dir / name).parent.mkdir(parents=True, exist_ok=True)
        (work / name).rename(agent_dir / name)
    withheld = []
    for f in sorted(work.iterdir()):
        if f == agent_dir:
            continue
        if f.is_dir():
            shutil.rmtree(f)
        else:
            f.unlink()
        withheld.append(f.name)
    for name in (SANDBOX_HOME_DIR, SANDBOX_TMP_DIR):
        (agent_dir / name).mkdir(exist_ok=True)
    files = _design_files(agent_dir)
    hashes = {}
    for role in _DESIGN_ROLES:
        for name in files[role]:
            hashes[name] = _sha256(agent_dir / name)
    shipped_rows = {}
    for name in files["parameter"]:
        shipped_rows.update(_parameter_rows(agent_dir / name))
    return {
        "agent_dir": agent_dir,
        "withheld": withheld,
        "petab_version": get_major_version(str(agent_dir / "problem.yaml")),
        "files": files,
        "design_hashes": hashes,
        "shipped_rows": shipped_rows,
        # Taken as the session takes its `shipped_parameter_df`, before the agent can touch it.
        "shipped_parameter_df": WorkingProblem.from_yaml(
            agent_dir / "problem.yaml"
        ).parameter_df.copy(deep=True),
    }


def _estimate_flag(cell) -> int | None:
    """A parameter table's ``estimate`` cell as 0/1 -- v1's ``0``/``1`` or v2's ``false``/``true``,
    so a row compares equal across the two -- or ``None`` where it cannot be read."""
    text = (cell or "").strip().lower()
    if text in ("true", "false"):
        return int(text == "true")
    try:
        return int(float(text))
    except ValueError:
        return None


def _parameter_rows(path: Path) -> dict[str, tuple[int | None, float | None]]:
    """``{parameterId: (estimate, nominalValue)}`` from a parameter TSV, tolerant of a table the
    agent left malformed (a row that cannot be parsed is reported as ``(None, None)``)."""
    import csv

    rows: dict[str, tuple[int | None, float | None]] = {}
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8", newline="") as fh:
        for rec in csv.DictReader(fh, delimiter="\t"):
            pid = (rec.get("parameterId") or "").strip()
            if not pid:
                continue
            est = _estimate_flag(rec.get("estimate"))
            try:
                nominal = float(rec.get("nominalValue", ""))
            except (TypeError, ValueError):
                nominal = None
            rows[pid] = (est, nominal)
    return rows


# --------------------------------------------------------------------------- delivery checks
def check_delivery(work: Path, staged: dict) -> dict:
    """What the agent left behind, against the rules it was given: ``design_modified`` (design
    tables whose bytes changed or left the YAML, and any the YAML lists beside them),
    ``shipped_rows_violated`` (shipped parameter rows removed, re-flagged, or — fixed ones —
    re-valued), ``model_ant_set_aside``, ``petab_lint`` (``{ok, messages}``), ``model_authored``,
    and, for a delivery that lints, ``estimated`` and ``uncharged`` (:func:`_reconciled`)."""
    out: dict = {
        "design_modified": [],
        "shipped_rows_violated": [],
        "model_ant_set_aside": False,
        "petab_lint": {"ok": False, "messages": []},
        "model_authored": None,
        "estimated": None,
        "uncharged": [],
    }
    # a model.ant beside the YAML would be loaded instead of the SBML (session.py prefers it)
    ant = work / "model.ant"
    if ant.exists():
        ant.rename(work / "model.ant.agent")
        out["model_ant_set_aside"] = True
    try:
        files = _design_files(work)
    except Exception as exc:  # noqa: BLE001 - a broken YAML is a delivery failure, reported below
        out["design_modified"] = sorted(staged["design_hashes"])
        out["petab_lint"] = {"ok": False, "messages": [f"problem.yaml unreadable: {exc}"]}
        return out
    listed = {n for role in _DESIGN_ROLES for n in files[role]}
    for name, digest in staged["design_hashes"].items():
        f = work / name
        if name not in listed or not f.is_file() or _sha256(f) != digest:
            out["design_modified"].append(name)
    # An extra measurement file beside the untouched originals is a changed design too: the judge
    # takes n from the delivered problem, so rows the model reproduces would lower its score (#1).
    for role in _DESIGN_ROLES:
        out["design_modified"].extend(sorted(set(files[role]) - set(staged["files"][role])))
    delivered_rows: dict = {}
    for name in files["parameter"]:
        delivered_rows.update(_parameter_rows(work / name))
    for pid, (est, nominal) in staged["shipped_rows"].items():
        got = delivered_rows.get(pid)
        if got is None:
            out["shipped_rows_violated"].append(f"{pid}: removed")
        elif got[0] != est:
            out["shipped_rows_violated"].append(f"{pid}: estimate {est} -> {got[0]}")
        elif est == 0 and got[1] != nominal:
            out["shipped_rows_violated"].append(f"{pid}: fixed value {nominal} -> {got[1]}")
    out["petab_lint"], out["model_authored"] = _lint(work / "problem.yaml")
    if out["petab_lint"]["ok"] and out["model_authored"]:
        try:
            out["estimated"], out["uncharged"] = _reconciled(
                work / "problem.yaml", staged["shipped_parameter_df"]
            )
        except Exception as exc:  # noqa: BLE001 - a model the server cannot reconcile is unscoreable
            out["petab_lint"] = {
                "ok": False,
                "messages": [f"the server's reconcile refuses the delivered model: {exc}"[:500]],
            }
    return out


def _reconciled(yaml_path: Path, shipped) -> "tuple[list[str], list[str]]":
    """``(estimated, uncharged)`` for the delivered problem as the panel's server accounts for a
    model: the ids its reconcile estimates, which is what the judge counts as the null's estimated
    parameters, and how that differs from the delivered table, one entry per row.

    The panel never has a model scored as written. The server's reconcile runs on every edit and
    gives each parameter the model declares an ESTIMATED row and each species' nonzero, non-unit
    initial value an estimated ``<sid>_init``, and no tool of the lead's can fix a parameter. A
    delivery is scored as shipped, so the same numbers cost the null nothing written as a fixed row
    the problem does not ship, as an SBML parameter value with no row, or as a numeric initial
    value (#1).

    So the server's own :func:`petab_mcp.reconcile.reconcile` runs here, on a copy of the delivered
    problem read as the server reads it (:class:`petab_mcp.problem.WorkingProblem`), with the
    parameter table the problem SHIPPED in place of the delivered one and ``shipped`` passed as
    the session passes its own. Shipped rows come back as the problem defined them and every other
    parameter as the reconcile defines it, so a row the agent added fixed comes back estimated.
    The delivered table's values are written into the copy's SBML first: they are the values the
    score reads, and the reconcile's 0/1 conventions for initial conditions are decided on them.
    The scored files are not touched.

    Charged, deliberately not refused: refusing would make the plain Antimony start ``A = 3.7`` a
    lost run for the null where the panel pays one parameter for it. Literals in the math are the
    server's ``hardcoded_constants``, for both arms. A rate law's local parameter is charged by
    neither: the reconcile reads global parameters only, and Antimony, which the null is told to
    write, emits none.
    """
    from petab_mcp import reconcile as R
    from petab_mcp.problem import WorkingProblem

    delivered = WorkingProblem.from_yaml(yaml_path)
    table = delivered.parameter_df
    sm = delivered.model.sbml_model
    for pid, nominal in table["nominalValue"].items():
        if sm.getParameter(str(pid)) is not None and nominal == nominal:  # NaN: keep the SBML's
            sm.getParameter(str(pid)).setValue(float(nominal))
    declared = {p.getId() for p in sm.getListOfParameters()}
    problem = delivered.replace(parameter_df=shipped.copy(deep=True))
    R.reconcile(problem, shipped=shipped.copy(deep=True))
    estimated, was = set(problem.x_free_ids), set(delivered.x_free_ids)
    reconciled = problem.parameter_df
    uncharged = [
        f"{pid} = {reconciled.loc[pid, 'nominalValue']:g}: "
        + ("estimate false -> true" if pid in table.index else "no parameter-table row")
        for pid in sorted(estimated - was)
    ]
    # Rows the reconcile no longer estimates, of the parameters the model declares: a shipped row
    # for one it never declared was never counted (`in_model: false`) and is not news.
    uncharged += [
        f"{pid}: "
        + ("estimate true -> false" if pid in reconciled.index else "pruned, nothing reads it")
        for pid in sorted((was - estimated) & declared)
    ]
    return sorted(estimated), uncharged


def _lint(yaml_path: Path) -> "tuple[dict, bool | None]":
    """petab's linter over the delivered problem in the PEtab version its YAML declares (v1's
    messages captured off its logger, v2's issues as it returns them), plus whether the SBML
    carries any dynamics."""
    messages: list[str] = []
    handler = logging.Handler()
    handler.emit = lambda record: messages.append(record.getMessage())  # type: ignore[method-assign]
    logger = logging.getLogger("petab.v1.lint")
    logger.addHandler(handler)
    try:
        from petab.versions import get_major_version

        if get_major_version(str(yaml_path)) == 1:
            import petab.v1 as petab

            problem = petab.Problem.from_yaml(str(yaml_path))
            has_problems = bool(petab.lint.lint_problem(problem))
        else:
            from petab import v2
            from petab.v2.lint import ValidationIssueSeverity

            problem = v2.Problem.from_yaml(str(yaml_path))
            issues = problem.validate()
            messages.extend(str(i) for i in issues if i.level >= ValidationIssueSeverity.WARNING)
            has_problems = issues.has_errors()
        model = problem.model
        authored = None
        if model is not None:
            sm = model.sbml_model
            authored = bool(sm.getNumReactions() > 0 or sm.getNumRules() > 0)
            # petab logs only "Not OK" for an SBML that fails libsbml's consistency check; the
            # errors themselves are what a reader of `unscoreable_reason` needs.
            if not model.is_valid():
                messages.extend(_sbml_errors(model.sbml_document))
        return {"ok": not has_problems, "messages": messages[:50]}, authored
    except Exception as exc:  # noqa: BLE001 - any load failure IS the lint verdict
        return {
            "ok": False,
            "messages": [*messages[:49], f"{type(exc).__name__}: {exc}"[:500]],
        }, None
    finally:
        logger.removeHandler(handler)


def _sbml_errors(doc, limit: int = 8) -> list[str]:
    """libsbml's consistency errors for ``doc`` — one line per distinct error id, with the count."""
    import libsbml

    doc.checkConsistency()
    log = doc.getErrorLog()
    seen: dict[int, tuple[int, str]] = {}
    for i in range(log.getNumErrors()):
        e = log.getError(i)
        if e.getSeverity() < libsbml.LIBSBML_SEV_ERROR:
            continue
        n, msg = seen.get(e.getErrorId(), (0, e.getMessage().strip().split("\n")[0][:200]))
        seen[e.getErrorId()] = (n + 1, msg)
    return [f"SBML error {eid} (x{n}): {msg}" for eid, (n, msg) in list(seen.items())[:limit]]


# --------------------------------------------------------------------------- results
def _write_unscoreable_result(work: Path, meta: dict, scenario: Scenario, reason: str) -> dict:
    """A ``result.json`` for a delivery that cannot be judged: a loss, not an ``aborted:`` result
    (which the sweep aggregation may drop as infrastructure noise)."""
    print(f"[null] the delivered problem cannot be scored ({reason}); recording the run as a loss.")
    result = {
        **meta,
        "scenario": scenario.name,
        "success": False,
        "delta_bic": None,
        "final_nll": None,
        "unscoreable_reason": reason,
        "run_ended_at": _now_iso(),
    }
    (work / "result.json").write_text(json.dumps(result, indent=2))
    return result


def _archive_delivery(work: Path, dest: Path) -> None:
    """Copy the delivered problem and the agent's small text artefacts beside the archived result
    (the version store, which holds the panel's models, is empty for this arm)."""
    out = dest / "delivered"
    out.mkdir(parents=True, exist_ok=True)
    for f in sorted(work.iterdir()):
        if not f.is_file() or f.suffix.lower() not in _ARCHIVE_SUFFIXES:
            continue
        if f.stat().st_size > _ARCHIVE_FILE_CAP_BYTES:
            continue
        with contextlib.suppress(OSError):
            shutil.copy2(f, out / f.name)


# --------------------------------------------------------------------------- the task
#: Names a lead-facing note may use that mean nothing to the null: the lead's tools and the
#: `model_edits` operations. `rename` is left out — in the notes it is the English verb.
_HARNESS_TOOL_NAMES = frozenset(ALLOWED_TOOLS) | {
    "make_changes",
    "finish",
    "add_reaction",
    "remove_reaction",
    "set_rate_law",
    "set_initial",
    "set_assignment",
    "remove_assignment",
    "model_edits",
    "base_version",
}
_HARNESS_TOOL_RE = re.compile(
    r"\b(" + "|".join(sorted(map(re.escape, _HARNESS_TOOL_NAMES))) + r")\b"
)


def check_note_is_tool_free(note: str) -> str:
    """``note`` unchanged, or ``ValueError`` naming any harness tool it mentions: a scenario's
    ``known_constants`` must carry only problem-specific facts (the lead's readout-mapping
    instruction is harness-owned, ``prompts.KNOWN_CONSTANTS_READOUT_NOTE``)."""
    residual = sorted(set(_HARNESS_TOOL_RE.findall(note)))
    if residual:
        raise ValueError(f"known_constants names harness tools: {residual}")
    return note


def barebones_system_prompt(
    staged: dict, *, execute_timeout: float, n_cpus: int, n_starts: int
) -> str:
    """``BAREBONES_SYSTEM_PROMPT_TMPL`` for the problem :func:`stage_workspace` staged: the slots
    for its PEtab version, and the design tables it may not modify named as its YAML lists them —
    the files :func:`check_delivery` holds to their staged bytes."""
    version = staged["petab_version"]
    slots = BAREBONES_PETAB_FORMATS.get(f"v{version}")
    if slots is None:
        raise ValueError(f"the null arm has no prompt for a PEtab v{version} problem")
    design = [f"`{name}`" for role in _DESIGN_ROLES for name in staged["files"][role]]
    if len(design) > 1:
        design = [f"{', '.join(design[:-1])} or {design[-1]}"]
    return BAREBONES_SYSTEM_PROMPT_TMPL.format(
        **slots,
        design=design[0],
        execute_timeout=execute_timeout,
        n_cpus=n_cpus,
        n_starts=n_starts,
    )


def barebones_task(scenario: Scenario) -> str:
    """``BAREBONES_TASK`` (which carries its own tool-free readout instruction) plus the
    scenario's ``known_constants`` and the entities the biocurator is handed: its derived
    ``entity_ids``, else its curated ``biocurator_entities``. The reference_handoff ablation off
    withholds the note and the curated names, not the derived ones (``Scenario.agent_entities``)."""
    from .. import ablations

    handoff = ablations.enabled("reference_handoff")
    task = BAREBONES_TASK
    if scenario.known_constants and handoff:
        task = f"{task}\n\n{check_note_is_tool_free(scenario.known_constants)}"
    names = list(getattr(scenario, "entity_ids", None) or []) or (
        list(getattr(scenario, "biocurator_entities", None) or []) if handoff else []
    )
    if names:
        task += BAREBONES_ENTITIES_TMPL.format(names=", ".join(str(n) for n in names))
    return task


# --------------------------------------------------------------------------- pipeline
def _split_wall_budget(max_wall_seconds: float | None) -> "tuple[float | None, float]":
    """``--max-wall-seconds`` as ``(the agent's budget, the scoring's limit after it)``: the limit
    is :data:`_SCORE_RESERVE_S` taken from the budget, or half of a budget too short for it. Without
    a budget the agent is unbounded and the scoring gets :data:`_SCORE_UNBUDGETED_S`."""
    if max_wall_seconds is None:
        return None, _SCORE_UNBUDGETED_S
    reserve = min(_SCORE_RESERVE_S, 0.5 * max_wall_seconds)
    return max_wall_seconds - reserve, reserve


async def _judge_delivered(
    scenario: Scenario,
    session,
    work: Path,
    args: argparse.Namespace,
    meta: dict,
    t0: float,
    agent_done: float,
) -> "tuple[int, dict]":
    """The judge over a server that holds the DELIVERED problem — the tail of :func:`.drive._drive`
    without the driver in the middle.

    Its clock starts with the compute no tool call saw: the agent's sandboxed commands, and
    everything since the agent stopped at ``agent_done`` (the delivery check, the server's compile
    and nominal score), recorded as ``scoring_seconds``. Without it ``llm_seconds``, which is wall
    minus ``compute_seconds``, booked all of that as LLM time (#8)."""
    scoring_s = time.monotonic() - agent_done
    meta = {**meta, "scoring_seconds": round(scoring_s, 1)}
    _wrap_call_timeout(session, args.n_starts)
    clock = _wrap_tool_clock(session)
    clock.seconds += meta["workspace_audit"]["shell_busy_seconds"] + scoring_s
    call = _make_call(session)
    desc = await call("describe_problem")
    print(
        f"[judge] delivered model '{desc.get('model_id')}' | "
        f"reactions={desc.get('n_reactions_detail')} | free params={desc.get('free_parameters')}"
    )
    if not desc.get("model_authored", False):
        meta = {**meta, "stop_reason": "no_model_built"}
    delivery = meta["delivery"]
    if delivery["uncharged"]:
        print(f"[judge] the reconcile changes the delivered estimates: {delivery['uncharged']}")
    # No baseline, as for a zero-shot build; every species the agent declared is "new".
    baseline = Baseline(nll=None, species=set(), structure={})
    return await _judge(
        scenario, call, baseline, work, meta, clock, t0, estimated=delivery["estimated"]
    )


async def _main_async_barebones(scenario: Scenario, args: argparse.Namespace) -> "tuple[int, dict]":
    if ablations.enabled("biology_blind"):
        raise SystemExit(
            "PETAB_BIOLOGY_BLIND=1 is not supported with --driver barebones; "
            "use --driver deepagents or unset PETAB_BIOLOGY_BLIND."
        )
    work = cleanup(scenario)
    started_at = _now_iso()
    t0 = time.monotonic()
    base_meta = _base_meta(scenario, args, started_at)
    staged = stage_workspace(work)
    agent_dir = staged["agent_dir"]
    print(
        f"[null] workspace staged at {agent_dir.relative_to(scenario.example_dir.parent)} "
        f"(withheld: {staged['withheld'] or 'nothing'})"
    )

    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("No OpenRouter API key: set OPENROUTER_API_KEY (or pass --api-key).")
    # An infrastructure abort (the node, not the agent), written as one.
    try:
        sandbox = sandbox_preflight(agent_dir, n_cpus=FIT_N_WORKERS)
    except RuntimeError as exc:
        return 1, _write_degraded_result(work, base_meta, scenario, f"sandbox: {exc}")
    print(
        f"[null] sandbox ok: bubblewrap {sandbox['bwrap']}, env {sandbox['env_prefix']} "
        "(read-only), workspace at /work, no network"
    )

    # One shell command may run as long as one server fit would.
    execute_timeout_s = fit_time_budget(args.n_starts)
    agent_wall_s, score_limit_s = _split_wall_budget(args.max_wall_seconds)
    system_prompt = barebones_system_prompt(
        staged, execute_timeout=execute_timeout_s, n_cpus=FIT_N_WORKERS, n_starts=args.n_starts
    )
    task = barebones_task(scenario)

    from ..tracing import record_run_result, run_span, setup_phoenix_tracing, trace_id_hex

    tracing = setup_phoenix_tracing(project_name=args.phoenix_project) if args.phoenix else None
    print(
        f"[null] driving {args.model} via OpenRouter + a barebones deepagent "
        f"(recursion_limit={args.recursion_limit}, execute timeout {execute_timeout_s:.0f}s, "
        f"scoring limit {score_limit_s:.0f}s)"
    )
    from ..agent.barebones import run_barebones

    span_attrs = {
        "scenario": scenario.name,
        "example_version": scenario.version,
        "model": args.model,
        "driver": args.driver,
        "temperature": args.temperature,
        "recursion_limit": args.recursion_limit,
        "n_starts": args.n_starts,
        "git_sha": base_meta["git_sha"],
        "run_started_at": started_at,
        "prompt_version": base_meta.get("prompt_version"),
        "skills_version": base_meta.get("skills_version"),
        "ablation_arm": base_meta.get("ablation_arm"),
        "ablation_state": base_meta.get("ablation_state"),
    }
    exit_code, result = 1, {}
    launched: dict = {}
    try:
        with run_span(tracing, name="model_build_run", attributes=span_attrs) as span:
            base_meta = {
                **base_meta,
                "phoenix_project": tracing.project_name if tracing else None,
                "phoenix_trace_id": trace_id_hex(span),
            }
            _write_live_marker(work, base_meta, scenario, include_reference=False)
            try:
                # The panel's bio-knowledge servers, opened the same supervised way.
                async with open_bio_sessions(BIO_MCP_URLS) as bio_sessions:
                    agent_out = await run_barebones(
                        agent_dir,
                        model=args.model,
                        base_url=args.base_url,
                        api_key=api_key,
                        system_prompt=system_prompt,
                        task=task,
                        recursion_limit=args.recursion_limit,
                        temperature=args.temperature,
                        execute_timeout_s=execute_timeout_s,
                        n_cpus=FIT_N_WORKERS,
                        max_wall_seconds=agent_wall_s,
                        bio_sessions=bio_sessions,
                        bio_literature=ablations.enabled("bio_literature"),
                    )
                agent_done = time.monotonic()
                delivery = check_delivery(agent_dir, staged)
                meta = {
                    **base_meta,
                    **agent_out,
                    "delivery": delivery,
                    "scoring_mode": "delivered_nominal",
                    "score_limit_seconds": score_limit_s,
                    "sandbox": sandbox,
                    "workspace_withheld": staged["withheld"],
                }
                violations = delivery["design_modified"] or delivery["shipped_rows_violated"]
                if violations:
                    result = _write_unscoreable_result(
                        work,
                        meta,
                        scenario,
                        "design_tampered: "
                        + "; ".join(
                            [f"modified {n}" for n in delivery["design_modified"]]
                            + delivery["shipped_rows_violated"]
                        )[:400],
                    )
                elif delivery["model_authored"] is False:
                    # The problem ships model-less, so its own lint fails on the undefined
                    # readout_* symbols. Reporting that as `petab_invalid` describes the SHIPPED
                    # state, not what the agent did — which is nothing.
                    result = _write_unscoreable_result(
                        work,
                        meta,
                        scenario,
                        "no_model_authored: the delivered problem still has no model",
                    )
                elif not delivery["petab_lint"]["ok"]:
                    result = _write_unscoreable_result(
                        work,
                        meta,
                        scenario,
                        "petab_invalid: " + "; ".join(delivery["petab_lint"]["messages"][:5])[:400],
                    )
                else:
                    build_root = scenario.example_dir / ".amici_build"
                    try:
                        async with _petab_server(
                            work,
                            build_root,
                            args,
                            scenario=scenario,
                            score_shipped_model=True,
                            handle_out=launched,
                            listen_timeout=score_limit_s,
                            yaml_path=agent_dir / "problem.yaml",
                        ) as server:
                            exit_code, result = await _judge_delivered(
                                scenario, server.session, work, args, meta, t0, agent_done
                            )
                    except (RuntimeError, TimeoutError) as exc:
                        # the server died compiling/scoring the delivery; server.log names why
                        tail = _log_tail(work / "server.log")
                        result = _write_unscoreable_result(
                            work,
                            meta,
                            scenario,
                            f"delivered_model_unscoreable: {_summarize_exc(exc)}"
                            + (f" | server.log: {tail}" if tail else ""),
                        )
                record_run_result(span, result)
            except BaseException as exc:
                if _should_reraise(exc):
                    raise
                server = launched.get("handle")
                rc = server.proc.poll() if server is not None else None
                result = _write_degraded_result(
                    work, base_meta, scenario, _summarize_exc(exc), server_returncode=rc
                )
                with contextlib.suppress(Exception):
                    record_run_result(span, result)
    finally:
        if tracing is not None:
            tracing.flush()
            print(
                f"[phoenix] traces recorded under project '{tracing.project_name}' "
                f"-> {tracing.ui_url}"
            )
    if result:
        dest = archive_run(work, result)
        if dest is not None:
            _archive_delivery(agent_dir, dest)
            result["archive_dir"] = str(dest)
            with contextlib.suppress(Exception):
                (work / "result.json").write_text(json.dumps(result, indent=2))
    return exit_code, result


def _log_tail(path: Path, lines: int = 8, chars: int = 600) -> str:
    """The last few lines of a log, one-lined and bounded, for a result.json reason string."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    tail = " | ".join(line.strip() for line in text.splitlines()[-lines:] if line.strip())
    return tail[-chars:]
