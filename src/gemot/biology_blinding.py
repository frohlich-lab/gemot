"""An anonymous PEtab view for the biology-blind ablation.

Renaming tool replies alone leaves names in formulas, SQL results and model exports. Instead
the server loads anonymized tables and SBML. The numerical design is unchanged; descriptive
metadata is omitted. The private bijection stays on the harness side, and the judge's caller
translates identifiers back so diagnostics and reference-sigma pairing retain their meaning.
The agent has only structured MCP tools, not access to these host files or this mapping.

The anonymous problem is PEtab v2 whichever version the source is. A v1 source is upgraded first,
by the loader the server itself uses (``petab_mcp.problem.WorkingProblem.from_yaml``), so the
blinded problem is the server's own v2 problem under other names, and a native v2 source takes
the same path. Blinding the v1 files and leaving the upgrade to the server would only work for
v1 sources.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

_TOKEN = re.compile(r"\b[A-Za-z_]\w*\b")
_PLACEHOLDER = re.compile(r"^(observable|noise)Parameter(\d+)_(.+)$")
_RESERVED = {
    "time",
    "t",
    "true",
    "false",
    "nan",
    "inf",
    "INF",
    "NaN",
    "pi",
    "exponentiale",
    "avogadro",
}
#: The columns each v2 table keeps; anything else (names, units, comments) is descriptive
#: metadata and is dropped. ``parameterScale`` is the server's estimation-scale extension.
_COLUMNS = {
    "observables": (
        "observableId",
        "observableFormula",
        "observablePlaceholders",
        "noiseFormula",
        "noisePlaceholders",
        "noiseDistribution",
    ),
    "parameters": (
        "parameterId",
        "parameterScale",
        "lowerBound",
        "upperBound",
        "nominalValue",
        "estimate",
        "priorDistribution",
        "priorParameters",
    ),
    "conditions": ("conditionId", "targetId", "targetValue"),
    "experiments": ("experimentId", "time", "conditionId"),
    "measurements": (
        "observableId",
        "experimentId",
        "time",
        "measurement",
        "observableParameters",
        "noiseParameters",
        "datasetId",
        "replicateId",
    ),
    "mapping": ("petabEntityId", "modelEntityId"),
}
#: Columns that hold enums or numbers, not identifiers. A source parameter called "normal" or
#: "log10" must not rewrite a distribution or scale.
_VALUES = {
    "parameterScale",
    "noiseDistribution",
    "priorDistribution",
    "priorParameters",
    "measurement",
    "time",
    "lowerBound",
    "upperBound",
    "nominalValue",
    "estimate",
}
#: The table files of the anonymous problem, by kind, and the YAML key that lists each.
#: Conditions, experiments and a mapping are written only when the problem has them.
_FILES = {kind: f"{kind}.tsv" for kind in _COLUMNS}
_YAML_KEYS = {
    "parameters": "parameter_files",
    "observables": "observable_files",
    "conditions": "condition_files",
    "experiments": "experiment_files",
    "measurements": "measurement_files",
    "mapping": "mapping_files",
}
_OPTIONAL = ("conditions", "experiments", "mapping")


def _translate(value, names):
    if isinstance(value, str):
        if value in names:
            return names[value]
        return _TOKEN.sub(lambda match: names.get(match[0], match[0]), value)
    if isinstance(value, list):
        return [_translate(item, names) for item in value]
    if isinstance(value, tuple):
        return tuple(_translate(item, names) for item in value)
    if isinstance(value, dict):
        return {_translate(key, names): _translate(item, names) for key, item in value.items()}
    return value


@dataclass
class BlindNames:
    """Run-local, injective names, allocated from training before seeing held-out data."""

    forward: dict[str, str] = field(default_factory=dict)
    reserved: set[str] = field(default_factory=set)
    #: The experiment's time-dependent inputs as ``(quantity, rule)`` pairs, in aliases: what a
    #: modeller is told about the design and the tables alone do not reveal.
    protocols: list[tuple[str, str]] = field(default_factory=list)

    def add(self, original: str, kind: str) -> str:
        if original in self.forward:
            return self.forward[original]
        number = 1
        used = self.reserved | set(self.forward.values())
        while (alias := f"{kind}_{number:03d}") in used:
            number += 1
        self.forward[original] = alias
        return alias

    def expression(self, value):
        """Register symbols, preserving mathematical functions and PEtab placeholders."""
        if not isinstance(value, str):
            return
        for match in _TOKEN.finditer(value):
            token = match[0]
            if token in self.forward or token in _RESERVED:
                continue
            placeholder = _PLACEHOLDER.fullmatch(token)
            if placeholder:
                kind, number, observable = placeholder.groups()
                self.forward[token] = (
                    f"{kind}Parameter{number}_{self.add(observable, 'observable')}"
                )
            elif not value[match.end() :].lstrip().startswith("("):
                self.add(token, "readout" if token.startswith("readout_") else "symbol")

    def register_table(self, kind, table) -> None:
        for column, prefix in (
            ("observableId", "observable"),
            ("parameterId", "parameter"),
            ("conditionId", "condition"),
            ("experimentId", "experiment"),
            ("datasetId", "dataset"),
            ("replicateId", "replicate"),
            # A condition's targets are the experiment's inputs, whatever model entity each is.
            ("targetId", "input"),
            ("petabEntityId", "entity"),
        ):
            if column in table:
                for value in table[column].dropna().unique():
                    # "" is a measurement with no experiment, not an identifier.
                    if isinstance(value, str) and value:
                        self.add(value, prefix)
        for column in (
            "observableFormula",
            "observablePlaceholders",
            "noiseFormula",
            "noisePlaceholders",
            "observableParameters",
            "noiseParameters",
            "targetValue",
        ):
            if column in table:
                for value in table[column]:
                    self.expression(value)

    def table(self, kind, table):
        self.register_table(kind, table)
        columns = [c for c in _COLUMNS[kind] if c in table]
        result = table[columns].copy()
        for column in columns:
            if column not in _VALUES:
                result[column] = result[column].map(lambda value: _translate(value, self.forward))
        return result

    def judge_call(self, call):
        """An original-vocabulary caller for the scripted self-test and private judge."""
        inverse = {alias: original for original, alias in self.forward.items()}

        async def translated(name, **kwargs):
            arguments = {key: _translate(value, self.forward) for key, value in kwargs.items()}
            return _translate(await call(name, **arguments), inverse)

        return translated


def _anonymous_sbml(document, names: BlindNames) -> str:
    import libsbml

    document = document.clone()
    document.unsetNotes()
    document.unsetAnnotation()
    document.unsetMetaId()
    document.unsetSBOTerm()
    elements = list(document.getListOfAllElements())
    ids = [element.getId() for element in elements if element.isSetId()]
    names.reserved.update(ids)
    for element in elements:
        if element.isSetId():
            names.add(element.getId(), "entity")
        element.unsetName()
        element.unsetNotes()
        element.unsetAnnotation()
        element.unsetMetaId()
        element.unsetSBOTerm()
        if isinstance(element, libsbml.Constraint):
            element.unsetMessage()
    # Two passes prevent a new name from being captured by another old identifier.
    temporary = {original: f"__blind_tmp_{i}" for i, original in enumerate(ids)}
    while set(temporary.values()) & (set(ids) | set(names.forward.values())):
        temporary = {old: "_" + new for old, new in temporary.items()}
    for mapping in (temporary, {temporary[old]: names.forward[old] for old in ids}):
        for element in elements:
            for old, new in mapping.items():
                element.renameSIdRefs(old, new)
                element.renameUnitSIdRefs(old, new)
            if element.isSetId() and element.getId() in mapping:
                if element.setId(mapping[element.getId()]) != libsbml.LIBSBML_OPERATION_SUCCESS:
                    raise ValueError("Cannot anonymize this SBML identifier")
    return libsbml.writeSBMLToString(document)


def _tables(problem) -> dict:
    """``problem``'s tables as plain frames, keyed by kind; an optional table it lacks is left out."""
    tables = {
        "parameters": problem.parameter_df.reset_index(),
        "observables": problem.observable_df.reset_index(),
        "conditions": problem.condition_df,
        "experiments": problem.experiment_df,
        "measurements": problem.measurement_df,
        "mapping": problem.mapping_df,
    }
    return {
        kind: table.copy()
        for kind, table in tables.items()
        if table is not None and (kind not in _OPTIONAL or len(table))
    }


def _write(table, path: Path) -> None:
    """One table as a v2 TSV: ``estimate`` spelled ``true``/``false``, as v2 has it."""
    from petab_mcp.problem import as_bool

    if "estimate" in table:
        table = table.assign(estimate=["true" if as_bool(v) else "false" for v in table.estimate])
    table.to_csv(path, sep="\t", index=False)


def prepare(source_yaml: Path, work: Path, validation_dir: Path | None = None) -> BlindNames:
    """Write a fresh anonymous working problem; keep the source and answer key untouched."""
    import pandas as pd
    import yaml
    from petab_mcp.problem import WorkingProblem
    from petab_mcp.validation_data import HELD_OUT_UNITS, as_v2, read_held_out

    private = private_dir(work)
    if private.exists():
        shutil.rmtree(private)
    private.mkdir(parents=True)
    problem = WorkingProblem.from_yaml(source_yaml)
    tables = _tables(problem)
    document = None if problem.model is None else problem.model.sbml_document
    names = BlindNames(
        reserved=set()
        if document is None
        else {element.getId() for element in document.getListOfAllElements() if element.isSetId()}
    )
    for table in tables.values():
        for column in table:
            names.reserved.add(str(column))
            for value in table[column]:
                if isinstance(value, str):
                    names.reserved.update(_TOKEN.findall(value))
    for kind, table in tables.items():
        names.register_table(kind, table)
    names.protocols = _anonymous_protocols(Path(source_yaml).parent, problem, names)
    work.mkdir(parents=True, exist_ok=True)
    config: dict = {"format_version": "2.0.0"}
    if document is not None:
        (work / "model.xml").write_text(_anonymous_sbml(document, names))
        # The model id is the YAML's key, not the SBML's: a fixed one names nothing.
        config["model_files"] = {"model": {"location": "model.xml", "language": "sbml"}}
    for kind, table in tables.items():
        _write(names.table(kind, table), work / _FILES[kind])
        config[_YAML_KEYS[kind]] = [_FILES[kind]]
    (work / "problem.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    # Held-out files remain outside the problem loaded by the agent's server. Extending
    # the mapping here cannot change any training alias or expose a held-out label.
    if validation_dir and (validation_dir / "measurements.tsv").is_file():
        destination = private / "validation"
        destination.mkdir()
        # Keyed by experiment against the SOURCE problem first, as the server would key it, so a
        # held-out row repeating a training design lands in that experiment's alias. The blinded
        # set is then v2 and the server's `as_v2` leaves it as it is.
        held_out = as_v2(read_held_out(validation_dir / "measurements.tsv"), problem)
        parameters = held_out.parameter_df
        for kind, table in (
            ("measurements", held_out.measurement_df),
            ("conditions", held_out.condition_df),
            ("experiments", held_out.experiment_df),
            ("parameters", None if parameters is None else parameters.reset_index()),
        ):
            if table is not None and len(table):
                _write(names.table(kind, table), destination / _FILES[kind])
        if held_out.units is not None:
            # The refit units of a transfer set: experiment ids under their training aliases,
            # unit ids as they are (harness-private, never shown to the agent).
            pd.DataFrame(
                {
                    "experimentId": [_translate(e, names.forward) for e in held_out.units],
                    "unitId": list(held_out.units.values()),
                }
            ).to_csv(destination / HELD_OUT_UNITS, sep="\t", index=False)
    (private / "identifiers.json").write_text(
        json.dumps(
            {
                "version": 1,
                "identifiers": names.forward,
                "protocols": names.protocols,
            },
            indent=2,
        )
        + "\n"
    )
    return names


def _anonymous_protocols(directory: Path, problem, names: BlindNames) -> list[tuple[str, str]]:
    """The problem's input protocols (``inputs.tsv``, read and checked by the server's
    :mod:`petab_mcp.protocols`) in the run's aliases.

    A protocol is numerical design, not biology: a time-dependent input the condition table only
    parameterises -- Fujita's EGF stimulus is a pulse or a ramp only under its rule, and its rate
    and end time are constant in training. The visible arms are shown the rule, with its real
    names, by the server; the blind task carries it instead. Without it here the blinded agent saw
    three anonymous columns and an instruction to use them, and put the constant two in roles no
    data could identify. The server refuses a rule that reads anything but condition targets and
    ``time``, or that does not change within a period.
    """
    from petab_mcp.protocols import load

    out = []
    for item in load(directory, problem):
        quantity = names.add(item.id, "stimulus")
        out.append((quantity, _translate(item.formula, names.forward)))
    return out


def private_dir(work: Path) -> Path:
    """Harness-owned files beside work/, outside the agent's problem tree."""
    return work.with_name(f".{work.name}_biology_blinding")


def read_names(work: Path) -> BlindNames | None:
    path = private_dir(work) / "identifiers.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    return BlindNames(
        data["identifiers"], protocols=[tuple(pair) for pair in data.get("protocols", [])]
    )


class AnonymousSession:
    """Mask host paths that could identify the example through a tool error or build log.

    The underlying server already uses anonymous identifiers. This proxy only hides its
    host location; the private judge retains the original session and diagnostic paths.
    """

    def __init__(self, session, example_dir: Path, *, call_tool=None):
        self.session = session
        self._call_tool = call_tool or session.call_tool
        self.paths = (str(example_dir.resolve()), str(example_dir), f"examples/{example_dir.name}")

    def __getattr__(self, name):
        return getattr(self.session, name)

    def _scrub(self, value):
        if isinstance(value, str):
            for path in self.paths:
                value = value.replace(path, "/task")
            return value
        if isinstance(value, list):
            return [self._scrub(item) for item in value]
        if isinstance(value, dict):
            return {self._scrub(key): self._scrub(item) for key, item in value.items()}
        return value

    async def call_tool(self, *args, **kwargs):
        from mcp.shared.exceptions import McpError

        try:
            result = await self._call_tool(*args, **kwargs)
        except McpError as error:
            raise McpError(
                type(error.error).model_validate(self._scrub(error.error.model_dump()))
            ) from None
        return type(result).model_validate(self._scrub(result.model_dump()))
