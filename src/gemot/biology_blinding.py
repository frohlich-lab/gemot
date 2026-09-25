"""An anonymous PEtab view for the biology-blind ablation.

Renaming tool replies alone leaves names in formulas, SQL results and model exports. Instead
the server loads anonymized tables and SBML. The numerical design is unchanged; descriptive
metadata is omitted. The private bijection stays on the harness side, and the judge's caller
translates identifiers back so diagnostics and reference-sigma pairing retain their meaning.
The agent has only structured MCP tools, not access to these host files or this mapping.
"""
from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

_TOKEN = re.compile(r"\b[A-Za-z_]\w*\b")
_PLACEHOLDER = re.compile(r"^(observable|noise)Parameter(\d+)_(.+)$")
_RESERVED = {"time", "t", "true", "false", "nan", "inf", "INF", "NaN", "pi", "exponentiale", "avogadro"}
_COLUMNS = {
    "observables": ("observableId", "observableFormula", "noiseFormula",
                    "observableTransformation", "noiseDistribution"),
    "parameters": ("parameterId", "parameterScale", "lowerBound", "upperBound",
                   "nominalValue", "estimate", "initializationPriorType",
                   "initializationPriorParameters", "objectivePriorType", "objectivePriorParameters"),
    "measurements": ("observableId", "preequilibrationConditionId", "simulationConditionId",
                     "measurement", "time", "observableParameters", "noiseParameters",
                     "datasetId", "replicateId"),
}


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
                self.forward[token] = f"{kind}Parameter{number}_{self.add(observable, 'observable')}"
            elif not value[match.end():].lstrip().startswith("("):
                self.add(token, "readout" if token.startswith("readout_") else "symbol")

    def register_table(self, kind, table) -> None:
        for column, prefix in (("observableId", "observable"), ("parameterId", "parameter"),
                               ("conditionId", "condition"), ("simulationConditionId", "condition"),
                               ("preequilibrationConditionId", "condition"),
                               ("datasetId", "dataset"), ("replicateId", "replicate")):
            if column in table:
                for value in table[column].dropna().unique():
                    if isinstance(value, str):
                        self.add(value, prefix)
        if kind == "conditions":
            for column in table:
                if column not in {"conditionId", "conditionName"}:
                    self.add(column, "input")
                    for value in table[column]:
                        self.expression(value)
        else:
            for column in ("observableFormula", "noiseFormula", "observableParameters", "noiseParameters"):
                if column in table:
                    for value in table[column]:
                        self.expression(value)

    def table(self, kind, table):
        self.register_table(kind, table)
        columns = ([c for c in table if c != "conditionName"] if kind == "conditions"
                   else [c for c in _COLUMNS[kind] if c in table])
        result = table[columns].copy()
        for column in columns:
            # These columns contain enums or numbers, not identifiers. A source parameter
            # called "normal" or "log10" must not rewrite a distribution or scale.
            if column in {"parameterScale", "observableTransformation", "noiseDistribution",
                          "initializationPriorType", "objectivePriorType", "measurement", "time",
                          "lowerBound", "upperBound", "nominalValue", "estimate",
                          "initializationPriorParameters", "objectivePriorParameters"}:
                continue
            result[column] = result[column].map(lambda value: _translate(value, self.forward))
        if kind == "conditions":
            result = result.rename(columns={c: self.forward[c] for c in columns if c != "conditionId"})
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


def prepare(source_yaml: Path, work: Path, validation_dir: Path | None = None) -> BlindNames:
    """Write a fresh anonymous working problem; keep the source and answer key untouched."""
    import pandas as pd
    import petab.v1 as petab
    import yaml

    private = private_dir(work)
    if private.exists():
        shutil.rmtree(private)
    private.mkdir(parents=True)
    problem = petab.Problem.from_yaml(str(source_yaml))
    tables = {"parameters": problem.parameter_df.reset_index(),
              "observables": problem.observable_df.reset_index(),
              "conditions": problem.condition_df.reset_index(),
              "measurements": problem.measurement_df.copy()}
    document = problem.model.sbml_model.getSBMLDocument()
    names = BlindNames(reserved={element.getId() for element in document.getListOfAllElements()
                                if element.isSetId()})
    for table in tables.values():
        for column in table:
            names.reserved.add(str(column))
            for value in table[column]:
                if isinstance(value, str):
                    names.reserved.update(_TOKEN.findall(value))
    for kind, table in tables.items():
        names.register_table(kind, table)
    names.protocols = _anonymous_protocols(Path(source_yaml).parent / "inputs.tsv", names)
    sbml = _anonymous_sbml(document, names)
    work.mkdir(parents=True, exist_ok=True)
    (work / "model.xml").write_text(sbml)
    for kind, table in tables.items():
        names.table(kind, table).to_csv(work / f"{kind}.tsv", sep="\t", index=False)
    (work / "problem.yaml").write_text(yaml.safe_dump({
        "format_version": 1, "parameter_file": "parameters.tsv", "problems": [{
            "sbml_files": ["model.xml"], "condition_files": ["conditions.tsv"],
            "observable_files": ["observables.tsv"], "measurement_files": ["measurements.tsv"],
        }],
    }, sort_keys=False))
    # Held-out files remain outside the problem loaded by the agent's server. Extending
    # the mapping here cannot change any training alias or expose a held-out label.
    if validation_dir and (validation_dir / "measurements.tsv").is_file():
        destination = private / "validation"
        destination.mkdir()
        for kind in ("measurements", "conditions", "parameters"):
            path = validation_dir / f"{kind}.tsv"
            if path.is_file():
                table = pd.read_csv(path, sep="\t")
                names.table(kind, table).to_csv(destination / path.name, sep="\t", index=False)
        units = validation_dir / "units.tsv"
        if units.is_file():
            # The refit units of a transfer set: condition ids under their training aliases,
            # unit ids as they are (harness-private, never shown to the agent).
            table = pd.read_csv(units, sep="\t", dtype=str)
            table["conditionId"] = table["conditionId"].map(lambda v: _translate(v, names.forward))
            table.to_csv(destination / units.name, sep="\t", index=False)
    (private / "identifiers.json").write_text(json.dumps({
        "version": 1, "identifiers": names.forward, "protocols": names.protocols,
    }, indent=2) + "\n")
    return names


def _anonymous_protocols(path: Path, names: BlindNames) -> list[tuple[str, str]]:
    """The problem's input protocols (``inputs.tsv``) in the run's aliases.

    A protocol is numerical design, not biology: a time-dependent input the condition table only
    parameterises -- Fujita's EGF stimulus is a pulse or a ramp only under its rule, and its rate
    and end time are constant in training. The visible arms read the rule, with its real names, in
    the scenario's known-constants note, which the blind task replaces; without it here the
    blinded agent saw three anonymous columns and an instruction to use them, and put the
    constant two in roles no data could identify. Each rule may read only condition columns and
    ``time``: anything else would carry a name the aliases do not cover, so it is refused.
    """
    if not path.is_file():
        return []
    import pandas as pd

    out = []
    for row in pd.read_csv(path, sep="\t", dtype=str).itertuples():
        quantity = names.add(row.inputId, "stimulus")
        rule = _translate(row.formula, names.forward)
        # Only the row's own condition columns: every table symbol has an alias by now, so
        # accepting any alias would let a rule read a model parameter as ``parameter_005``.
        columns = [c for c in str(row.conditionInputs).split(";") if c]
        allowed = {names.forward[c] for c in columns
                   if names.forward.get(c, "").startswith("input_")}
        # A token followed by "(" is a function (piecewise, exp), not a name.
        leaked = sorted({match[0] for match in _TOKEN.finditer(rule)
                         if match[0] not in allowed and match[0] not in _RESERVED
                         and not rule[match.end():].lstrip().startswith("(")})
        if leaked or len(allowed) != len(columns):
            raise ValueError(f"input protocol {row.inputId!r} must read only its condition "
                             f"columns {columns}; it reads {leaked or 'a column the condition '
                             'table does not have'}, so it cannot be anonymized")
        out.append((quantity, rule))
    return out


def private_dir(work: Path) -> Path:
    """Harness-owned files beside work/, outside the agent's problem tree."""
    return work.with_name(f".{work.name}_biology_blinding")


def read_names(work: Path) -> BlindNames | None:
    path = private_dir(work) / "identifiers.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    return BlindNames(data["identifiers"],
                      protocols=[tuple(pair) for pair in data.get("protocols", [])])


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
            raise McpError(type(error.error).model_validate(
                self._scrub(error.error.model_dump()))) from None
        return type(result).model_validate(self._scrub(result.model_dump()))
