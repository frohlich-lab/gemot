"""One configuration object for the route, exchanged with the JAX-side worker as JSON.

Interpreters, wall clock, threads, whether the field is sized to the data, the dosing convention
the problem reader cannot see, and overrides for :class:`~.train.TrainConfig` and
:class:`~.distil.DistilConfig`. Overrides are plain mappings applied with
:func:`dataclasses.replace`, so the two configs' defaults stand wherever nothing is set. The PySR
operator lists are checked here: an operator without a sympy image cannot be written as antimony,
and the failure would otherwise surface hours later, after the training that fed it.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path


class ConfigError(ValueError):
    """A configuration the route refuses before anything trains."""


BINARY_OPERATORS = frozenset({"+", "-", "*", "/", "hill"})
UNARY_OPERATORS = frozenset({"exp", "log", "sqrt", "square", "cube"})
SELECTIONS = frozenset({"best", "accuracy"})
NON_NEGATIVE = ("jitter", "parsimony")
POSITIVE_INTEGERS = ("niterations", "population_size", "maxsize", "n_samples", "n_times", "workers")


@dataclass(frozen=True)
class Config:
    """``interpreter`` holds the JAX stack (unset: run here if it imports, else refuse);
    ``pysr_interpreter`` holds PySR (unset: this interpreter); ``seconds`` is the clock for the
    whole route; ``threads`` above one lifts the inherited one-thread BLAS pinning;
    ``dosed`` names condition columns that dose a
    state where no naming convention says so; ``train`` and ``distil`` override the two config
    types field by field."""

    interpreter: str | None = None
    pysr_interpreter: str | None = None
    seconds: float | None = None
    threads: int | None = None
    keep_candidates: bool = True
    dosed: tuple[str, ...] = ()
    train: dict = field(default_factory=dict)
    distil: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.seconds is not None and not self.seconds > 0:
            raise ConfigError(f"seconds must be positive, got {self.seconds!r}")
        if self.threads is not None and int(self.threads) < 1:
            raise ConfigError(f"threads must be at least 1, got {self.threads!r}")
        object.__setattr__(self, "dosed", tuple(self.dosed))
        d = self.distil
        for name in NON_NEGATIVE:
            if name in d and not float(d[name]) >= 0:
                raise ConfigError(f"distil.{name} must be non-negative")
        for name in POSITIVE_INTEGERS:
            if d.get(name) is not None and int(d[name]) < 1:
                raise ConfigError(f"distil.{name} must be a positive integer")
        if "selection" in d and d["selection"] not in SELECTIONS:
            raise ConfigError(f"distil.selection must be one of {sorted(SELECTIONS)}")
        for name, allowed in (
            ("binary_operators", BINARY_OPERATORS),
            ("unary_operators", UNARY_OPERATORS),
        ):
            if name in d and (not d[name] or set(d[name]) - allowed):
                raise ConfigError(f"distil.{name} must be a non-empty subset of {sorted(allowed)}")

    def as_dict(self) -> dict:
        return {**dataclasses.asdict(self), "dosed": list(self.dosed)}

    @classmethod
    def from_dict(cls, document: dict) -> "Config":
        unknown = set(document) - {f.name for f in dataclasses.fields(cls)}
        if unknown:
            raise ConfigError(f"unknown configuration keys: {sorted(unknown)}")
        return cls(**document)

    def dump(self, path: Path | str) -> Path:
        Path(path).write_text(json.dumps(self.as_dict(), indent=1) + "\n")
        return Path(path)

    @classmethod
    def load(cls, path: Path | str) -> "Config":
        document = json.loads(Path(path).read_text())
        if not isinstance(document, dict):
            raise ConfigError(f"{path} does not hold a JSON object")
        return cls.from_dict(document)

    def train_config(self, train, supplied=None):
        """``train.TrainConfig`` with :attr:`train` applied; a supplied config is the base."""
        base = train.TrainConfig() if supplied is None else supplied
        return dataclasses.replace(base, **self.train) if self.train else base

    def distil_config(self, distil, supplied=None):
        """``distil.DistilConfig`` with :attr:`distil` applied and :attr:`pysr_interpreter` as its
        interpreter; a supplied config is the base."""
        base = (
            distil.DistilConfig(interpreter=self.pysr_interpreter) if supplied is None else supplied
        )
        overrides = {k: tuple(v) if k.endswith("_operators") else v for k, v in self.distil.items()}
        return dataclasses.replace(base, **overrides) if overrides else base
