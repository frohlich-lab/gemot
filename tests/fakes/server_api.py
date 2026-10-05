"""Stand-ins for the server modules :mod:`gemot.server_api` hands out, for the suite outside
``tests/server/`` — which runs with no petab-mcp installed.

``tests/conftest.py`` swaps these in for every test outside ``tests/server/``. Each is the
SMALLEST faithful copy of what gemot reads: the noise-sigma thresholds, the fit settings (the
per-start cap, the worker count, the AMICI tolerance), the criteria's penalty formulas, the two
noise-sigma predicates, the observable-scaling resolver, a fixed skills version. Faithful matters, because
judge tests assert real numbers through them, so ``tests/server/test_server_api_fakes.py`` runs
every one against the real module and fails when they disagree; a server change that moves a
verdict then fails there, at the pin bump that brought it.

Anything NOT copied here raises :class:`NotFaked` naming the attribute. That is deliberate: a test
whose path needs more of the server than this belongs in ``tests/server/``, and should say so
rather than quietly run against a guess.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, Callable, Mapping


class NotFaked(AttributeError):
    """A server attribute this module does not stand in for."""


class _FakeModule(ModuleType):
    def __getattr__(self, name: str):
        raise NotFaked(
            f"tests/fakes/server_api.py does not fake petab_mcp.{self.__name__}.{name}: a test "
            "that needs it runs against the real server, under tests/server/"
        )


def _module(name: str, **attrs: Any) -> _FakeModule:
    mod = _FakeModule(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    return mod


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


# ----------------------------------------------------------------------------- constants
#: petab_mcp.constants: the noise-sigma thresholds the judge's gate shares with the server's.
NOISE_SIGMA_COLLAPSE_FRACTION = 0.01
NOISE_SIGMA_REFERENCE_FACTOR = 3.0
NOISE_SIGMA_BAND_ALPHA = 0.01
NOISE_SIGMA_BAND_MIN_DF = 3.0


# ------------------------------------------------------------------------------ criteria
#: petab_mcp.criteria.PENALTIES: ``(k, n, k_eff) -> penalty``, ``None`` where undefined.
PENALTIES: dict[str, Callable[[int, int, "float | None"], "float | None"]] = {
    "bic": lambda k, n, k_eff: k * math.log(n),
    "aic": lambda k, n, k_eff: 2.0 * k,
    "aicc": lambda k, n, k_eff: (
        None if n - k - 1 <= 0 else 2.0 * k + 2.0 * k * (k + 1) / (n - k - 1)
    ),
    "caic": lambda k, n, k_eff: k * (math.log(n) + 1.0),
    "hqc": lambda k, n, k_eff: None if n < 3 else 2.0 * k * math.log(math.log(n)),
    "tic": lambda k, n, k_eff: None if _finite_float(k_eff) is None else 2.0 * float(k_eff),
}


@dataclass(frozen=True)
class Criterion:
    """petab_mcp.criteria.Criterion: ``name``, ``from_name``, ``penalty``, ``value``,
    ``score_of``."""

    name: str = "bic"
    _penalty: Callable = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        name = str(self.name).strip().lower()
        if name not in PENALTIES:
            raise ValueError(f"unknown complexity criterion {self.name!r}")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "_penalty", PENALTIES[name])

    @classmethod
    def from_name(cls, name: "str | None") -> "Criterion | None":
        try:
            return cls(name) if isinstance(name, str) and name.strip() else None
        except ValueError:
            return None

    def penalty(self, k, n, k_eff=None):
        if k is None or n is None or k < 0 or n <= 0:
            return None
        return self._penalty(int(k), int(n), k_eff)

    def value(self, nll, k, n, k_eff=None):
        nll = _finite_float(nll)
        pen = self.penalty(k, n, k_eff)
        return None if (nll is None or pen is None) else 2.0 * nll + pen

    def score_of(self, record: Mapping) -> float | None:
        by_name = _finite_float(record.get(self.name))
        if by_name is not None:
            return by_name
        return (
            _finite_float(record.get("criterion_value"))
            if record.get("criterion") == self.name
            else None
        )


CANONICAL: dict[str, Criterion] = {name: Criterion(name) for name in PENALTIES}


def resolve(criterion: "Criterion | str | None" = None) -> Criterion:
    """petab_mcp.criteria.resolve: the argument, else ``PETAB_MCP_CRITERION``, else BIC."""
    if isinstance(criterion, Criterion):
        return criterion
    if criterion is not None:
        return Criterion(criterion)
    return Criterion.from_name(os.environ.get("PETAB_MCP_CRITERION")) or Criterion()


# ------------------------------------------------------------------------------- fitting
def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    try:
        val = float(raw) if raw is not None else default
    except ValueError:
        return default
    return val if math.isfinite(val) and val > 0 else default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    try:
        val = int(raw) if raw is not None else default
    except ValueError:
        return default
    return val if val >= 1 else default


def _default_fit_workers() -> int:
    slurm = _env_int("SLURM_CPUS_PER_TASK", 0)
    return slurm if slurm >= 1 else max(1, (os.cpu_count() or 1) - 1)


#: petab_mcp.fitting's fit settings, resolved from the same environment variables the same way
#: and, like the server's, once: at import.
FIT_PER_START_TIMEOUT_S = _env_float("PETAB_MCP_FIT_PER_START_TIMEOUT", 180.0)
FIT_N_WORKERS = _env_int("PETAB_MCP_FIT_N_WORKERS", _default_fit_workers())
FIT_AMICI_ATOL = _env_float("PETAB_MCP_FIT_AMICI_ATOL", 1e-10)


def _entry_inflated(entry: dict) -> bool:
    # The effective sd where the entry records one (None there means "unknown", not inflated);
    # `fitted` only for an entry older than that field.
    eff = entry.get("effective_sigma") if "effective_sigma" in entry else entry.get("fitted")
    cap = entry.get("upper_bound_cap")
    try:
        return eff is not None and cap is not None and float(eff) > float(cap)
    except (TypeError, ValueError):
        return False


def _entry_collapsed(entry: dict) -> bool:
    frac = entry.get("fraction_of_data_std")
    return isinstance(frac, (int, float)) and frac < NOISE_SIGMA_COLLAPSE_FRACTION


def noise_sigma_inflated(report: "dict | None") -> bool:
    """petab_mcp.fitting.noise_sigma_inflated."""
    if not isinstance(report, dict):
        return False
    return any(_entry_inflated(e) for e in report.values() if isinstance(e, dict))


def noise_sigma_collapsed(report: "dict | None") -> bool:
    """petab_mcp.fitting.noise_sigma_collapsed."""
    if not isinstance(report, dict):
        return False
    return any(_entry_collapsed(e) for e in report.values() if isinstance(e, dict))


# ----------------------------------------------------------------------------- reconcile
_IDENT_RE = re.compile(r"(?<![\w.])[A-Za-z_]\w*")


def _formula_symbols(formula: Any) -> set[str]:
    if formula is None:
        return set()
    if not isinstance(formula, str):
        try:
            if not math.isfinite(float(formula)):
                return set()
        except (TypeError, ValueError):
            return set()
    return set(_IDENT_RE.findall(str(formula)))


def observable_scaling_ids(problem) -> set[str]:
    """petab_mcp.reconcile.observable_scaling_ids: every symbol of ``observableFormula`` and of
    the measurement table's ``observableParameters``."""
    names: set[str] = set()
    for frame, column in (
        (problem.observable_df, "observableFormula"),
        (problem.measurement_df, "observableParameters"),
    ):
        if frame is not None and column in frame.columns:
            for value in frame[column]:
                names |= _formula_symbols(value)
    return names


# -------------------------------------------------------------------------------- skills
#: Not the server's hash, which is over files this suite does not have: what a test reads off
#: it is that one was recorded, never its value.
SKILLS_VERSION = "fake-skills"


def constants() -> ModuleType:
    return _module(
        "constants",
        NOISE_SIGMA_COLLAPSE_FRACTION=NOISE_SIGMA_COLLAPSE_FRACTION,
        NOISE_SIGMA_REFERENCE_FACTOR=NOISE_SIGMA_REFERENCE_FACTOR,
        NOISE_SIGMA_BAND_ALPHA=NOISE_SIGMA_BAND_ALPHA,
        NOISE_SIGMA_BAND_MIN_DF=NOISE_SIGMA_BAND_MIN_DF,
    )


def criteria() -> ModuleType:
    return _module(
        "criteria", PENALTIES=PENALTIES, Criterion=Criterion, CANONICAL=CANONICAL, resolve=resolve
    )


def fitting() -> ModuleType:
    return _module(
        "fitting",
        FIT_PER_START_TIMEOUT_S=FIT_PER_START_TIMEOUT_S,
        FIT_N_WORKERS=FIT_N_WORKERS,
        FIT_AMICI_ATOL=FIT_AMICI_ATOL,
        noise_sigma_inflated=noise_sigma_inflated,
        noise_sigma_collapsed=noise_sigma_collapsed,
    )


def reconcile() -> ModuleType:
    return _module("reconcile", observable_scaling_ids=observable_scaling_ids)


def skills() -> ModuleType:
    return _module("skills", skills_version=lambda: SKILLS_VERSION, skill_texts=lambda: {})


def identifiability() -> ModuleType:
    return _module("identifiability")


def validation_data() -> ModuleType:
    return _module("validation_data")


#: The :mod:`gemot.server_api` accessors this module replaces, by name.
ACCESSORS = (
    "constants",
    "criteria",
    "fitting",
    "identifiability",
    "reconcile",
    "skills",
    "validation_data",
)


def install(set_attr: Callable[[Any, str, Any], None] = setattr) -> None:
    """Point every :mod:`gemot.server_api` accessor at its fake.

    ``tests/conftest.py`` passes ``monkeypatch.setattr`` so the swap is undone after each test; a
    test's SUBPROCESS, which the fixture cannot reach, calls this bare (with ``tests/`` on its
    path) to run the same fakes in a fresh interpreter.
    """
    from gemot import server_api

    for name in ACCESSORS:
        set_attr(server_api, name, globals()[name])
