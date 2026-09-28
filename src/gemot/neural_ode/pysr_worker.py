"""The PySR regression, in a module that imports neither JAX nor the rest of this package, run as
a child process by file path.

    python pysr_worker.py <input.npz> <settings.json> <output.json>

A child, not a call: XLA and Julia's LLVM JIT do not share one address space well. The trainer
this replaces aborted on the cluster with ``LLVM compilation error: Cannot allocate memory`` at
6 GB of a 16 GB allocation, because ``vm.max_map_count`` was full of XLA's per-executable
mappings by the time Julia wanted its own. A child starts with an empty table, a Julia abort
costs the distillation and nothing else, and the child may be a different interpreter -- on the
cluster no one environment holds both stacks, and :attr:`DistilConfig.interpreter` names the one
that holds PySR. Run by file path rather than ``-m``, so the package's ``__init__`` and its
imports never load here.

One search per output column, several at a time when asked: the searches share nothing, so
states are the axis worth spending cores on, and each child search is then serial so the pool
cannot oversubscribe the allocation. Expressions cross back as text over positional placeholders
``x0, x1, ...``, chosen so that no problem's own vocabulary (``S``, ``beta``, ``E``) can collide
with sympy's.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import sympy as sp


def regress(
    features: np.ndarray, derivatives: np.ndarray, placeholders: list[str], settings: dict
) -> list[dict]:
    """One PySR run per output column: ``[{selected, r2, front}]`` with expressions as strings and
    ``front`` as ``[[complexity, expression, r2], ...]`` cheapest first."""
    workers = int(settings.get("workers") or 1)
    n_columns = int(derivatives.shape[1])
    if workers > 1 and n_columns > 1:
        # Before PySR is imported: importing it starts Julia, and a forked Julia is not fork-safe.
        return _regress_parallel(
            features, derivatives, placeholders, settings, min(workers, n_columns)
        )
    regressor = pysr_class()
    if regressor is None:
        raise ImportError(
            "PySR is not installed here (it needs a Julia toolchain); the field was sampled "
            "but not regressed. `pip install pysr` in the interpreter the distillation names."
        )
    out: list[dict] = []
    for column in range(n_columns):
        model = regressor(
            niterations=int(settings["niterations"]),
            population_size=int(settings["population_size"]),
            maxsize=int(settings["maxsize"]),
            parsimony=float(settings["parsimony"]),
            binary_operators=list(settings["binary_operators"]),
            unary_operators=list(settings["unary_operators"]),
            extra_sympy_mappings=_sympy_mappings(),
            random_state=int(settings["seed"]),
            # Not deterministic: PySR's deterministic mode forces a serial search, and reproducibility
            # comes from replication over seeds instead.
            deterministic=False,
            parallelism=settings.get("parallelism", "multithreading"),
            procs=int(settings.get("procs") or 0),
            progress=False,
            verbosity=0,
            # PySR's hall-of-fame files go next to the inputs, not into the working directory.
            output_directory=settings.get("output_directory"),
        )
        target = derivatives[:, column]
        model.fit(features, target, variable_names=list(placeholders))
        out.append(
            {
                "selected": str(selected_expression(model, settings.get("selection", "best"))),
                "r2": r_squared(model.predict(features), target),
                "front": [
                    [complexity, str(expression), value]
                    for complexity, expression, value in pareto_front(model, features, target)
                ],
            }
        )
    return out


def _regress_parallel(features, derivatives, placeholders, settings, workers: int) -> list[dict]:
    from concurrent.futures import ProcessPoolExecutor  # noqa: PLC0415

    serial = {**settings, "workers": 1, "parallelism": "serial", "procs": 0}
    payloads = [
        (features, derivatives[:, i : i + 1], placeholders, serial)
        for i in range(derivatives.shape[1])
    ]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return [result[0] for result in pool.map(_one, payloads)]  # map keeps column order


def _one(payload) -> list[dict]:
    return regress(*payload)


def pysr_class():
    """``PySRRegressor``, or ``None`` where PySR or Julia is unavailable; a function so this module
    imports anywhere and a test can substitute a stand-in."""
    try:
        from pysr import PySRRegressor  # noqa: PLC0415
    except Exception:  # noqa: BLE001 -- juliacall failures are not all ImportError
        return None
    return PySRRegressor


def _sympy_mappings() -> dict:
    """Sympy images of the custom operators, inlined into arithmetic so Antimony can read them;
    mirrors ``HILL_OPERATOR`` in :mod:`.distil`."""
    # The Julia side adds `eps(x)` -- machine epsilon at x -- which sympy has no term for. The two
    # agree to ~1e-16 everywhere except at k == x == 0, where Julia gives 0 and this gives 0/0, so
    # an emitted model could be undefined at a point the fitted one was not. `hill` is off by
    # default and no run uses it; fix this before turning it on.
    return {"hill": lambda x, k: x / (k**2 + sp.Abs(x))}


def selected_expression(model, selection: str) -> sp.Expr:
    """PySR's own accuracy/complexity pick (``best``), or the lowest-loss row (``accuracy``)."""
    if selection == "accuracy":
        frame = model.equations_[-1] if isinstance(model.equations_, list) else model.equations_
        expression = model.sympy(int(frame["loss"].idxmin()))
    else:
        expression = model.sympy()
    return sp.sympify(expression[0] if isinstance(expression, list) else expression)


def pareto_front(model, features: np.ndarray, target: np.ndarray) -> list[tuple]:
    """``[(complexity, expression, r2)]`` off one run, cheapest first, every row re-scored against
    the field rather than read off PySR's own loss, which carries its parsimony term."""
    equations = getattr(model, "equations_", None)
    if equations is None:
        return []
    frame = equations[-1] if isinstance(equations, list) else equations
    front = []
    for index, row in frame.iterrows():
        try:
            expression = sp.sympify(model.sympy(int(index)))
            value = r_squared(model.predict(features, index=int(index)), target)
        except Exception:  # noqa: BLE001 -- one unparseable row must not lose the front
            continue
        front.append((int(row["complexity"]), expression, value))
    return sorted(front, key=lambda item: item[0])


def r_squared(prediction: np.ndarray, target: np.ndarray) -> float:
    residual = float(np.sum((np.asarray(prediction) - target) ** 2))
    total = float(np.sum((target - target.mean()) ** 2))
    return 1.0 - residual / total if total > 0 else float("nan")


def main(argv: list[str]) -> int:
    arrays = np.load(argv[1])
    with open(argv[2]) as handle:
        settings = json.load(handle)
    result = regress(
        arrays["features"], arrays["derivatives"], list(settings["placeholders"]), settings
    )
    with open(argv[3], "w") as handle:
        json.dump(result, handle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
