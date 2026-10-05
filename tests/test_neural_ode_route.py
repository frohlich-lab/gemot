"""The route with stand-in modules: its clocks, its ladders, its process boundary."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from gemot.neural_ode import route as R
from gemot.neural_ode.config import Config, ConfigError


@dataclass(frozen=True)
class _TrainConfig:
    n_augmented: int = 0
    width: int = 32
    depth: int = 2
    steps: int = 10
    shared_initial_conditions: bool = False
    multiplicative: bool = False

    def key(self) -> str:
        return f"w{self.width}_a{self.n_augmented}"


@dataclass(frozen=True)
class _DistilConfig:
    maxsize: int = 18
    n_samples: int = 100
    interpreter: str | None = None
    binary_operators: tuple = ("+", "*")

    def key(self) -> str:
        return f"pysr_max{self.maxsize}"


@dataclass
class _Candidate:
    complexity_budget: int
    antimony: str = "model c\n  A' = -k*A;\n  k = 1;\nend"
    expressions: tuple = ("-k*A",)
    n_constants: int = 1
    field_r2: tuple = (0.9,)


@dataclass
class _Distilled:
    config: _DistilConfig
    pysr_available: bool = True
    candidates: tuple = ()
    field_r2: tuple = (0.5,)
    n_constants: int = 2
    warnings: tuple = ()
    antimony: str = "model sel\n  A' = -k*A + c;\n  k = 1;\n  c = 0.1;\nend"
    expressions: tuple = ("-k*A + c",)
    state_names: tuple = ("A",)


def _result(config, *, nll=10.0, n_ode=2, weights=1000, failed=False):
    return SimpleNamespace(
        solver_failed=failed,
        nll=nll,
        n_ode_parameters=n_ode,
        n_network_weights=weights,
        steps_run=10,
        restart_failures=(),
        stage_objectives=(nll,),
        notes=(),
        config=config,
    )


class _Problem:
    name, warnings = "toy", ()

    def counts(self, *, shared_ics: bool) -> dict:
        return {
            "n_measurements": 100,
            "n_experiments": 1,
            "n_modelled_states": 1,
            "n_dosed": 0,
            "n_inputs": 0,
            "n_fixed_ics": 0,
            "n_sigma": 1,
        }


class _ProblemError(ValueError):
    pass


def _modules(train=None, distil=None, load=None):
    """Stand-ins for ``(problem, train, distil)``: a loadable problem, a fit, a distillation.

    ``trainer.saved`` records where the route asked for the fitted field to be kept."""

    def default_distil(_p, _r, config):
        return _Distilled(config=config, candidates=(_Candidate(config.maxsize - 6),))

    saved: list[Path] = []
    problem = SimpleNamespace(
        load=load or (lambda path, *, name, dosed=(): _Problem()), ProblemError=_ProblemError
    )
    trainer = SimpleNamespace(
        TrainConfig=_TrainConfig,
        train=train or (lambda _p, _d, c: _result(c)),
        save_bundle=lambda _r, directory: saved.append(Path(directory)),
        saved=saved,
    )
    distiller = SimpleNamespace(DistilConfig=_DistilConfig, distil=distil or default_distil)
    return problem, trainer, distiller


def _write_problem(directory: Path, times, *, version="1") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "measurements.tsv").write_text(
        "observableId\ttime\tmeasurement\n" + "".join(f"obs\t{t}\t1.0\n" for t in times)
    )
    (directory / "problem.yaml").write_text(
        "format_version: 1\nproblems:\n- measurement_files: [measurements.tsv]\n"
        if version.startswith("1")
        else f"format_version: {version}\nmeasurement_files: [measurements.tsv]\n"
    )
    return directory


def test_config_applies_overrides_and_refuses_what_antimony_could_not_express(tmp_path):
    _p, T, D = _modules()
    assert Config().train_config(T) == _TrainConfig()
    assert Config(train={"steps": 5}).train_config(T).steps == 5
    assert Config().train_config(T, _TrainConfig(width=4)).width == 4  # supplied kept
    # The distiller learns which Python holds PySR from the route's config, nothing else.
    assert Config().distil_config(D).interpreter is None
    assert Config(pysr_interpreter="/pysr/python", distil={"maxsize": 30}).distil_config(
        D
    ) == _DistilConfig(maxsize=30, interpreter="/pysr/python")
    assert Config(distil={"binary_operators": ["+", "hill"]}).distil_config(D).binary_operators == (
        "+",
        "hill",
    )
    for bad in (
        {"binary_operators": ["+", "pow"]},
        {"selection": "random"},
        {"parsimony": -1},
        {"workers": 0},
    ):
        with pytest.raises(ConfigError):
            Config(distil=bad)
    with pytest.raises(ConfigError):
        Config.from_dict({"interpretter": "/x"})
    config = Config(interpreter="/x/python", seconds=100.0, dosed=["S"], train={"steps": 5})
    assert config.dosed == ("S",) and Config.load(config.dump(tmp_path / "c.json")) == config


def test_the_two_clocks():
    assert R._worker_seconds(8400) == 7800 and R._worker_seconds(100) == 90
    assert [R._distil_reserve(s) for s in (None, 100, 8400, 100_000)] == [0, 50, 1260, 1800]


def test_refusals_at_entry(monkeypatch, tmp_path):
    settled = _write_problem(tmp_path / "settled", ["inf", "inf"])
    assert "steady state" in R.refuse_settled(settled / "problem.yaml")
    assert R.refuse_settled(_write_problem(tmp_path / "v1", [0.0, 1.0]) / "problem.yaml") is None
    assert "no problem.yaml" in R.propose(tmp_path / "missing").refusal.reason

    def refuse(path, *, name, dosed=()):
        raise _ProblemError(f"{path} is not a PEtab v1 problem")

    monkeypatch.setattr(R, "_modules", lambda: _modules(load=refuse))
    v2 = R.propose(_write_problem(tmp_path / "p" / "problem", [0.0, 1.0], version="2.0.0"))
    assert v2.problem == "p" and "not a PEtab v1 problem" in v2.refusal.reason

    def absent():
        raise ImportError("No module named 'jax'")

    monkeypatch.setattr(R, "_modules", absent)
    assert "not importable" in R.propose(_write_problem(tmp_path / "q", [0.0, 1.0])).refusal.reason
    with pytest.raises(R.RouteError):
        R.Outcome("p")


def test_the_route_assembles_proposals_and_they_survive_the_json_boundary(monkeypatch, tmp_path):
    monkeypatch.setattr(R, "_modules", lambda: _modules())
    problem = _write_problem(tmp_path / "toy" / "problem", [0.0, 1.0, 2.0])
    outcome = R.propose(problem, Config(distil={"maxsize": 24}), work_dir=tmp_path / "w")
    assert [p.model_name for p in outcome.proposals] == ["toy_node_selected", "toy_node_c18"]
    record = outcome.proposals[0].record
    assert record["distil_config"] == "pysr_max24" and record["tuning"] is None
    assert record["n_ode_parameters"] == 2 and record["stage_objectives"] == [10.0]
    assert R.Outcome.from_dict(json.loads(json.dumps(outcome.as_dict()))) == outcome
    # `tuning` is always None now: the route takes the architecture it is given. The ladders that
    # used to fill it chose by criteria that measurement refuted, and are gone with the module.
    assert (
        R.propose(problem, Config(seconds=1000.0), work_dir=tmp_path / "w2")
        .proposals[0]
        .record["tuning"]
        is None
    )


def test_the_command_line_writes_the_outcome_and_exits_one_on_a_refusal(monkeypatch, tmp_path):
    from gemot.neural_ode import __main__ as cli

    problem = _write_problem(tmp_path / "toy" / "problem", [0.0, 1.0])
    seen = {}
    monkeypatch.setattr(
        R,
        "propose",
        lambda d, config, **kw: (
            seen.update(config=config) or R.Outcome("toy", refusal=R.Refusal("no trainer"))
        ),
    )
    assert (
        cli.main(
            [
                "propose",
                str(problem),
                "--out",
                str(tmp_path / "out"),
                "--seconds",
                "12",
                "--dosed",
                "S",
            ]
        )
        == 1
    )
    assert seen["config"].seconds == 12.0
    assert seen["config"].dosed == ("S",)
    assert (
        json.loads((tmp_path / "out" / "proposals.json").read_text())["refusal"]["reason"]
        == "no trainer"
    )


def test_the_route_keeps_the_fitted_field_beside_the_staged_problem(monkeypatch, tmp_path):
    """Staging writes the network's INITIAL weights and nothing writes them again, so without this
    the fitted field dies with the worker: only its distillations and the record's scalars survive,
    and re-distilling a field at other settings, or re-scoring it, means retraining it."""
    problem, trainer, distiller = _modules()
    monkeypatch.setattr(R, "_modules", lambda: (problem, trainer, distiller))
    directory = _write_problem(tmp_path / "toy", [0.0, 1.0, 2.0])
    outcome = R.propose(directory, Config(), work_dir=tmp_path / "w")
    assert outcome.proposals and trainer.saved, "a proposal without a kept field is the bug"

    destination, key = trainer.saved[0], _TrainConfig().key()
    assert destination.name == key and destination.parent.name == "field"
    # BESIDE the staged directory, never inside it: the bundle is a copy of that directory, so a
    # destination within it would copy into itself.
    staged = destination.parent.parent / key
    assert staged not in destination.parents and destination != staged
