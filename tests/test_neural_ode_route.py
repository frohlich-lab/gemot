"""The route with stand-in modules: its clocks, its ladders, its process boundary."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from petab_harness.neural_ode import route as R
from petab_harness.neural_ode import worker as W
from petab_harness.neural_ode.config import Config, ConfigError


@dataclass(frozen=True)
class _TrainConfig:
    n_augmented: int = 0
    width: int = 32
    depth: int = 2
    steps: int = 10
    shared_initial_conditions: bool = False

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
    return SimpleNamespace(solver_failed=failed, nll=nll, n_ode_parameters=n_ode, n_network_weights=weights,
                           steps_run=10, restart_failures=(), stage_objectives=(nll,), notes=(), config=config)


class _Problem:
    name, warnings = "toy", ()

    def counts(self, *, shared_ics: bool) -> dict:
        return {"n_measurements": 100, "n_experiments": 1, "n_modelled_states": 1, "n_dosed": 0,
                "n_inputs": 0, "n_fixed_ics": 0, "n_sigma": 1}


class _ProblemError(ValueError):
    pass


def _modules(train=None, distil=None, load=None):
    """Stand-ins for ``(problem, train, distil)``: a loadable problem, a fit, a distillation."""

    def default_distil(_p, _r, config):
        return _Distilled(config=config, candidates=(_Candidate(config.maxsize - 6),))

    problem = SimpleNamespace(load=load or (lambda path, *, name, dosed=(): _Problem()), ProblemError=_ProblemError)
    trainer = SimpleNamespace(TrainConfig=_TrainConfig, train=train or (lambda _p, _d, c: _result(c)))
    distiller = SimpleNamespace(DistilConfig=_DistilConfig, distil=distil or default_distil)
    return problem, trainer, distiller


def _write_problem(directory: Path, times, *, version="1") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "measurements.tsv").write_text(
        "observableId\ttime\tmeasurement\n" + "".join(f"obs\t{t}\t1.0\n" for t in times))
    (directory / "problem.yaml").write_text(
        "format_version: 1\nproblems:\n- measurement_files: [measurements.tsv]\n"
        if version.startswith("1") else
        f"format_version: {version}\nmeasurement_files: [measurements.tsv]\n")
    return directory


def test_config_applies_overrides_and_refuses_what_antimony_could_not_express(tmp_path):
    _p, T, D = _modules()
    assert Config().train_config(T) == _TrainConfig()
    assert Config(train={"steps": 5}).train_config(T).steps == 5
    assert Config().train_config(T, _TrainConfig(width=4)).width == 4     # supplied kept
    # The distiller learns which Python holds PySR from the route's config, nothing else.
    assert Config().distil_config(D).interpreter is None
    assert Config(pysr_interpreter="/pysr/python", distil={"maxsize": 30}).distil_config(D) == _DistilConfig(
        maxsize=30, interpreter="/pysr/python")
    assert Config(distil={"binary_operators": ["+", "hill"]}).distil_config(D).binary_operators == ("+", "hill")
    for bad in ({"binary_operators": ["+", "pow"]}, {"selection": "random"}, {"parsimony": -1}, {"workers": 0}):
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
    # used to fill it chose by criteria that measurement refuted (see `neural_ode.tune`).
    assert R.propose(problem, Config(seconds=1000.0),
                     work_dir=tmp_path / "w2").proposals[0].record["tuning"] is None


def test_the_parent_passes_an_earlier_clock_the_config_and_the_env_and_reads_proposals_back(
        monkeypatch, tmp_path):
    captured = {}
    proposal = R.Proposal("model m\n  A' = -k*A;\n  k = 1;\nend", ("A",), "toy_node_selected", 1)

    def run(command, **kwargs):
        captured.update(command=command, timeout=kwargs["timeout"], env=kwargs["env"])
        Path(command[command.index("--out") + 1]).write_text(json.dumps(
            {"ok": True, "interpreter": "/jax/python",
             "outcome": R.Outcome("toy", proposals=(proposal,)).as_dict()}))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    config = Config(interpreter=sys.executable, pysr_interpreter="/pysr/python", threads=4, dosed=["S"])
    outcome = R._run_in_interpreter(tmp_path, "toy", config, work_dir=tmp_path, seconds=100.0,
                                    local_reason="x")
    command, env = captured["command"], captured["env"]
    assert captured["timeout"] == 100.0 and float(command[command.index("--seconds") + 1]) == 90.0
    # The config crosses whole, the PySR interpreter and the dosing convention included.
    assert Config.load(command[command.index("--config") + 1]) == config
    assert env["OMP_NUM_THREADS"] == "4" and env["PYTHONPATH"].split(":")[0].endswith("src")
    assert outcome.proposals[0].record == {"jax_interpreter": "/jax/python", "ran_out_of_process": True}

    monkeypatch.setattr(subprocess, "run", lambda command, **_k: subprocess.CompletedProcess(
        command, 1, stdout="", stderr="boom"))
    assert "exited 1" in R._run_in_interpreter(tmp_path, "toy", config, work_dir=tmp_path / "w2",
                                               seconds=None, local_reason="x").refusal.reason
    assert R._run_in_interpreter(tmp_path, "toy", Config(), work_dir=tmp_path, seconds=None,
                                 local_reason="x") is None


def test_the_worker_never_delegates_again_and_the_real_round_trip_carries_an_outcome(
        monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(R, "propose", lambda d, config, **kw: (seen.update(config=config, **kw)
                                                               or R.Outcome("toy", refusal=R.Refusal("x"))))
    config_path = Config(interpreter="/some/python").dump(tmp_path / "c.json")
    assert W.main(["p", "--out", str(tmp_path / "o.json"), "--config", str(config_path),
                   "--seconds", "90"]) == 0
    assert seen["config"].interpreter is None and seen["seconds"] == 90.0
    monkeypatch.undo()
    # A REAL subprocess: this interpreter plays the JAX one after the parent's stack is declared
    # absent. The child reads the toy problem, refuses it as not v1, and the boundary carries that.
    def absent():
        raise ImportError("no jax in the parent")
    monkeypatch.setattr(R, "_modules", absent)
    problem = _write_problem(tmp_path / "toy" / "problem", [0.0, 1.0, 2.0], version="2.0.0")
    outcome = R.propose(problem, Config(interpreter=sys.executable, seconds=300.0), work_dir=tmp_path / "work")
    assert outcome.problem == "toy" and "not a PEtab v1 problem" in outcome.refusal.reason
    assert json.loads((tmp_path / "work" / "neural_proposals.json").read_text())["interpreter"]


def test_the_command_line_writes_the_outcome_and_exits_one_on_a_refusal(monkeypatch, tmp_path):
    from petab_harness.neural_ode import __main__ as cli

    problem = _write_problem(tmp_path / "toy" / "problem", [0.0, 1.0])
    seen = {}
    monkeypatch.setattr(R, "propose", lambda d, config, **kw: (
        seen.update(config=config) or R.Outcome("toy", refusal=R.Refusal("no trainer"))))
    assert cli.main(["propose", str(problem), "--out", str(tmp_path / "out"), "--seconds", "12",
                     "--dosed", "S"]) == 1
    assert seen["config"].seconds == 12.0
    assert seen["config"].dosed == ("S",)
    assert json.loads((tmp_path / "out" / "proposals.json").read_text())["refusal"]["reason"] == "no trainer"
