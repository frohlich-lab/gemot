"""The route's process boundary with the REAL server installed: the JAX worker is handed this
interpreter's ``gemot`` and ``petab_mcp`` and nothing else, and a real child process round-trips
an outcome. The route's clocks and ladders, with stand-in modules, are in
``tests/test_neural_ode_route.py``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


from gemot.neural_ode import route as R
from gemot.neural_ode import worker as W
from gemot.neural_ode.config import Config
from test_neural_ode_route import _write_problem


def test_the_parent_passes_an_earlier_clock_the_config_and_the_env_and_reads_proposals_back(
    monkeypatch, tmp_path, server_src
):
    captured = {}
    proposal = R.Proposal("model m\n  A' = -k*A;\n  k = 1;\nend", ("A",), "toy_node_selected", 1)

    def run(command, **kwargs):
        captured.update(command=command, timeout=kwargs["timeout"], env=kwargs["env"])
        Path(command[command.index("--out") + 1]).write_text(
            json.dumps(
                {
                    "ok": True,
                    "interpreter": "/jax/python",
                    "outcome": R.Outcome("toy", proposals=(proposal,)).as_dict(),
                }
            )
        )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    config = Config(
        interpreter=sys.executable, pysr_interpreter="/pysr/python", threads=4, dosed=["S"]
    )
    outcome = R._run_in_interpreter(
        tmp_path, "toy", config, work_dir=tmp_path, seconds=100.0, local_reason="x"
    )
    command, env = captured["command"], captured["env"]
    assert captured["timeout"] == 100.0 and float(command[command.index("--seconds") + 1]) == 90.0
    # The config crosses whole, the PySR interpreter and the dosing convention included.
    assert Config.load(command[command.index("--config") + 1]) == config
    assert env["OMP_NUM_THREADS"] == "4"
    # The worker's interpreter need not have gemot or petab_mcp: it gets this one's, nothing else.
    import gemot

    packages = Path(env["PYTHONPATH"].split(":")[0])
    assert sorted(p.name for p in packages.iterdir()) == ["gemot", "petab_mcp"]
    assert (packages / "gemot").resolve() == Path(gemot.__file__).resolve().parent
    assert (packages / "petab_mcp").resolve() == server_src.resolve()
    assert outcome.proposals[0].record == {
        "jax_interpreter": "/jax/python",
        "ran_out_of_process": True,
    }

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **_k: subprocess.CompletedProcess(command, 1, stdout="", stderr="boom"),
    )
    assert (
        "exited 1"
        in R._run_in_interpreter(
            tmp_path, "toy", config, work_dir=tmp_path / "w2", seconds=None, local_reason="x"
        ).refusal.reason
    )
    assert (
        R._run_in_interpreter(
            tmp_path, "toy", Config(), work_dir=tmp_path, seconds=None, local_reason="x"
        )
        is None
    )


def test_the_worker_never_delegates_again_and_the_real_round_trip_carries_an_outcome(
    monkeypatch, tmp_path
):
    seen = {}
    monkeypatch.setattr(
        R,
        "propose",
        lambda d, config, **kw: (
            seen.update(config=config, **kw) or R.Outcome("toy", refusal=R.Refusal("x"))
        ),
    )
    config_path = Config(interpreter="/some/python").dump(tmp_path / "c.json")
    assert (
        W.main(
            [
                "p",
                "--out",
                str(tmp_path / "o.json"),
                "--config",
                str(config_path),
                "--seconds",
                "90",
            ]
        )
        == 0
    )
    assert seen["config"].interpreter is None and seen["seconds"] == 90.0
    monkeypatch.undo()

    # A REAL subprocess: this interpreter plays the JAX one after the parent's stack is declared
    # absent. The child reads the toy problem, refuses it as not v1, and the boundary carries that.
    def absent():
        raise ImportError("no jax in the parent")

    monkeypatch.setattr(R, "_modules", absent)
    problem = _write_problem(tmp_path / "toy" / "problem", [0.0, 1.0, 2.0], version="2.0.0")
    outcome = R.propose(
        problem, Config(interpreter=sys.executable, seconds=300.0), work_dir=tmp_path / "work"
    )
    assert outcome.problem == "toy" and "not a PEtab v1 problem" in outcome.refusal.reason
    assert json.loads((tmp_path / "work" / "neural_proposals.json").read_text())["interpreter"]
