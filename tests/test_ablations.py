"""The ablation registry is only authoritative if nothing can bypass it.

Two classes of test. The ones here pin the registry's own behaviour (resolution, `arm()`, the
printed catalogue). The drift guard, in ``tests/server/test_ablations.py`` because the server
reads switches too, greps both packages for environment variables that gate behaviour and fails
if one is neither registered nor explicitly declared non-ablation. Without it, "the single
entrypoint" holds only for as long as everyone remembers to update the registry.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from gemot import ablations


SRC = Path(__file__).resolve().parents[1] / "src"


# --------------------------------------------------------------------------- the registry itself
def test_every_ablation_defaults_to_the_baseline_arm(monkeypatch):
    """With no ablation variable set, every switch reads its declared default and arm() is clean."""
    for a in ablations.ABLATIONS:
        monkeypatch.delenv(a.env, raising=False)
    assert ablations.state() == {a.name: a.default_on for a in ablations.ABLATIONS}
    assert ablations.arm() == "baseline"


@pytest.mark.parametrize("spelling", ["0", "false", "FALSE", "no", "off", " Off "])
def test_off_words_turn_a_default_on_switch_off(monkeypatch, spelling):
    monkeypatch.setenv("PETAB_DATA_REGIMES", spelling)
    assert ablations.enabled("data_regimes") is False


@pytest.mark.parametrize("spelling", ["1", "true", "yes", "on", "anything"])
def test_non_off_words_leave_a_default_on_switch_on(monkeypatch, spelling):
    monkeypatch.setenv("PETAB_DATA_REGIMES", spelling)
    assert ablations.enabled("data_regimes") is True


def test_the_printed_setting_actually_flips_each_arm(monkeypatch):
    """Execute the catalogue's own instructions.

    `describe()` tells the reader to export `a.setting`, and nothing else checks that the string
    works. It is easy to get wrong, because the value that flips an arm is not uniform: a switch
    whose variable names a PATH is disabled by the empty string, and `=0` would set it to a
    directory called "0" while leaving it on. A catalogue that documents a no-op is worse than
    none, so: parse each printed setting, apply it, and assert the arm moved off its default.
    """
    for a in ablations.ABLATIONS:
        var, _, value = a.setting.partition("=")
        assert var == a.env, f"{a.name}: setting names {var}, not its env var {a.env}"
        monkeypatch.setenv(var, value)
        assert a.enabled() is (not a.default_on), (
            f"{a.name}: the documented setting `{a.setting}` did not flip it"
        )
        assert not a.is_default()
        assert f"{a.name}=" in ablations.arm()
        monkeypatch.delenv(var)


def test_arm_names_only_the_deviations(monkeypatch):
    """A new ablation left at its default must not rename existing arms, or old runs stop being
    comparable to new ones the moment a switch is added."""
    monkeypatch.delenv("PETAB_BIO_LITERATURE", raising=False)
    monkeypatch.delenv("PETAB_DATA_REGIMES", raising=False)
    assert ablations.arm() == "baseline"  # both at their defaults
    monkeypatch.setenv("PETAB_DATA_REGIMES", "1")
    assert ablations.arm() == "data_regimes=on"
    monkeypatch.setenv("PETAB_BIO_LITERATURE", "1")
    assert ablations.arm() == "data_regimes=on,bio_literature=on"


def test_unknown_ablation_raises_rather_than_reading_as_off():
    """A typo must not silently select the baseline — that is an A/B whose halves are identical."""
    with pytest.raises(KeyError, match="unknown ablation"):
        ablations.enabled("data_regime")  # missing 's'


def test_state_records_every_switch_not_only_the_deviations(monkeypatch):
    """A run silent about a switch is indistinguishable from one predating it."""
    monkeypatch.delenv("PETAB_DATA_REGIMES", raising=False)
    monkeypatch.delenv("PETAB_BIO_LITERATURE", raising=False)
    assert set(ablations.state()) == {a.name for a in ablations.ABLATIONS}


def test_cli_lists_every_arm_and_json_round_trips(capsys):
    assert ablations._main([]) == 0
    text = capsys.readouterr().out
    for a in ablations.ABLATIONS:
        assert a.name in text and a.setting in text and a.effect[:40] in text

    assert ablations._main(["--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["arm"] == ablations.arm()
    assert [x["name"] for x in payload["ablations"]] == [a.name for a in ablations.ABLATIONS]


def test_module_runs_as_a_script():
    """`python -m gemot.ablations` is the advertised entrypoint, so run it as one."""
    out = subprocess.run(
        [sys.executable, "-m", "gemot.ablations"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Ablations" in out.stdout and "data_regimes" in out.stdout


# --------------------------------------------------------------------------- the drift guard
def test_the_removed_bio_literature_flag_stays_removed():
    """`--bio-literature` was an arm settable only from the command line and recorded nowhere.

    It is now `PETAB_BIO_LITERATURE`. Re-adding the flag would restore two ways to set one arm,
    which is how it gets recorded from the wrong one.
    """
    assert (
        'add_argument("--bio-literature"' not in (SRC / "gemot" / "runner" / "cli.py").read_text()
    )
