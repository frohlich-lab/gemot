"""``paths.repo_root``: the harness is installed apart from the checkout it drives."""

from __future__ import annotations

import sys

from gemot import paths


def test_the_checkout_is_found_from_the_running_script_whatever_the_working_directory(
    tmp_path, monkeypatch
):
    """Nothing about the install says where the checkout is, so a run started from outside it
    (``python ~/checkout/examples/x/run.py``) is found from its script, or it would archive into
    the working directory."""
    checkout = tmp_path / "checkout"
    (checkout / "examples" / "x").mkdir(parents=True)
    (checkout / "pyproject.toml").write_text("")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.delenv(paths.REPO_ROOT_ENV, raising=False)
    monkeypatch.chdir(elsewhere)
    monkeypatch.setattr(sys, "argv", [str(checkout / "examples" / "x" / "run.py")])
    paths.repo_root.cache_clear()
    try:
        assert paths.repo_root() == checkout.resolve()
    finally:
        paths.repo_root.cache_clear()
