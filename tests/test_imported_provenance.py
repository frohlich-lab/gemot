"""A run records the code it IMPORTS, not the code pip installed.

Runs that put a source checkout of the server and of gemot ahead of their installs (``PYTHONPATH``)
recorded the installed commits, 848722e and 329210f, while running others: the install metadata does
not know what the path overrides. ``_imported_provenance`` finds the module where Python will import
it from and reads that checkout.
"""

from __future__ import annotations

import subprocess
import sys

from gemot.runner import provenance as P


def _repo_with_package(tmp_path, name):
    repo = tmp_path / "repo"
    (repo / name).mkdir(parents=True)
    (repo / name / "__init__.py").write_text("")
    for cmd in (["init", "-q"], ["add", "."], ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "x"]):
        subprocess.run(["git", *cmd], cwd=repo, check=True)
    sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=repo).decode().strip()
    return repo, sha


def test_a_checkout_on_the_path_is_recorded_as_itself(tmp_path, monkeypatch):
    repo, sha = _repo_with_package(tmp_path, "pkg_under_test")
    monkeypatch.syspath_prepend(str(repo))
    sys.modules.pop("pkg_under_test", None)
    got = P._imported_provenance("pkg_under_test", "no-such-distribution")
    assert got["git_sha"] == sha and got["git_diff"] is None
    assert got["path"] == str((repo / "pkg_under_test").resolve())


def test_an_uncommitted_change_is_fingerprinted(tmp_path, monkeypatch):
    repo, sha = _repo_with_package(tmp_path, "pkg_dirty")
    (repo / "pkg_dirty" / "__init__.py").write_text("x = 1\n")
    monkeypatch.syspath_prepend(str(repo))
    got = P._imported_provenance("pkg_dirty", "no-such-distribution")
    assert got["git_sha"] == f"{sha}+dirty" and got["git_diff"]["files"] == ["pkg_dirty/__init__.py"]


def test_a_module_outside_any_checkout_falls_back_to_the_install():
    assert P._imported_provenance("no_such_module_xyz", "no-such-distribution") is None
    assert P._imported_provenance("json", "no-such-distribution") is None  # stdlib: not a checkout


def test_the_dependencies_that_compute_the_numbers_are_listed():
    assert {m for m, _ in P._DEPENDENCIES} == {"petab", "pypesto", "amici", "fides", "indra", "pysb"}
