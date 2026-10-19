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


def test_the_roster_categories_are_disjoint_and_all_examples_is_their_union_in_order():
    """Each example is exactly one kind — synthetic, collection-based, or a model-free data package —
    because the kind decides what may be done to it: only collection examples are materialized and
    refit (setup.py and refit_references.sh walk BENCHMARK_EXAMPLES), and only data packages are
    judged reference-free. An example in two lists would be refit against a model it does not have."""
    groups = (paths.SYNTHETIC_EXAMPLES, paths.BENCHMARK_EXAMPLES, paths.DATA_PACKAGE_EXAMPLES)
    assert paths.ALL_EXAMPLES == [name for group in groups for name in group]
    assert len(set(paths.ALL_EXAMPLES)) == len(paths.ALL_EXAMPLES)
    assert paths.DATA_PACKAGE_EXAMPLES == ["dessauges", "fontaine", "lun", "thomas"]


def test_large_examples_are_still_the_five_large_collection_networks():
    """LARGE_EXAMPLES is the TAIL of BENCHMARK_EXAMPLES, so a new kind of example appended there
    instead of to its own list would silently replace raimundez and friends as the examples given
    the bigger per-task budget."""
    assert paths.LARGE_EXAMPLES == frozenset(
        {"isensee", "alkan", "bachmann", "lucarelli", "raimundez"}
    )
    assert not paths.LARGE_EXAMPLES & set(paths.DATA_PACKAGE_EXAMPLES)
