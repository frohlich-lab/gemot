"""The training-ΔBIC trajectory figures: what a run's curve is, and that both render at the floor.

A panel's line is the median, and its band the range, of each run's BEST-SO-FAR ΔBIC -- and a run
can deliver a version other than its best, so its delivered ΔBIC is drawn separately, where the run
ended. These pin the arithmetic under that and render both figures from a synthetic payload,
where the type floor is checked on save.
"""
import numpy as np

from petab_harness import paper_bic_trajectories as trajectories


def _run(points, head, final):
    return {"run": "0" * 8, "label": "x", "points": points, "head_version": head,
            "final_dbic": final}


def test_best_so_far_carries_the_best_forward_and_starts_at_the_first_fit():
    points = [[1, 50.0], [2, 80.0], [4, -3.0]]
    assert trajectories.best_so_far(points, 5) == [None, 50.0, 50.0, 50.0, -3.0, -3.0]


def test_a_run_ends_on_its_head_when_the_head_came_after_its_last_fit():
    assert trajectories.last_version(_run([[0, 1.0], [3, 2.0]], 5, None)) == 5
    assert trajectories.last_version(_run([[0, 1.0], [3, 2.0]], None, None)) == 3


def test_the_envelope_is_median_and_range_carried_to_the_longest_run():
    runs = [_run([[0, 10.0], [1, 0.0]], 1, 0.0), _run([[0, 30.0], [3, -20.0]], 3, -20.0)]
    versions, median, low, high = trajectories.envelope(runs)
    assert list(versions) == [0, 1, 2, 3]
    assert list(median) == [20.0, 15.0, 15.0, -10.0]
    assert list(low) == [10.0, 0.0, 0.0, -20.0]
    assert list(high) == [30.0, 30.0, 30.0, 0.0]


def test_the_limits_follow_what_is_drawn_not_every_fitted_version():
    """A run's worst version never reaches its best-so-far curve, so it must not set the scale."""
    run = _run([[0, 100.0], [1, 5e6], [2, -40.0]], 2, -40.0)
    low, high = trajectories._limits(trajectories._values([run]))
    assert high < 1e3 and low < -40.0


def test_tick_labels_are_compact_signed_and_free_of_superscripts():
    assert [trajectories.compact(v) for v in (0, 10, -100, 1000, -1e6)] == [
        "0", "10", "−100", "1k", "−1M"]


def _payload():
    visible = [_run([[0, 400.0], [2, 20.0], [5, -12.0]], 5, -12.0),
               _run([[0, 90.0], [3, -30.0]], 1, 60.0)]
    blinded = [_run([[0, 900.0], [4, -8.0]], 4, -8.0)]
    history = {"support": 1, "model": "z-ai/glm-5.3-flash", "runs": visible, "no_model": []}
    return {
        "schema_version": 1,
        "support_levels": {"1": "curator + biology + gemot", "2": "biology + gemot",
                           "3": "biology + deepagents", "4": "gemot"},
        "figure2": {"crauste": visible, "perelson": blinded},
        "figure3": {
            "baseline": {**history, "code": "C1"},
            "null": {"code": "C3", "support": 3, "model": "z-ai/glm-5.3-flash",
                     "runs": [_run([], -1, 216.0), _run([], -1, -493.0)], "no_model": ["G1"]},
            "astra": {**history, "code": "F2", "support": 2, "model": "openai/gpt-6-astra"},
        },
        "figure4_blinded": {"crauste": blinded},
    }


def test_both_figures_render_at_the_type_floor(tmp_path):
    payload = _payload()
    for name, render in (("figure2", trajectories.render_figure2),
                         ("figures34", trajectories.render_figure34)):
        path = tmp_path / f"{name}.png"
        render(payload, path)   # _save runs paper_style.check_fonts, which raises under 6 pt
        assert path.stat().st_size > 0
    assert np.isfinite(trajectories.envelope(payload["figure2"]["crauste"])[1]).all()
