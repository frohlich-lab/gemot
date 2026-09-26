"""Training ΔBIC per model version: how far, and how fast, each benchmark's runs converge.

Two supplementary figures from one frozen input (``paper/bic_trajectories.json``, built by
``slurm/_bic_trajectories_input.py``):

* :func:`render_figure2` -- the Figure 2 runs, one panel per benchmark, in Figure 2's order.
* :func:`render_figure34` -- figure 3's Fujita arms as a model x rung grid, then figure 4's
  benchmarks with the biology-visible and biology-blinded runs overlaid.

Each run contributes its BEST-SO-FAR training ΔBIC at every version: the best model it had fitted
by then, carried forward once the run ends. A panel draws the median over its runs as a line and
the min-max range as a band. Best-so-far is not always what a run delivered -- a run can end on a
version other than its best fit -- so each run's delivered ΔBIC, the value its paper figure
plots, is marked as well, at the version the run ended on. The y axis is symmetric-log, linear
within ±10: the first versions of a run sit hundreds to thousands above the reference, and the
part that matters is near zero.
"""
from __future__ import annotations

import json
from pathlib import Path

from petab_harness.paper_fujita import MODEL_MARKERS, SUPPORT_RAMPS, shaded

VISIBLE, BLINDED = SUPPORT_RAMPS[0][0], SUPPORT_RAMPS[0][3]
BETTER_FILL, BETTER_INK, ZERO_INK, MUTED = "#EEF6F2", "#3E7863", "#526370", "#5F6B74"
TITLE, BODY, SMALL = 7.0, 6.5, 6.0
LINTHRESH, LINSCALE = 10, 0.6
X_LABEL, Y_LABEL = "Iteration (model version)", "Training ΔBIC, best so far"


def read_input(root: Path) -> dict | None:
    path = Path(root) / "paper" / "bic_trajectories.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported BIC trajectory schema; rebuild with "
                         "slurm/_bic_trajectories_input.py")
    return payload


def best_so_far(points: list[list[float]], until: int) -> list[float | None]:
    """The run's best ΔBIC at each version 0..until; None before its first fitted version."""
    by_version = dict((int(v), d) for v, d in points)
    out, best = [], None
    for version in range(until + 1):
        if version in by_version:
            best = by_version[version] if best is None else min(best, by_version[version])
        out.append(best)
    return out


def last_version(run: dict) -> int:
    """The version the run ended on: its last fitted one, or its head if that came later."""
    return max([int(v) for v, _ in run["points"]] + [run.get("head_version") or 0])


def envelope(runs: list[dict]):
    """``(versions, median, low, high)`` over the runs, each carried to the longest run."""
    import numpy as np

    until = max(last_version(run) for run in runs)
    curves = np.array([[np.nan if y is None else y for y in best_so_far(run["points"], until)]
                       for run in runs], dtype=float)
    versions = np.arange(until + 1)
    with np.errstate(all="ignore"):
        return (versions, np.nanmedian(curves, axis=0), np.nanmin(curves, axis=0),
                np.nanmax(curves, axis=0))


def draw(axis, runs: list[dict], color: str, marker: str = "o", zorder: int = 3) -> None:
    """Median line and range band of the runs' best-so-far, and where each run ended."""
    import numpy as np

    runs = [run for run in runs if run["points"]]
    if not runs:
        return
    versions, median, low, high = envelope(runs)
    keep = ~np.isnan(median)
    axis.fill_between(versions[keep], low[keep], high[keep], color=color, alpha=0.22, lw=0,
                      step="post", zorder=zorder)
    axis.step(versions[keep], median[keep], where="post", color=shaded(color, 0.85), lw=1.0,
              zorder=zorder + 1)
    delivered = [(last_version(run), run["final_dbic"]) for run in runs
                 if run.get("final_dbic") is not None]
    if delivered:
        xs, ys = zip(*delivered)
        axis.scatter(xs, ys, s=7, marker=marker, facecolors="white",
                     edgecolors=shaded(color, 0.85), linewidths=0.6, zorder=zorder + 2,
                     clip_on=False)


def delivered_only(axis, runs: list[dict], color: str, marker: str) -> None:
    """The delivered ΔBICs, for a driver that archives no version history.

    The deepagents driver keeps only the model it delivered, so there is no trajectory -- only
    the state a trajectory ends in: the band and median line of the delivered ΔBICs, drawn flat
    across a panel with no version axis, and each run's delivered model on it.
    """
    import statistics

    finals = sorted(run["final_dbic"] for run in runs if run.get("final_dbic") is not None)
    axis.set_xlim(-1, 1)
    axis.tick_params(axis="x", bottom=False, labelbottom=False)
    axis.set_xlabel("no version history", fontsize=SMALL, color=MUTED, style="italic",
                    labelpad=2)
    if not finals:
        return
    axis.axhspan(finals[0], finals[-1], color=color, alpha=0.22, lw=0, zorder=3)
    axis.axhline(statistics.median(finals), color=shaded(color, 0.85), lw=1.0, zorder=4)
    step = 0.5 / max(len(finals) - 1, 1)
    axis.scatter([-0.25 + step * i for i in range(len(finals))], finals, s=7, marker=marker,
                 facecolors="white", edgecolors=shaded(color, 0.85), linewidths=0.6, zorder=5)


def style(axis, ylim) -> None:
    axis.set_yscale("symlog", linthresh=LINTHRESH, linscale=LINSCALE)
    axis.set_ylim(*ylim)
    # As in figures 3A and 4: the region better than the reference is shaded, the reference is
    # the dashed line.
    axis.axhspan(ylim[0], 0, color=BETTER_FILL, zorder=0, lw=0)
    axis.axhline(0, color=ZERO_INK, lw=0.6, ls="--", zorder=1)
    axis.tick_params(labelsize=SMALL, length=2, pad=1.5)
    axis.tick_params(which="minor", length=1.2)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)


def compact(value: float, _=None) -> str:
    """``−1k``, ``100``, ``10M``: short enough for a 0.8 in panel, no superscripts under 6 pt."""
    if value == 0:
        return "0"
    sign, size = ("−" if value < 0 else ""), abs(value)
    for divisor, suffix in ((1e6, "M"), (1e3, "k")):
        if size >= divisor:
            return f"{sign}{size / divisor:g}{suffix}"
    return f"{sign}{size:g}"


def decade_ticks(axis, gap_pt: float = 8.5) -> None:
    """Label 0 and as many decades as fit ``gap_pt`` apart; the rest become minor ticks.

    The linear core compresses ±10 to within a few points of 0 on a panel that spans six
    decades, so a fixed tick list collides there. Kept greedily: 0 first, then the decades from
    the outside in, each only if it clears every label already kept.
    """
    from matplotlib.ticker import FixedLocator, FuncFormatter

    low, high = axis.get_ylim()
    decades = [sign * 10 ** power for power in range(1, 8) for sign in (1, -1)
               if low <= sign * 10 ** power <= high]
    to_axes = axis.transScale + axis.transLimits
    height = axis.get_position().height * axis.figure.get_figheight() * 72

    def position(value):
        return to_axes.transform((0, value))[1] * height

    kept = [0.0]
    for value in sorted(decades, key=abs, reverse=True):
        if all(abs(position(value) - position(other)) >= gap_pt for other in kept):
            kept.append(value)
    axis.yaxis.set_major_locator(FixedLocator(sorted(kept)))
    axis.yaxis.set_minor_locator(FixedLocator([d for d in decades if d not in kept]))
    axis.yaxis.set_major_formatter(FuncFormatter(compact))


def _limits(values: list[float]) -> tuple[float, float]:
    """Padded in the symlog's own units, so a mark on the extreme still clears the frame."""
    import math

    def pad(value, factor):
        return value * factor if abs(value) > LINTHRESH else math.copysign(LINTHRESH * 1.6, value)

    low, high = min([*values, -LINTHRESH]), max([*values, LINTHRESH])
    return pad(low, 1.9) if low < 0 else -LINTHRESH, pad(high, 1.9)


def _values(runs: list[dict]) -> list[float]:
    """Everything a panel draws: the best-so-far curves and the delivered marks.

    Not every fitted version -- a run's worst versions never reach a best-so-far curve, and
    scaling to them left a panel's data in its bottom third.
    """
    values = [run["final_dbic"] for run in runs if run.get("final_dbic") is not None]
    for run in runs:
        if run["points"]:
            values += [y for y in best_so_far(run["points"], last_version(run)) if y is not None]
    return values


def _figure(width, height):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from petab_harness import paper_style

    paper_style.register_fonts()
    matplotlib.rcParams.update({**paper_style.rc(), "font.size": SMALL})
    return plt.figure(figsize=(width, height), facecolor=paper_style.SURFACE), paper_style


def _save(fig, output, paper_style) -> None:
    import matplotlib.pyplot as plt

    paper_style.check_fonts(fig)
    fig.savefig(output, dpi=paper_style.DPI, facecolor=paper_style.SURFACE,
                metadata={"CreationDate": None} if str(output).endswith(".pdf") else None)
    plt.close(fig)


def _key(fig, entries, anchor, loc="upper right") -> None:
    """Line = median, band = range, open mark = a run's delivered model."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    handles = []
    for color, label in entries:
        handles.append(Line2D([], [], color=shaded(color, 0.85), lw=1.0, label=label))
    handles += [Patch(facecolor=entries[0][0], alpha=0.22, lw=0, label="range"),
                Line2D([], [], ls="", marker="o", ms=2.6, mfc="white", mew=0.6,
                       mec=shaded(entries[0][0], 0.85), label="delivered")]
    fig.legend(handles=handles, loc=loc, bbox_to_anchor=anchor, ncol=len(handles),
               frameon=False, fontsize=SMALL, handlelength=1.4, handletextpad=0.5,
               columnspacing=1.1, borderpad=0.0, borderaxespad=0.0)


def render_figure2(payload: dict, output, ncols: int = 6) -> None:
    """One panel per Figure 2 benchmark, each on its own y scale."""
    import math

    from matplotlib.ticker import MaxNLocator

    groups = payload["figure2"]
    names = list(groups)
    nrows = math.ceil(len(names) / ncols)
    height = 1.02 * nrows + 0.46
    fig, paper_style = _figure(_width(), height)
    grid = fig.add_gridspec(nrows, ncols, left=0.075, right=0.985, top=1 - 0.30 / height,
                            bottom=0.30 / height, wspace=0.46, hspace=0.62)
    for index, name in enumerate(names):
        axis = fig.add_subplot(grid[index // ncols, index % ncols])
        runs = groups[name]
        style(axis, _limits(_values(runs)))
        draw(axis, runs, VISIBLE)
        axis.set_title(name, loc="left", fontsize=TITLE, fontweight="bold", pad=2.5)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
        decade_ticks(axis)
    _key(fig, [(VISIBLE, "median")], (0.985, 1 - 0.05 / height))
    fig.supxlabel(X_LABEL, fontsize=BODY, y=0.03 / height)
    fig.supylabel(Y_LABEL, fontsize=BODY, x=0.004)
    _save(fig, output, paper_style)


def _width() -> float:
    from petab_harness import paper_style

    return paper_style.TEXT_WIDTH


def render_figure34(payload: dict, output) -> None:
    """Figure 3's arms as model x rung, then figure 4's benchmarks, visible against blinded."""
    from matplotlib.ticker import MaxNLocator

    arms, blinded, visible = payload["figure3"], payload["figure4_blinded"], payload["figure2"]
    height = 4.05
    fig, paper_style = _figure(_width(), height)
    inch = 1 / height
    outer = fig.add_gridspec(2, 1, height_ratios=(2.0, 1.0), hspace=0.62, left=0.105,
                             right=0.985, top=1 - 0.52 * inch, bottom=0.30 * inch)
    top = outer[0].subgridspec(2, 4, wspace=0.16, hspace=0.42)
    ylim = _limits([v for spec in arms.values() for v in _values(spec["runs"])])
    by_cell = {(spec["code"][0], spec["support"]): spec for spec in arms.values()}
    support = {int(k): v for k, v in payload.get("support_levels", {}).items()}
    shared = None
    for row, family in enumerate(("C", "F")):
        model = next(spec["model"] for spec in arms.values() if spec["code"][0] == family)
        for column, rung in enumerate(range(1, 5)):
            axis = fig.add_subplot(top[row, column], sharey=shared)
            spec = by_cell.get((family, rung))
            if row == 0 and rung in support:
                axis.annotate(support[rung], (0, 1), xycoords="axes fraction",
                              xytext=(0, 13), textcoords="offset points", ha="left",
                              va="baseline", fontsize=SMALL, color=MUTED)
            if spec is None:
                axis.set_axis_off()
                axis.text(0.02, 1.0, f"{family}{rung}", transform=axis.transAxes, ha="left",
                          va="bottom", fontsize=TITLE, fontweight="bold", color="#A7B0B6")
                axis.text(0.5, 0.5, "not run", transform=axis.transAxes, ha="center",
                          va="center", fontsize=SMALL, color=MUTED, style="italic")
                continue
            shared = shared or axis
            style(axis, ylim)
            color = SUPPORT_RAMPS[0 if family == "C" else 1][rung - 1]
            marker = MODEL_MARKERS[0 if family == "C" else 1]
            if any(run["points"] for run in spec["runs"]):
                draw(axis, spec["runs"], color, marker)
                axis.xaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
            else:
                delivered_only(axis, spec["runs"], color, marker)
            note = f"   + {len(spec['no_model'])} built no model" if spec["no_model"] else ""
            axis.set_title(spec["code"], loc="left", fontsize=TITLE, pad=2.5,
                           color=shaded(color, 0.85), fontweight="bold")
            if note:
                axis.annotate(note.strip(), (1, 1), xycoords="axes fraction", xytext=(0, 2.5),
                              textcoords="offset points", ha="right", va="bottom",
                              fontsize=SMALL, color=MUTED)
            if column and by_cell.get((family, rung - 1)) is not None:
                axis.tick_params(labelleft=False)
        decade_ticks(shared)
        box = top[row, 0].get_position(fig)
        fig.text(0.052, (box.y0 + box.y1) / 2, model.split("/")[-1], rotation=90, ha="center",
                 va="center", fontsize=BODY)
    bottom = outer[1].subgridspec(1, 4, wspace=0.42)
    for column, name in enumerate(blinded):
        axis = fig.add_subplot(bottom[0, column])
        style(axis, _limits(_values(visible[name]) + _values(blinded[name])))
        draw(axis, visible[name], VISIBLE, zorder=3)
        draw(axis, blinded[name], BLINDED, zorder=6)
        axis.set_title(name, loc="left", fontsize=TITLE, fontweight="bold", pad=2.5)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
        decade_ticks(axis)
    for grid, letter, title, lift in ((outer[0], "A", "Figure 3 arms", 0.34),
                                      (outer[1], "B", "Figure 4 benchmarks", 0.20)):
        y = grid.get_position(fig).y1 + lift * inch
        fig.text(0.004, y, f"{letter}   {title}", ha="left", va="baseline", fontsize=TITLE + 0.5,
                 fontweight="bold")
        if letter == "B":
            _key(fig, [(VISIBLE, "C1  Biology visible"), (BLINDED, "C4  Biology blinded")],
                 (0.985, y), loc="lower right")
    fig.supxlabel(X_LABEL, fontsize=BODY, y=0.03 * inch)
    fig.supylabel(Y_LABEL, fontsize=BODY, x=0.004)
    _save(fig, output, paper_style)
