"""The non-fujita held-out figure: training fit against prediction, one panel per benchmark.

Figure 3A asks whether fitting the training data better predicts better, on fujita, where the
ablation arms live. This asks the same question on the four other benchmarks that ship a
held-out set, with the same axes so the two read together -- training ΔBIC on x, held-out ΔNLL
on y, both candidate minus reference, both signed so that DOWN and LEFT is better. The shaded
quadrant is where a candidate beats the reference on prediction.

Three arms: the Figure 2 runs, the biology-blinded runs of the same examples, and neural-ODE
fields distilled into rate laws where distillation succeeded (crauste, perelson) -- see
``slurm/_heldout_benchmarks_input.py``.

Two scoring modes are mixed and the panel subtitle says which. crauste, lucarelli and perelson
REFIT the candidate's structure on each held-out unit; raimundez is scored at its fitted point,
with only the new per-blot observation scalings solved, against the published control, because its held-out set is fixed rather than a transfer set. A
ΔNLL is not the same quantity in the two cases and the figure does not pretend otherwise.
"""

from __future__ import annotations

import json
from pathlib import Path

from gemot.paper_fujita import shaded

#: Figure 3's encoding, taken literally: hue and marker name the language model, the step on its
#: ramp names the rung. Both LLM arms here are GLM-5.3-flash runs, so both are circles on the blue
#: ramp -- the baseline at O1's step, the blinded runs at O4's (paper_fujita.SUPPORT_RAMPS).
#: The labels carry figure 3's arm codes, so the two figures' blues read as the same arms.
ARM_STYLE = {
    "regular": {"color": "#0A3559", "marker": "o", "label": "O1  Biology visible"},
    "blinded": {"color": "#9BD0F2", "marker": "o", "label": "O4  Biology blinded"},
    # No language model, so neither of figure 3's model hues, and a marker no model uses. It
    # was burnt orange, which is figure 3's Astra ramp. Magenta clears both blues by OKLab
    # ΔE >= 15.9 under every simulated colour-vision deficiency; a charcoal did not (9.9).
    "distilled": {"color": "#C0407F", "marker": "^", "label": "Neural ODE, distilled"},
}
#: Figure 3A's band, label ink and reference lines.
BETTER_FILL = "#EEF6F2"
BETTER_INK = "#3E7863"
ZERO_INK = "#526370"
GRID = "#E2E7EA"
MUTED = "#5F6B74"


def read_input(root: Path) -> dict | None:
    path = Path(root) / "paper" / "heldout_benchmarks.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError(
            "unsupported held-out benchmarks schema; rebuild with "
            "slurm/_heldout_benchmarks_input.py"
        )
    return payload


def _limits(points, pad=0.12):
    """Symmetric-ish bounds that always include zero, so the quadrant is visible."""
    values = [p for p in points if p is not None]
    low, high = min([*values, 0.0]), max([*values, 0.0])
    span = (high - low) or 1.0
    return low - pad * span, high + pad * span


def _clipped_limits(values, pad=0.12, whisker=3.0):
    """Bounds set by the BULK, plus the threshold above which points are drawn off-scale.

    A Tukey upper fence on the pooled values: q3 + `whisker`*IQR. The earlier rule keyed off the
    single largest value, which stopped working the moment a third arm arrived -- perelson's
    +1,023 was no longer extreme enough relative to lucarelli's +1,241 to trigger a clip, so
    both stayed on the axis and flattened all four panels into a band. A quantile fence does not
    care how many runaways there are.

    The lower bound is never clipped: "better than the reference" is the interesting direction
    and there are no runaways in it, so cutting there would hide the result rather than an
    artefact.
    """
    import numpy as np

    finite = sorted(v for v in values if v is not None)
    if len(finite) < 4:
        return _limits(finite, pad), None
    q1, q3 = np.percentile(finite, [25, 75])
    fence = float(q3 + whisker * (q3 - q1))
    kept = [v for v in finite if v <= fence]
    if len(kept) == len(finite):
        return _limits(finite, pad), None
    low, high = _limits(kept, pad)
    return (low, high), max(kept)


def _x_fence(values, whisker=1.5):
    """Training ΔBIC beyond which a run is drawn off the panel's right edge: q3 + 1.5 IQR.

    Looser than the y fence (3 IQR) because x is per panel, over ten-odd runs, where one run
    can hold the whole axis; at 1.5 only lucarelli's +2,642 is outside, in any panel.
    """
    import numpy as np

    if len(values) < 4:
        return float("inf")
    q1, q3 = np.percentile(values, [25, 75])
    return float(q3 + whisker * (q3 - q1))


def _marginal(axis, arms, ylim, fontsize):
    """Median and min-max range per arm, as figure 3A draws its marginals.

    A range bar with caps and a median marker rather than a box: with four or five runs an arm
    a box plot draws quartiles off too few points to mean anything, and the question here is
    only whether the centre moves between the arms.
    """
    import numpy as np

    offsets = np.linspace(0.32, -0.32, len(ARM_STYLE))
    for offset, (name, style) in zip(offsets, ARM_STYLE.items(), strict=True):
        sample = [
            min(p["heldout_delta_nll"], ylim[1]) for p in arms.get(name, {}).get("points", [])
        ]
        if not sample:
            continue
        low, high, middle = min(sample), max(sample), float(np.median(sample))
        axis.plot(
            [offset, offset],
            [low, high],
            color=style["color"],
            linewidth=0.9,
            solid_capstyle="butt",
            zorder=3,
        )
        for end in (low, high):
            axis.plot(
                [offset - 0.09, offset + 0.09],
                [end, end],
                color=style["color"],
                linewidth=0.9,
                zorder=3,
            )
        axis.plot(
            [offset],
            [middle],
            marker=style["marker"],
            markersize=2.6,
            markerfacecolor=style["color"],
            markeredgecolor=shaded(style["color"]),
            markeredgewidth=0.4,
            linestyle="none",
            zorder=4,
        )
    axis.axhline(0, color=ZERO_INK, linestyle="--", linewidth=0.7, zorder=1)
    axis.set_xlim(-0.6, 0.6)
    # Transparent: a run past its cloud's right edge is drawn in the spacer and can reach into
    # this axes, whose white background would otherwise paint over half of it.
    axis.patch.set_visible(False)
    axis.set_xticks([])
    axis.tick_params(labelleft=False, left=False, length=0)
    for spine in ("top", "right", "left", "bottom"):
        axis.spines[spine].set_visible(False)


#: Type sizes in points at print size; ICLR's 6 pt floor is SMALL.
TITLE, BODY, SMALL = 7.5, 6.5, 6.0
#: Panel titles' baseline, in axes fractions: above the band over the frame where off-scale runs
#: are drawn (render places that band 0.055 of the y span up).
TITLE_Y = 1.15


def render(payload: dict, output, fontsize: float = BODY) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    from gemot import paper_style

    matplotlib.rcParams.update(paper_style.rc())

    examples = payload["examples"]
    # ONE y scale for every panel. The panels answer the same question on four benchmarks, so a
    # per-panel scale would let a ±12 spread and a ±300 spread look alike. The cost is that the
    # smaller benchmarks occupy little of the axis; the clip keeps one runaway from making that
    # worse, and it is drawn outside the frame rather than dropped -- named by its place, not by
    # a printed value, as figure 3A draws its own.
    every = [
        p["heldout_delta_nll"] for e in examples for a in e["arms"].values() for p in a["points"]
    ]
    ylim, offscale = _clipped_limits(every)
    matplotlib.rcParams.update({"pdf.compression": 9, "svg.hashsalt": "heldout-benchmarks"})
    # Full ICLR text width, placed at 100%, so every size below is a size on the page. ONE row:
    # the page budget cannot spare the 2 x 2 grid's 4.3 in. The cost is ~1 in per scatter, so
    # the x axes carry at most three ticks and share one label, and the subtitles are terse.
    # Margins in INCHES, so they stay tight at any height: key and titles above, tick labels
    # and the shared x label below.
    page = (paper_style.TEXT_WIDTH, 1.61)
    above, below = 0.42, 0.27
    # Each benchmark is a scatter plus a slim marginal, as panel 3A is: the cloud shows the
    # runs, the marginal shows median and range per arm so the two arms can be compared without
    # the reader estimating a centre by eye. Only the y marginal is carried over -- with the y
    # axis shared it is the comparable one.
    fig = plt.figure(figsize=page, facecolor="white")
    # (cloud, marginal, gap) per benchmark, the gap column left empty. wspace cannot do this: it
    # is uniform, so any value that separates the benchmarks also separates each cloud from its
    # own marginal.
    # A thin spacer between each cloud and its marginal holds the band past the cloud's right
    # edge, where a run beyond the panel's ΔBIC scale is drawn -- on the marginal it sat on the
    # range bars.
    ratios, columns = [], []
    for index in range(len(examples)):
        columns.append(len(ratios))
        ratios += [1.0, 0.07, 0.16] if index == len(examples) - 1 else [1.0, 0.07, 0.16, 0.20]
    grid = fig.add_gridspec(
        1,
        len(ratios),
        width_ratios=ratios,
        left=0.098,
        right=0.995,
        top=1 - above / page[1],
        bottom=below / page[1],
        wspace=0.0,
    )
    axes, margins = [], []
    for index, column in enumerate(columns):
        main = fig.add_subplot(grid[0, column], sharey=axes[0] if axes else None)
        margin = fig.add_subplot(grid[0, column + 2], sharey=main)
        # sharey= on add_subplot links the limits but does NOT hide the duplicate tick labels
        # the way plt.subplots(sharey=True) does.
        if index:
            main.tick_params(labelleft=False)
        axes.append(main)
        margins.append(margin)

    titles = []
    for axis, margin, entry in zip(axes, margins, examples, strict=True):
        arms = entry["arms"]
        xs = [p["training_dbic"] for a in arms.values() for p in a["points"]]
        # x is clipped per panel, as figure 3A clips its ΔBIC axis: lucarelli's one blinded run
        # at +2,642 otherwise stretched the panel to four times its bulk. It is drawn past the
        # right edge, and a run above the shared scale in a band over the frame: OUTSIDE the
        # scales, so no point sits at a coordinate it does not have.
        xfence = _x_fence(xs)
        xlim = _limits([x for x in xs if x <= xfence])
        beside = xlim[1] + 0.05 * (xlim[1] - xlim[0])
        over = ylim[1] + 0.055 * (ylim[1] - ylim[0])
        # The "better than reference" region is y < 0: the candidate predicts the held-out data
        # better than the published model does. Drawn before the points so it never hides one.
        axis.axhspan(ylim[0], 0, color=BETTER_FILL, zorder=0)
        axis.axhline(0, color=ZERO_INK, linestyle="--", linewidth=0.8, zorder=1)
        axis.axvline(0, color="#C5CDD2", linewidth=0.7, linestyle=":", zorder=1)

        for name, arm in arms.items():
            if not arm["points"]:
                continue
            style = ARM_STYLE[name]
            # Each run at its own coordinates, or OUTSIDE the scale it has left: past the right
            # edge when its ΔBIC is beyond the panel's, in the band over the frame when its ΔNLL
            # is beyond the shared scale, and in the corner where the two meet when both are.
            # Drawn like any other run -- where it sits is what says it is off the scale.
            xs_drawn = [
                beside if p["training_dbic"] > xfence else p["training_dbic"] for p in arm["points"]
            ]
            ys_drawn = [
                over
                if offscale is not None and p["heldout_delta_nll"] > ylim[1]
                else p["heldout_delta_nll"]
                for p in arm["points"]
            ]
            axis.scatter(
                xs_drawn,
                ys_drawn,
                s=15,
                c=style["color"],
                marker=style["marker"],
                # Figure 3's outline: the mark's own hue, darker. White erased the
                # pale blinded step against the page.
                edgecolors=shaded(style["color"]),
                linewidths=0.5,
                zorder=3,
                clip_on=False,
            )

        _marginal(margin, arms, ylim, fontsize)
        axis.set_xlim(*xlim)
        axis.set_ylim(*ylim)
        # pad clears the mode line below it; the two are one stack, not two labels
        # competing for the same corner.
        # Name and mode on ONE line: the bold name, then the mode and size in muted ink right
        # after it. Terse, because a panel is ~1.3 in to the next one; the caption spells out
        # what each mode means. Lifted clear of the off-scale band over the frame.
        kind = {"transfer": "transfer", "fixed": "fixed"}.get(entry["mode"], entry["mode"])
        # n is the held-out measurements the panel's ΔNLL is computed on (the reference score
        # records them), not the training count: that one orders the panels and is Figure 2's,
        # and printed beside "transfer"/"fixed" it read as the size of the held-out set.
        size = (entry.get("reference") or {}).get("n")
        subtitle = f"{kind}" + (f" · n={size:,}" if size else "")
        # BASELINE-aligned, both halves: va="bottom" aligns the boxes' bottoms, and a 7.5 pt
        # bold name and a 6 pt label have different descents, so the two sat on different lines.
        name = axis.text(
            0,
            TITLE_Y,
            entry["example"],
            transform=axis.transAxes,
            fontsize=TITLE,
            fontweight="bold",
            va="baseline",
        )
        titles.append((axis, name, subtitle))
        axis.grid(axis="y", color=GRID, linewidth=0.5, zorder=0)
        axis.tick_params(labelsize=SMALL, length=2, pad=1.5)
        # raimundez spans ~1,400 ΔBIC and matplotlib's default five-plus ticks ran into each
        # other at this panel width.
        axis.xaxis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 2, 2.5, 5, 10]))
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)

    # One label for the shared axis, centred on the whole left edge rather than repeated per row.
    axes[0].set_ylabel("Held-out ΔNLL\n(candidate − reference)", fontsize=BODY, labelpad=2)
    fig.supxlabel("Training ΔBIC  (candidate − reference)", fontsize=BODY, y=0.0, va="bottom")

    # The mode follows its name at the name's measured width, which needs a renderer.
    renderer = fig.canvas.get_renderer()
    for axis, name, subtitle in titles:
        end = name.get_window_extent(renderer).transformed(axis.transAxes.inverted()).x1
        axis.text(
            end,
            TITLE_Y,
            "  " + subtitle,
            transform=axis.transAxes,
            fontsize=SMALL,
            color=MUTED,
            va="baseline",
        )

    handles = [
        Line2D(
            [],
            [],
            color=s["color"],
            marker=s["marker"],
            linestyle="none",
            markersize=3.6,
            markeredgecolor=shaded(s["color"]),
            markeredgewidth=0.4,
            label=s["label"],
        )
        for s in ARM_STYLE.values()
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        frameon=False,
        fontsize=BODY,
        handletextpad=0.3,
        ncol=len(handles),
        columnspacing=1.0,
        borderpad=0.0,
    )
    paper_style.check_fonts(fig)
    fig.savefig(
        output,
        dpi=paper_style.DPI,
        facecolor="white",
        metadata={"CreationDate": None} if str(output).endswith(".pdf") else None,
    )
    plt.close(fig)
