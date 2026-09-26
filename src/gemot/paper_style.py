"""The page every manuscript figure shares: white surface, Arial-metric type.

Figures 2 to 4 were drawn by three renderers that each chose a font and a background, and they
drifted: DejaVu Sans against the default, an off-white surface against a white one. They now take
both from here.

Arial is proprietary and absent from the cluster, so it cannot be vendored. Liberation Sans is
metric-compatible with it -- every glyph has Arial's advance width, so text set in one occupies
exactly the space of the other -- and is OFL-licensed, so it ships in ``paper/fonts/``. Arial is
listed first: where it is installed it wins, and the layout does not move either way.
"""
from __future__ import annotations

from pathlib import Path

SURFACE = "#FFFFFF"
#: ICLR's ``\textwidth`` (iclr2027_conference.sty: 5.5 true in). Every main-text figure is drawn
#: AT this width, so LaTeX places it at 100% and a point in the artwork is a point on the page.
TEXT_WIDTH = 5.5
#: The page height a full-width figure may claim: figure2.tex caps it at 0.82\textheight (9 in).
MAX_HEIGHT = 0.82 * 9.0
DPI = 300
#: The smallest type anywhere in a figure, in points at print size.
MIN_FONT = 6.0
#: The one exception: short annotations set beside a mark, tagged ``gid=NOTE_GID``, may go to 5 pt
#: where 6 does not fit (figure 3B's latent-depth notes). Everything else stays at MIN_FONT.
NOTE_FONT, NOTE_GID = 5.0, "note"
FONT_FAMILY = ["Arial", "Liberation Sans"]
FONT_DIR = Path(__file__).resolve().parents[2] / "paper" / "fonts"


def register_fonts() -> None:
    """Make the vendored faces visible to Matplotlib; idempotent."""
    from matplotlib import font_manager

    known = {entry.fname for entry in font_manager.fontManager.ttflist}
    for path in sorted(FONT_DIR.glob("*.ttf")):
        if str(path) not in known:
            font_manager.fontManager.addfont(str(path))


def rc() -> dict:
    """The rcParams every figure applies before drawing."""
    register_fonts()
    # The inks are figure 3's, which set them first; figure 4 reads as its companion panel.
    return {"font.family": "sans-serif", "font.sans-serif": FONT_FAMILY,
            "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE, "pdf.fonttype": 42, "ps.fonttype": 42,
            "axes.edgecolor": "#69737B", "axes.labelcolor": "#26343E", "axes.linewidth": 0.6,
            "xtick.color": "#43515C", "ytick.color": "#43515C", "text.color": "#26343E",
            "xtick.major.width": 0.6, "ytick.major.width": 0.6}


def small_text(fig, floor: float = MIN_FONT) -> list[tuple[float, str]]:
    """Every visible text in ``fig`` set below ``floor`` points, as ``(size, text)``.

    Tick labels included -- they are Text objects too. Called before saving, so a panel that
    reintroduces a 4-pt annotation fails the build instead of shipping unreadable.
    """
    from matplotlib.text import Text

    found = []
    for text in fig.findobj(Text):
        limit = NOTE_FONT if text.get_gid() == NOTE_GID else floor
        if text.get_visible() and text.get_text().strip() and text.get_fontsize() < limit - 1e-6:
            found.append((round(text.get_fontsize(), 2), text.get_text()[:40]))
    return found


def check_fonts(fig, floor: float = MIN_FONT) -> None:
    offenders = small_text(fig, floor)
    if offenders:
        listed = "; ".join(f"{size}pt {text!r}" for size, text in offenders[:8])
        raise ValueError(f"{len(offenders)} text(s) below {floor}pt: {listed}")
