"""
Shared style for the journal-style table figures.

The figure scripts share one geometry (page width, margins, rules, fonts);
this module makes that shared contract explicit so the figures cannot drift
apart.

What lives here
---------------
    geometry   column pitch, gaps, label gutter, row height, figure size
    type       every font size the two tables use
    ink        rule / muted / faint colours
    palettes   ICC reliability tiers (SEQUENTIAL, one hue) and the LME
               direction tints (DIVERGING, two hues)
    save       one writer, so both tables come out at the same width

Why the two palettes differ in kind
-----------------------------------
The ICC cell tint encodes a TIER — an ordered magnitude — so it is a single-hue
sequential ramp, light to dark. The LME cell tint encodes DIRECTION — a signed
quantity around a neutral zero — so it is a two-hue diverging pair. Keeping the
two jobs on different palette types is what lets a reader see both panels at
once without reading one as the other; blue never means "reliable" in one panel
and "negative" in the next.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager

# ── Font ─────────────────────────────────────────────────────────────────────
_ARIAL_DIR = Path('/System/Library/Fonts/Supplemental')
for _fn in ('Arial.ttf', 'Arial Bold.ttf', 'Arial Italic.ttf',
            'Arial Bold Italic.ttf'):
    _fp = _ARIAL_DIR / _fn
    if _fp.exists():
        font_manager.fontManager.addfont(str(_fp))
FAMILY = 'Arial' if (_ARIAL_DIR / 'Arial.ttf').exists() else 'DejaVu Sans'


def apply_rcparams(font=None):
    """`font` overrides the shared FAMILY for one caller only (e.g. a table
    that wants Times New Roman) without touching every other script that
    imports this module — see the FIG_FONT leak this mirrors in
    stats_analysis/band_peak_detection.py."""
    plt.rcParams.update({'font.family': font if font is not None else FAMILY,
                          'pdf.fonttype': 42})


# ── Geometry (data units; 1 column = 1.0) ────────────────────────────────────
COL_W = 1.0
GAP_HEMI = 0.35        # between the L and R blocks of one condition
GAP_COND = 0.9         # between conditions
X_LABEL = -6.2         # left edge of the feature-label gutter
ROW_H = 1.0

# figure size, in inches, from the drawn extent
FIG_W_PER_UNIT = 0.62
FIG_W_PAD = 0.6
FIG_H_PER_ROW = 0.30
FIG_H_PAD = 2.4          # title + legend + footnote band
FIG_H_PAD_PANEL = 0.9    # panel mode: no title, no legend, no footnote


def fig_width(x_total: float) -> float:
    return FIG_W_PER_UNIT * (x_total - X_LABEL) + FIG_W_PAD


# ── Type ─────────────────────────────────────────────────────────────────────
SIZE_TITLE = 12.5
SIZE_PANEL = 13.0        # the A / B panel letter
SIZE_COND = 11
SIZE_HEADER = 9
SIZE_SUBHEADER = 6.6     # the per-pair column labels
SIZE_GROUP = 8.6
SIZE_LABEL = 8.2
SIZE_VALUE = 7.4
SIZE_CI = 4.9
SIZE_LEGEND = 7.6
SIZE_FOOT = 7.2

# ── Ink ──────────────────────────────────────────────────────────────────────
INK = '#000000'
MUTED = '#000000'
RULE = '#000000'
FAINT = '#d7dde3'
SWATCH_EDGE = '#CBD5E1'
SURFACE = '#ffffff'

LW_RULE_TOP = 1.3
LW_RULE_MID = 1.0
LW_RULE_BOTTOM = 1.1
LW_HEADER_UNDERLINE = 0.9
LW_SEP_COND = 0.8
LW_SEP_HEMI = 0.6
LW_CELL = 0.8

# ── ICC reliability tiers ────────────────────────────────────────────────────
# The original tint set, kept by request: white -> blue -> green -> deeper
# green. Two caveats to know when reading the merged panel figure:
#   - it is two hues, not one, so it is not a sequential ramp in the strict
#     sense; the blue-to-green step reads as a category change rather than as
#     one more step up an order;
#   - blue also carries "lower at the first-named session" in the LME panel, so
#     in a merged figure one hue does two jobs.
# Swap to a single-hue or neutral ramp if either becomes a problem.
RATING_HM = {'poor': '#ffffff', 'moderate': '#DBEAFE',
             'good': '#DCFCE7', 'excellent': '#BBF7D0'}
RATING_NAME = {'poor': 'Poor <0.50', 'moderate': 'Moderate 0.50–0.75',
               'good': 'Good 0.75–0.90', 'excellent': 'Excellent >0.90'}
RATING_ORDER = ['poor', 'moderate', 'good', 'excellent']

# ── LME direction: DIVERGING, two hues about a neutral (unfilled) middle ─────
# Validated with dataviz/scripts/validate_palette.js against a white surface.
TINT = {'neg': '#9ec5f4', 'pos': '#f2a49c'}

# ── Delivered page width ─────────────────────────────────────────────────────
# Every figure is authored to land at ONE printed width, so a point size means
# the same thing on the page in all of them.  Without this a shared type scale
# is decorative: fig_trajectories was drawn 286 mm wide and fig_icc_summary
# 163 mm, so the same declared 8.2 pt label printed at 4.9 pt in one and 8.5 pt
# in the other once both were placed in a 170 mm column.
#
# Scripts save with bbox_inches='tight', which crops to the drawn artists, so
# the delivered width is smaller than the declared figsize by a factor that
# depends on the layout.  Each script therefore divides by its own measured
# ratio rather than passing TARGET_W_IN to figsize directly.
TARGET_W_MM = 185.0
TARGET_W_IN = TARGET_W_MM / 25.4

PNG_DPI = 900


def save(fig, out_dir: Path, stem: str, fig_w: float | None = None):
    """Write png. `fig_w` pins the output width so two tables saved from
    different scripts line up: bbox_inches='tight' otherwise crops each figure
    to its own legend text and returns slightly different widths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    bbox = None if fig_w else 'tight'
    fig.savefig(out_dir / f'{stem}.png', dpi=PNG_DPI, facecolor=SURFACE,
                bbox_inches=bbox)
    return out_dir / f'{stem}.png'

