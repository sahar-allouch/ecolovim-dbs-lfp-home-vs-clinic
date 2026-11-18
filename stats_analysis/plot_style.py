"""
plot_style.py
=============
Unified Plotly style for LFP spectral feature visualisation (ECOLOVIM).

Design philosophy
-----------------
Every color dimension carries meaning:

  Conditions (2 × 2 factorial)
  ┌──────────────┬───────────────┬───────────────┐
  │              │  Stim OFF     │  Stim ON      │
  │              │  warm hue     │  cool hue     │
  ├──────────────┼───────────────┼───────────────┤
  │  Rest  light │ #F5C27A amber │ #7FBBD4 blue  │
  │  Mvt   dark  │ #C97B2E burnt │ #2B6FA8 navy  │
  └──────────────┴───────────────┴───────────────┘
  → Grouping by movement or stim is immediately readable
    from lightness or hue alone.

  Timepoints (temporal gradient)
  M3 (steel blue) → home (sage green) → M6 (terracotta)
  → Cooler = earlier; warmer = later — mirrors the clinical arc.

  ICC reliability (perceptual ramp)
  poor (blush) → moderate (peach) → good (sage) → excellent (sky)
  → Desaturated enough for print; distinct enough for colour-blind readers.

  Significance
  Significant = deep coral accent (#C75B3B); n.s. = neutral gray.

Typography
  Helvetica Neue (Arial fallback). Clear four-level hierarchy:
  title (14) > panel headers (12) > axis labels (11) > ticks/annotations (10/9).

Layout
  White background. Left-aligned titles. Bottom + left spines only (no mirror).
  Subtle horizontal grid (#F2F2F2). Generous but consistent margins.
  Export: SVG + high-res PNG (scale=3, ≈300 dpi) via kaleido.
"""

import os
import plotly.graph_objects as go  # noqa: F401 — re-exported for convenience

# ── Typography ────────────────────────────────────────────────────────────────
FONT         = 'Helvetica Neue, Arial, sans-serif'
SIZE_TITLE   = 14    # figure title
SIZE_HEAD    = 12    # subplot / panel headers
SIZE_LABEL   = 11    # axis labels
SIZE_TICK    = 10    # tick labels
SIZE_ANNOT   = 9     # in-panel annotations (r, ICC, p-values)
SIZE_LEGEND  = 10    # legend text
SIZE_CAPTION = 9     # interpretation caption

_INK = '#222222'      # near-black — titles, axis labels
_MID = '#555555'      # mid-gray   — ticks, annotations
_DIM = '#888888'      # dim gray   — captions, secondary text

# ── Prism palette (Plotly qualitative) ───────────────────────────────────────
# Source: plotly.express.colors.qualitative.Prism
PRISM = [
    '#5F4690',  # 0  purple
    '#1D6996',  # 1  blue
    '#38A6A5',  # 2  teal
    '#0F8554',  # 3  dark green
    '#73AF48',  # 4  lime green
    '#EDAD08',  # 5  amber
    '#E17C05',  # 6  orange
    '#CC503E',  # 7  brick red
    '#94346E',  # 8  dark pink
    '#6F4070',  # 9  dark purple
    '#994E95',  # 10 medium purple
    '#666666',  # 11 gray
]

# ── Condition colors — 2×2 semantic encoding ─────────────────────────────────
# Hue  : warm (amber/orange) = Stim OFF  /  cool (teal/blue) = Stim ON
# Value: lighter = rest      /  darker   = movement
COND_COLORS = {
    'rest_stimOFF': PRISM[5],    # amber        — rest, no stim
    'mvt_stimOFF':  PRISM[6],    # orange       — movement, no stim
    'rest_stimON':  PRISM[2],    # teal         — rest, stim
    'mvt_stimON':   PRISM[1],    # blue         — movement, stim
}

# Compact display labels
COND_LABELS = {
    'rest_stimOFF': 'Rest / Stim OFF',
    'mvt_stimOFF':  'Mvt / Stim OFF',
    'rest_stimON':  'Rest / Stim ON',
    'mvt_stimON':   'Mvt / Stim ON',
}

# ── Timepoint colors ──────────────────────────────────────────────────────────
TP_COLORS = {
    'M3':   PRISM[0],    # purple   — 3 months post-op
    'home': PRISM[3],    # dark green — at-home
    'M6':   PRISM[7],    # brick red  — 6-month follow-up
}

# ── ICC reliability rating colors — perceptual ramp ──────────────────────────
# red → amber → lime → blue  (bad → good)
# Muted, desaturated — same tonal register as the teal heatmap
# (S ≈ 30–45 %, L ≈ 45–58 %)
ICC_RATING_COLORS = {
    'poor':      '#C06060',   # muted rose-red    S=40% L=56%
    'moderate':  '#C8963C',   # muted amber-brown S=55% L=51%
    'good':      '#5A9E6A',   # muted sage-green  S=35% L=49%
    'excellent': '#4A82A8',   # muted steel-blue  S=38% L=47%
}
ICC_COLOR_LIST = [
    ICC_RATING_COLORS['poor'],
    ICC_RATING_COLORS['moderate'],
    ICC_RATING_COLORS['good'],
    ICC_RATING_COLORS['excellent'],
]

# ── Session-pair colors (pairwise ICC) ───────────────────────────────────────
# Use 3 of the 4 COND_COLORS directly — visually unified with scatter plots.
PAIR_COLORS = {
    'M3 vs home': COND_COLORS['mvt_stimOFF'],    # 2nd — orange
    'home vs M6': COND_COLORS['rest_stimON'],    # 3rd — teal
    'M3 vs M6':   COND_COLORS['mvt_stimON'],     # 4th — blue
}

# ── Scatter plot dot color ────────────────────────────────────────────────────
# 4th COND_COLOR — used for all scatter panels regardless of condition.
SCATTER_COLOR = COND_COLORS['mvt_stimON']        # blue

# ── ICC heatmap colorscale ────────────────────────────────────────────────────
# Sequential teal — light (low ICC) → dark teal (high ICC)
ICC_COLORSCALE = [
    [0.000, '#f7f7f7'],
    [0.167, '#dbe9e8'],
    [0.333, '#b8d8d8'],
    [0.500, '#7fbcc0'],
    [0.667, '#4fa3a5'],
    [0.833, '#2c7f82'],
    [1.000, '#1f5f63'],
]

# ── Significance colors ───────────────────────────────────────────────────────
SIG_COLOR   = PRISM[7]   # brick red — significant
NOSIG_COLOR = PRISM[11]  # gray      — non-significant

# ── Legacy Wong palette (kept for compatibility) ──────────────────────────────
WONG = ['#E69F00', '#56B4E9', '#009E73', '#F0E442',
        '#0072B2', '#D55E00', '#CC79A7', '#000000']

# ── Base axis template ────────────────────────────────────────────────────────
_SPINE = '#CCCCCC'   # axis line + tick color

_AXIS = dict(
    showline=True, linecolor=_SPINE, linewidth=1, mirror=False,
    ticks='outside', ticklen=4, tickwidth=1, tickcolor=_SPINE,
    tickfont=dict(family=FONT, size=SIZE_TICK,  color=_MID),
    title_font=dict(family=FONT, size=SIZE_LABEL, color=_INK),
    showgrid=False,
    zeroline=False,
    automargin=True,
)

_GRID = dict(
    showgrid=True,
    gridcolor='#F2F2F2',
    gridwidth=1,
)


# ── Core style function ───────────────────────────────────────────────────────

def apply_style(fig, title='', height=None, width=720, showlegend=True):
    """
    Apply the ECOLOVIM Plotly style to a figure.

    Parameters
    ----------
    fig         : go.Figure
    title       : str   — figure title (left-aligned)
    height      : int   — figure height in px (None = auto)
    width       : int   — figure width in px
    showlegend  : bool
    """
    layout = dict(
        title=dict(
            text=title,
            font=dict(family=FONT, size=SIZE_TITLE, color=_INK),
            x=0.01, xanchor='left',
            pad=dict(l=0, b=8),
        ),
        font=dict(family=FONT, size=SIZE_TICK, color=_MID),
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=width,
        margin=dict(l=75, r=145, t=65, b=75),
        showlegend=showlegend,
        legend=dict(
            x=1.02, y=1.0,
            xanchor='left', yanchor='top',
            borderwidth=0,
            bgcolor='rgba(255,255,255,0)',
            font=dict(family=FONT, size=SIZE_LEGEND, color=_MID),
            title_font=dict(family=FONT, size=SIZE_LEGEND, color=_INK),
            tracegroupgap=4,
        ),
    )
    if height is not None:
        layout['height'] = height
    fig.update_layout(**layout)
    fig.update_xaxes(**_AXIS)
    fig.update_yaxes(**{**_AXIS, **_GRID})
    return fig


def add_caption(fig, text):
    """
    Add a small gray interpretation caption below the plot area.
    Supports <br> for manual line breaks.

    The caption y-offset and bottom margin are scaled with figure height so
    the caption is never cropped in tall figures (e.g. multi-row scatter grids).
    """
    height = fig.layout.height or 500
    # Each line of caption is ~14 px; count <br> occurrences to estimate height
    n_lines = text.count('<br>') + 2        # +2: wrap allowance
    caption_px = n_lines * 14 + 10

    # y in paper coords: negative = below plot area; scale by figure height
    y_offset = -(caption_px + 10) / height

    fig.add_annotation(
        text=text,
        xref='paper', yref='paper',
        x=0.0, y=y_offset,
        showarrow=False,
        align='left',
        font=dict(family=FONT, size=SIZE_CAPTION, color=_DIM),
        xanchor='left', yanchor='top',
    )
    # Ensure the bottom margin is at least as large as the caption block
    current = fig.layout.margin or {}
    b = current.b if hasattr(current, 'b') and current.b else 75
    fig.update_layout(margin=dict(b=max(b, caption_px + 40)))
    return fig


_WEBP_MAX_PX = 8000   # kaleido crashes above ~50 MP; cap longest edge here

def save_fig(fig, results_dir, name):
    """Show and export figure as SVG + WebP at ~300 dpi + interactive HTML. Requires kaleido."""
    
    # fig.show()
    # svg_path  = os.path.join(results_dir, f'{name}.svg')
    # webp_path = os.path.join(results_dir, f'{name}.webp')
    html_path = os.path.join(results_dir, f'{name}.html')
    # try:
    #     fig.write_image(svg_path)
    #     fig.write_image(webp_path, scale=0.2)
    #     print(f'Saved → {name}.svg / .webp  (scale=0.20)')
    # except Exception as e:
    #     print(f'Warning: figure export failed — {e}')
    #     print('Install kaleido: pip install kaleido')
    # 'directory' writes plotly.min.js once beside the html files and links it
    # relatively, instead of pulling it from cdn.plot.ly at open time.  The CDN
    # form renders a blank page anywhere the external script is blocked -- VS
    # Code's built-in preview, an offline machine, a proxied network -- which
    # looks exactly like a figure with no data in it.  One ~3.5 MB js file per
    # results directory, and every html in that directory stays small.
    fig.write_html(html_path, include_plotlyjs='directory')
    print(f'Saved → {name}.html')
    return fig
