"""Standardised effect sizes (Cohen's d) for the per-condition LME contrasts, bilateral.

The results table (lme_m3m6_summary_bilateral.csv) holds every beta and every
interval.  Beta cannot be read across features, because it sits on
whatever scale the feature was fitted on (natural log for band / tremor / broadband
power and the exponent, log10 for the offset, peak power and flattened power, raw Hz
for peak centre frequency and bandwidth).  A -0.60 in the delta row and a -0.60 in
the offset row are not the same size of effect.

This plate puts every contrast on ONE scale so it can be read across features:

    d = beta / sigma_total,   sigma_total = sqrt(group_var + scale)

taken from the `cohen_d` column of the CSV, not recomputed here.  d is scale-free,
so the Hz rows go through it unchanged: dividing by a SD is valid whatever the unit.

All 156 estimates are drawn, one dot each, so any individual cell can be found.
Three channels, three separate jobs, none of them doubled:

    position   feature x condition x contrast
    area       |d|            magnitude, on the size legend's four steps
    fill hue   sign of d      direction — TWO colours, not a ramp
    opacity    evidence       three steps: survives BH (q < 0.05) opaque,
                              nominal p < 0.05 only at 48%, p >= 0.05 at 16%
    contour    evidence       same-hue ring, doubling the two upper opacity steps
                              so they survive greyscale print and the smallest
                              dots: full 0.7 pt for BH survivors, 0.4 pt at 60%
                              for nominal p < 0.05 only

Magnitude is carried by area alone.  A continuous colour ramp over d was tried and
dropped: it made hue a second, harder-to-read copy of the size channel and needed a
colour bar to decode, when the question the plate answers is which way a feature
moved and whether the move is big.

LAYOUT IS IN MILLIMETRES.  The single axes spans the whole page and its data
coordinates ARE millimetres, y increasing downward, so every constant in the
GEOMETRY block below is the measurement it is named after and can be checked with
a ruler on the printed page.  This replaced an earlier arbitrary-unit grid whose
spacing could only be judged by eye and drifted every time the figure was resized.
The figure is saved with the default bbox (NOT bbox_inches='tight'), so the
declared page size is the delivered one and every point size is a real point size.

Sign convention: the model is
    feature ~ C(timepoint, Treatment("home")) + (1|patient)
fitted separately per condition, so d is a difference from the at-home session and
`M3 vs M6` is the post-hoc contrast M3 - M6.  Columns are labelled with the signed
name (M3-Home, M6-Home, M3-M6).  Negative = lower at the FIRST-named timepoint.
M3-M6 runs backwards in time: positive means the feature decreased by M6.

BH is applied per timepoint contrast (13 features x 4 conditions = 52 tests), so the
opacity step is set by a q corrected across a *column block*, not across the whole
plate.  Peak-parameter rows are conditional on a beta peak being detected; their
n_obs is smaller than the rest, so a large d there is a statement about a subset -
see peak_params_analysis.py.

Output: Results/Figures/fig_lme_effectsize_bilateral.png
Values read from Results/stats/lme/lme_m3m6_summary_bilateral.csv; none estimated.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle

import table_figure_style as S
from make_icc_summary_figure import GROUPS
from stats_analysis.lme_analysis import ALPHA

CSV = Path('Results/stats/lme/lme_m3m6_summary_bilateral.csv')
OUT = Path('Results/Figures')
STEM = 'fig_lme_effectsize_bilateral'

# ── Rows: grouping, order AND wording are the ICC summary plates' own — imported,
# not retyped, so the plates read row for row and cannot drift apart. Two rows
# need care when read here: 'Broadband power' is total power over PSD_FREQ_RANGE,
# periodic content included (not the aperiodic component — exponent and offset
# carry that); 'Aperiodic-corrected power' is flat_abs_beta, the FOOOF log10
# residual band-summed, NOT absolute beta power (the "Beta" row of the first
# family). The f0 labels are mathtext, which needs the mathtext font wired to
# Arial in main() or matplotlib falls back to DejaVu for those two labels alone.
# Shaded behind alternate families only, so the band groups rows without a rule
# crossing the dots. Index into GROUPS.
SHADED = {0, 2}

CONDS = [('rest_stimOFF', 'Rest'), ('mvt_stimOFF', 'Movement'),
         ('rest_stimON', 'Rest'), ('mvt_stimON', 'Movement')]
STIM_SPAN = [('Stim OFF', 0, 2), ('Stim ON', 2, 4)]
# Column headers are abbreviated because they cannot be otherwise: at 7.5 pt
# "M3−Home" measures 12.2 mm against a 10 mm column pitch, so the full names of
# adjacent columns overlap. Restoring them needs a 13.2 mm pitch, i.e. a 211 mm
# page. "H" is the at-home session — state it in the caption, the plate no longer
# expands it.
CONTRASTS = [('M3 vs home', 'M3−H'), ('M3 vs M6', 'M3−M6'),
             ('M6 vs home', 'M6−H')]

# Deep petrol / warm coral. dataviz/scripts/validate_palette.js against a white
# surface: CVD separation 13.0 (protan) / 34.4 (tritan), normal-vision floor 30.4
# and 3:1 contrast all PASS, the last one mattering because the faint dots draw at
# 16% alpha. The petrol FAILS the chroma floor (0.078 against 0.10) — dark and
# desaturated enough to read near-black at the smallest dots; #14807F lifts it over
# the floor at the same hue if that matters on paper.
DIR_C = {'neg': '#0F6B6D', 'pos': '#E76F51'}

INK = '#222222'
MUTED = '#6B7280'
BAND_FILL = '#F7F8F9'
SURFACE = '#ffffff'

# Evidence is read off opacity, in three steps. One place, used by both the dots
# and the legend swatches, so the two cannot drift apart.
ALPHA_FDR = 1.00     # q < ALPHA
ALPHA_NOM = 0.48     # p < ALPHA, q >= ALPHA
ALPHA_NS = 0.16      # p >= ALPHA
# Contour on the cells that survive BH: opacity alone separates them from the
# nominal ones only by a step in lightness, which a greyscale print or a small
# dot loses. Drawn as its own ring, never as the dot's edge, so the dot's alpha
# cannot fade it, and set a hair outside the fill so the two never merge.
# Same hue as the dot it circles (a halo), so it adds no colour to decode.
RING_LW = 0.7        # points, BH survivors
# Nominal-only cells get a lighter, thinner ring of the same hue, so the two
# evidence steps are told apart by weight and the FDR ring stays the strongest.
RING_LW_NOM = 0.4
RING_ALPHA_NOM = 0.6
RING_GAP = 1.2       # points, ring diameter minus dot diameter


def ring(ax, x, y, area, color, nominal=False):
    """Significance contour around a dot of the given marker area (points^2).

    Full weight for BH survivors; nominal=True draws the lighter, thinner ring.
    """
    ax.scatter([x], [y], s=(area ** 0.5 + RING_GAP) ** 2, facecolor='none',
               edgecolor=color, linewidths=RING_LW_NOM if nominal else RING_LW,
               alpha=RING_ALPHA_NOM if nominal else 1.0, zorder=4)

# ── Type ─────────────────────────────────────────────────────────────────────
# Family is Arial via S.apply_rcparams, shared with make_trajectory_figure; the
# sizes are this plate's own, set so the hierarchy reads
#   10 bold  Stim OFF  >  9 bold  Canonical band power
#   >  9 reg  Rest  >  8 reg  Beta (13-35 Hz)  >  7.5 reg  M3-Home
SZ_STIM = 10.0
SZ_COND = 9.0
SZ_SUB = 7.5
SZ_SECTION = 9.0
SZ_LABEL = S.SIZE_LABEL   # same as the ICC summary plates' feature labels
SZ_LEGEND_TITLE = 8.0
SZ_LEGEND = 7.0

# ── GEOMETRY, all millimetres ────────────────────────────────────────────────
PAGE_W = 185.0
PAGE_H = 147.5        # +7.5 mm for the 3 widened family gaps, +4 mm bottom margin under the legend

# horizontal
X_LABEL_R = 44.0        # right edge of the feature-name column; the dot grid
                        # is laid out from it
# Outline indentation, as on the ICC summary plates: family headings at the
# margin, every feature name left-aligned at one common indent under them.
X_HEAD = 2.6
X_LABEL_L = 9.8
GAP_LABEL_DOT = 5.5     # label column -> first dot centre
COL_PITCH = 10.0        # dot column centre-to-centre inside one condition block
GAP_BLOCK = 5.0         # EXTRA space between Rest and Movement of one stim state
GAP_STIM = 10.0         # EXTRA space across the OFF/ON divider

# vertical
Y_TOP = 4.5             # centre of the Stim OFF/ON line
GAP_STIM_COND = 4.5     # -> Rest / Movement
GAP_COND_RULE = 2.2     # -> the hairline under Rest / Movement
GAP_RULE_SUB = 2.2      # -> M3-Home etc
GAP_SUB_HEAD = 4.5      # -> first family heading
GAP_HEAD_ROW = 5.0      # family heading -> its first data row
ROW_PITCH = 5.75        # data row centre-to-centre inside one family
GAP_ROW_HEAD = 8.0      # last row of a family -> next family heading
BAND_PAD = 2.25         # shaded block above/below the outermost dot centres

GAP_DATA_LEGEND = 11.0  # last data row -> legend heading
LEG_TITLE_ROW = 4.0     # legend heading -> first entry
LEG_ROW = 4.3           # legend entry centre-to-centre
# The three legend columns are NOT spread evenly across the page: they start at
# the page margin, at the first dot column, and at the first dot column of the
# stim-ON half, so each one lines up with a landmark of the grid above it rather
# than sitting at an arbitrary offset. Computed in draw_legend, never hardcoded.
LEG_X_MARGIN = X_HEAD        # legend starts on the family-heading margin
LEG_SWATCH_DX = 5.0          # column left -> swatch centre
LEG_TEXT_DX = 9.5            # column left -> label text

DMAX = 2.2              # area cap; must exceed the largest |d| in the table
                        # (the script stops if it does not), so nothing clips
AREA_MIN, AREA_SPAN = 5.0, 150.0   # points^2; max dot is ~4.4 mm across against
                                   # a 5.75 mm row pitch
# One rule weight and one rule colour for BOTH separators — the hairline under
# Rest/Movement and the OFF/ON divider. They do the same job (organise, not
# announce), so they must not read as two different kinds of line.
RULE_C = '#C4C9CE'
LW_RULE = 0.25


def marker_area(d: float) -> float:
    """Marker AREA in points^2, proportional to |d| so the eye reads area, not
    radius. Capped at DMAX, set above the observed maximum so nothing clips."""
    return AREA_MIN + AREA_SPAN * min(abs(d), DMAX) / DMAX


def load() -> pd.DataFrame:
    df = pd.read_csv(CSV)
    order = [f for _, feats in GROUPS for f, _ in feats]
    missing = set(order) - set(df.feature.unique())
    if missing:
        raise SystemExit(f'features absent from {CSV}: {sorted(missing)}')
    return df.set_index(['feature', 'condition', 'contrast'])


def column_x() -> list[list[float]]:
    """x centre of every dot column, in mm, as [condition][contrast]."""
    xs, x = [], X_LABEL_R + GAP_LABEL_DOT
    for ci in range(len(CONDS)):
        xs.append([x + ki * COL_PITCH for ki in range(len(CONTRASTS))])
        x = xs[-1][-1] + COL_PITCH + (GAP_STIM if ci == 1 else GAP_BLOCK)
    return xs


def row_y() -> tuple[list[tuple], list[tuple], float]:
    """(rows, headings, y of the last row), in mm."""
    y = Y_TOP + GAP_STIM_COND + GAP_COND_RULE + GAP_RULE_SUB + GAP_SUB_HEAD
    rows, heads = [], []
    for gi, (gname, feats) in enumerate(GROUPS):
        heads.append((gname, y, gi))
        y += GAP_HEAD_ROW
        for f, lab in feats:
            rows.append((f, lab, y, gi))
            y += ROW_PITCH
        y += GAP_ROW_HEAD - ROW_PITCH
    return rows, heads, rows[-1][2]


def draw(ax, df):
    xs = column_x()
    rows, heads, y_last = row_y()
    y_cond = Y_TOP + GAP_STIM_COND
    y_rule = y_cond + GAP_COND_RULE
    y_sub = y_rule + GAP_RULE_SUB

    ax.set_xlim(0, PAGE_W)
    ax.set_ylim(PAGE_H, 0)      # y grows downward, so mm read top-down
    ax.axis('off')

    # shaded family blocks, padded from the outermost dot centres only
    for gi in SHADED:
        ys = [y for _, _, y, g in rows if g == gi]
        ax.add_patch(Rectangle((0, min(ys) - BAND_PAD), PAGE_W,
                               max(ys) - min(ys) + 2 * BAND_PAD,
                               facecolor=BAND_FILL, edgecolor='none', zorder=0))

    for _, lab, y, _ in rows:
        ax.text(X_LABEL_L, y, lab, ha='left', va='center', fontsize=SZ_LABEL,
                color=INK)
    for gname, y, _ in heads:
        ax.text(X_HEAD, y, gname, ha='left', va='center', fontsize=SZ_SECTION,
                fontweight='bold', color=INK)

    for ci, (cond, _) in enumerate(CONDS):
        for ki, (key, _) in enumerate(CONTRASTS):
            x = xs[ci][ki]
            for f, _, y, _ in rows:
                r = df.loc[(f, cond, key)]
                d, p, q = float(r.cohen_d), float(r.p), float(r.p_fdr)
                col = DIR_C['neg'] if d < 0 else DIR_C['pos']
                if q < ALPHA:
                    alpha = ALPHA_FDR
                elif p < ALPHA:
                    alpha = ALPHA_NOM
                else:
                    alpha = ALPHA_NS
                ax.scatter([x], [y], s=marker_area(d), facecolor=col,
                           edgecolor='none', alpha=alpha, zorder=3)
                if q < ALPHA:
                    ring(ax, x, y, marker_area(d), col)
                elif p < ALPHA:
                    ring(ax, x, y, marker_area(d), col, nominal=True)

    # ── header ───────────────────────────────────────────────────────────────
    for name, c0, c1 in STIM_SPAN:
        ax.text((xs[c0][0] + xs[c1 - 1][-1]) / 2, Y_TOP, name, ha='center',
                va='center', fontsize=SZ_STIM, fontweight='bold', color=INK)
    for ci, (_, lbl) in enumerate(CONDS):
        xa, xb = xs[ci][0], xs[ci][-1]
        ax.text((xa + xb) / 2, y_cond, lbl, ha='center', va='center',
                fontsize=SZ_COND, color=INK)
        half = COL_PITCH / 2
        ax.plot([xa - half, xb + half], [y_rule, y_rule], color=RULE_C,
                lw=LW_RULE, solid_capstyle='butt')
        for ki, (_, sub) in enumerate(CONTRASTS):
            ax.text(xs[ci][ki], y_sub, sub, ha='center', va='center',
                    fontsize=SZ_SUB, color=INK)

    # divider: organises, does not compete — starts under the header, stops at
    # the last data row, never enters the legend
    x_div = (xs[1][-1] + xs[2][0]) / 2
    ax.plot([x_div, x_div], [y_rule, y_last + BAND_PAD], color=RULE_C,
            lw=LW_RULE, solid_capstyle='butt')
    return y_last


def draw_legend(ax, y_last):
    xs = column_x()
    LEG_X = [LEG_X_MARGIN, xs[0][0], xs[2][0]]
    y0 = y_last + GAP_DATA_LEGEND
    titles = ['Effect size |d|', 'Statistical significance', 'Direction']
    for x, t in zip(LEG_X, titles):
        ax.text(x, y0, t, ha='left', va='center', fontsize=SZ_LEGEND_TITLE,
                fontweight='bold', color=INK)

    def entry(col, i, label):
        return LEG_X[col], y0 + LEG_TITLE_ROW + i * LEG_ROW, label

    for i, d in enumerate([0.2, 0.5, 0.8, 1.5]):
        x, y, lab = entry(0, i, f'|d| = {d}')
        ax.scatter([x + LEG_SWATCH_DX], [y], s=marker_area(d),
                   facecolor='#BFC5CC', edgecolor='none', zorder=3)
        ax.text(x + LEG_TEXT_DX, y, lab, ha='left', va='center',
                fontsize=SZ_LEGEND, color=INK)

    ev = [(ALPHA_FDR, f'survives BH FDR (q < {ALPHA})'),
          (ALPHA_NOM, f'nominal p < {ALPHA}'),
          (ALPHA_NS, f'p ≥ {ALPHA}')]
    for i, (a, lab) in enumerate(ev):
        x, y, _ = entry(1, i, lab)
        ax.scatter([x + LEG_SWATCH_DX], [y], s=marker_area(0.8),
                   facecolor=DIR_C['pos'], edgecolor='none', alpha=a, zorder=3)
        if a == ALPHA_FDR:
            ring(ax, x + LEG_SWATCH_DX, y, marker_area(0.8), DIR_C['pos'])
        elif a == ALPHA_NOM:
            ring(ax, x + LEG_SWATCH_DX, y, marker_area(0.8), DIR_C['pos'],
                 nominal=True)
        ax.text(x + LEG_TEXT_DX, y, lab, ha='left', va='center',
                fontsize=SZ_LEGEND, color=INK)

    dirs = [(DIR_C['neg'], 'd < 0 — lower at the first-named timepoint'),
            (DIR_C['pos'], 'd > 0 — higher at the first-named timepoint')]
    for i, (c, lab) in enumerate(dirs):
        x, y, _ = entry(2, i, lab)
        ax.scatter([x + LEG_SWATCH_DX], [y], s=marker_area(0.8), facecolor=c,
                   edgecolor='none', zorder=3)
        ax.text(x + LEG_TEXT_DX, y, lab, ha='left', va='center',
                fontsize=SZ_LEGEND, color=INK)


def main():
    S.apply_rcparams()
    plt.rcParams.update({
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
        'savefig.facecolor': SURFACE, 'text.color': INK,
        # mathtext in the same family as everything else, so the italic f in the
        # tremor rows is Arial italic and not DejaVu
        'mathtext.fontset': 'custom',
        'mathtext.rm': S.FAMILY, 'mathtext.it': f'{S.FAMILY}:italic',
        'mathtext.bf': f'{S.FAMILY}:bold',
    })
    df = load()
    dmax_obs = float(df.cohen_d.abs().max())
    if dmax_obs > DMAX:
        raise SystemExit(f'|d| max {dmax_obs:.2f} exceeds the DMAX cap {DMAX}')

    fig = plt.figure(figsize=(PAGE_W / 25.4, PAGE_H / 25.4), facecolor=SURFACE)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    y_last = draw(ax, df)
    draw_legend(ax, y_last)

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f'{STEM}.png', dpi=PNG_DPI, facecolor=SURFACE)
    fig.savefig(OUT / f'{STEM}.tiff', dpi=PNG_DPI, facecolor=SURFACE,
                pil_kwargs={'compression': 'tiff_lzw'})
    plt.close(fig)
    print(f'wrote {OUT / STEM}.png  ({PAGE_W:.0f} x {PAGE_H:.0f} mm)')


PNG_DPI = 900   # the smallest marks are ~1.8 mm dots; they alias at 300


if __name__ == '__main__':
    main()
