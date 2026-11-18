"""Summary plate for the ICC(2,1) results.

The results table (icc_all_pairwise_bilateral.csv) holds all 156 cells, every
estimate and interval.  A reader who wants a specific number should go there.
What a table cannot show is a *pattern* over many cells, for example whether
reliability differs between stimulation off and on across features.

The plot puts that question on the panel split: one panel per stimulation
state, and inside each, two segments per feature for the two session sets (M3-M6,
and everything involving the at-home session).  A segment runs from the smallest
to the largest estimate it covers, so its length is the spread over the cells it
pools.

Stimulation on the panels rather than on colour, so colour carries the session
set alone and a row holds two segments instead of four.  Reading down a panel is
then uncluttered, and the level (position) and the session spread are read inside
one axis; the cost is that OFF versus ON is a comparison across two separated
axes rather than a vertical offset inside one row.  A single-axis variant with
all four segments per feature was dropped: it put both factors on colour and the
row became four near-parallel lines to disentangle.

No n is drawn here.  Peak parameters are conditional on a beta peak being
detected at both timepoints, so they rest on fewer patients than every other
feature; read `n_subjects` in the CSV beside any peak row.

Nothing is averaged across session pairs or conditions: n is 4-9, several cells
are carried by a single patient, and a mean over three pairs would hide exactly
the spread this figure exists to show.

Output: Results/Figures/fig_icc_summary_stim_{bilateral,L,R}.png (+ .tiff)
Values read from Results/stats/icc/icc_all_pairwise_{side}.csv; none estimated.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle

import table_figure_style as S

ICC_CSV = 'Results/stats/icc/icc_all_pairwise_{side}.csv'
SIDES = ('bilateral', 'L', 'R')   # one plate per analysis unit
OUT = Path('Results/Figures')

# ── Rows: same grouping, order and wording as the result tables ─────────────
BANDS = ['delta', 'theta', 'alpha', 'beta']
BAND_NAME = {'delta': 'Delta', 'theta': 'Theta', 'alpha': 'Alpha', 'beta': 'Beta'}

GROUPS = [
    ('Absolute band power', [('abs_' + b, BAND_NAME[b]) for b in BANDS]),
    ('Tremor-frequency power',
     [('abs_tremor_f0', r'$f_0$ power'),
      ('abs_tremor_2f', r'$2f_0$ power')]),
    ('Aperiodic features',
     [('exponent', 'Exponent'),
      ('offset', 'Offset'),
      ('broadband_power', 'Broadband power')]),
    ('Beta peak features',
     [('peak_beta_cf', 'Centre frequency'),
      ('peak_beta_pw', 'Peak power'),
      ('peak_beta_bw', 'Bandwidth'),
      ('flat_abs_beta', 'Aperiodic-corrected power')]),
]

# Koo & Li boundaries.  Drawn as bands rather than gridlines because the text
# argues in tiers, so the reader must be able to place a dot in a tier by eye.
# Blended ~30% toward the table tints: enough for the segments to sit clearly
# in front, not so far that the tier ladder stops reading as four steps.
TIERS = [(-0.15, 0.50, 'poor', '#F7F8F9'), (0.50, 0.75, 'moderate', '#EFF4FB'),
         (0.75, 0.90, 'good', '#ECF5F0'), (0.90, 1.00, 'exc.', '#DDEFE6')]

# Ink and type scale are taken from make_trajectory_figure.py rather than chosen
# here, so the two plates can sit in the same manuscript without one looking
# like a different document: charcoal ink, grey for secondary annotation, and
# the same role -> size mapping off table_figure_style's constants.
INK = '#222222'            # text, rules
MUTED = '#6B7280'          # ticks and secondary annotation
SURFACE = '#ffffff'

SZ_SECTION = S.SIZE_COND - 2.0      #  9.0  group headings
SZ_LABEL = S.SIZE_LABEL             #  8.2  feature labels
SZ_TITLE = S.SIZE_HEADER - 0.8      #  8.2  column / panel headers
SZ_TICK = S.SIZE_VALUE - 1.6        #  5.8  axis tick labels
SZ_LEGEND = S.SIZE_LEGEND           #  7.6  legend
SZ_ANNOT = S.SIZE_CI + 1.6          #  6.5  tier names

# Delivered width, in mm.  Not S.TARGET_W_MM: these plates are published beside
# fig_rest_mvt_panels_bilateral.png, which is sized by a hardcoded figsize and
# comes out at 233 mm.  Two figures placed at the same page width are scaled by
# the ratio of their delivered widths, so a plate drawn at 170 mm next to one
# drawn at 233 mm prints its type ~1.37x larger than its neighbour's despite
# both declaring the same point sizes.  Matching the width makes both shrink by
# the same factor and the two plates carry one apparent type scale.
TARGET_W_MM = 185.0
TARGET_W_IN = TARGET_W_MM / 25.4

# Width and height are scaled together off S.TARGET_W_MM's layout rather than
# the width alone: every position in this figure is in data units, so widening
# without the matching height would stretch the panels against fixed-point text
# and change the design instead of just its size.
SCALE = TARGET_W_MM / S.TARGET_W_MM

# Inches of declared figure height per y data unit, measured off the layout this
# plate was tuned at.  It fixes the visual size of a row, so the type scale does
# not drift when the row pitches change.
IN_PER_Y = 0.3492

# bbox_inches='tight' crops to the drawn artists, so the delivered width is a
# fixed fraction of the declared figsize for a given layout.  Measured once; dividing the target by it makes the plate come out at TARGET_W_MM.
TIGHT_CROP = 0.7922


# The split the results text makes: the two clinical visits against each other,
# and everything involving the at-home session.  Clinic-home pools M3-home and
# home-M6, so a segment there spans 4 cells (2 session pairs x 2 behavioural
# conditions) against M3-M6's 2.
SESSIONS = [('M3–M6', ['M3 vs M6']),
            ('Clinic–home', ['M3 vs home', 'home vs M6'])]

# Rest and movement are pooled into a range rather than drawn as two marks:
# the behavioural contrast is not what this figure is about, and collapsing it
# leaves one uncluttered segment per stimulation state whose length is the
# spread across the two behavioural conditions.  Each stimulation state gets its
# own sub-line inside the feature row so the two segments never overlap.
STIMS = [('stimOFF', 'Stim OFF'), ('stimON', 'Stim ON')]

# The panel split carries stimulation, so hue is free to carry the session set.
# Orange and teal separate well at 1.6 pt line width and hold up under the common
# colour-vision deficiencies, which matters here because the two segments in a
# row are often within 0.1 of each other.
#
# Read the warm/cool split with care: COND_COLORS in stats_analysis/plot_style.py
# uses warm = stim OFF and cool = stim ON, and this plate uses the same two hue
# families for something else.  The panel headings are the authority on stim, and
# colour never encodes it here.
SESS_COLOUR = {'M3–M6': '#E17C05',        # orange
               'Clinic–home': '#38A6A5'}  # teal

# Two sub-lines per feature, so the two session sets never overlap inside a row.
SESS_DY = {'M3–M6': 0.16, 'Clinic–home': -0.16}

# The two conditions pooled into each segment, in table order.
STIM_CONDS = {'stimOFF': ['rest_stimOFF', 'mvt_stimOFF'],
              'stimON': ['rest_stimON', 'mvt_stimON']}

# X0 sits just below the lowest tier edge (-0.15), not lower: the panel is only
# painted from that edge, so any extra axis below it became unpainted lead-in on
# the left of every panel — width spent on a region no unclipped estimate reaches
# and which read as a wider inter-panel gap than PANEL_GAP declares.
X0, X1 = -0.16, 1.02      # ICC range drawn inside each condition panel
PANEL_W = X1 - X0         # data width of one panel
# Everything here is in data units and the delivered width is fixed, so the
# gutters and the panels compete for one budget: shrinking a gap widens both
# panels by the same amount.  PANEL_GAP only has to clear the 1.00 tick label of
# the panel on its left; R_PAD only has to clear the markers at the right edge.
PANEL_GAP = 0.12          # between the two stimulation panels
R_PAD = 0.10              # right panel edge -> axis edge
X_LABEL = -1.08           # left edge of the feature-label gutter


def _clip(icc: float) -> float:
    """Pin an off-scale estimate to the panel edge.

    Five cells (all peak parameters, minimum -1.19) fall below X0.  Extending
    the axis to hold them would spend most of every panel on those five and
    compress the 0.5-1.0 region where the differences that matter live, so they
    are drawn at the edge and flagged in the footnote instead.
    """
    return min(max(icc, X0 + 0.01), X1)


SEG_LEN = 0.22        # legend swatch length, in x data units
SEG_TEXT_GAP = 0.07   # swatch -> its label
ENTRY_GAP = 0.34      # between two entries on the same legend row


def _draw_legend(ax, fig, legend_rows, y0, dy, x_centre):
    """Draw legend rows centred on x_centre.

    Label widths are measured off the renderer rather than assumed: the axis is
    in ICC units, so a string has no fixed width in data coordinates and any
    hand-set column pitch drifts as soon as the font size or figure width moves.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    for k, entries in enumerate(legend_rows):
        y = y0 - k * dy
        widths = []
        for _, label in entries:
            probe = ax.text(0, y, label, fontsize=SZ_LEGEND)
            bb = probe.get_window_extent(renderer=renderer)
            (xa, _), (xb, _) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y0)])
            widths.append(xb - xa)
            probe.remove()
        total = (sum(SEG_LEN + SEG_TEXT_GAP + w for w in widths)
                 + ENTRY_GAP * (len(entries) - 1))
        x = x_centre - total / 2
        for (colour, label), w in zip(entries, widths):
            ax.plot([x, x + SEG_LEN], [y] * 2, color=colour, lw=1.6, zorder=6)
            for xe in (x, x + SEG_LEN):
                ax.plot(xe, y, 'o', ms=4.4, mfc=colour, mec='#FFFFFF', mew=0.7,
                        clip_on=False, zorder=6)
            ax.text(x + SEG_LEN + SEG_TEXT_GAP, y, label, ha='left', va='center',
                    fontsize=SZ_LEGEND, color=INK)
            x += SEG_LEN + SEG_TEXT_GAP + w + ENTRY_GAP


def load(side='bilateral'):
    d = pd.read_csv(ICC_CSV.format(side=side))
    d['stim'] = d.condition.str.split('_').str[1]
    return d


def summarise(d: pd.DataFrame, feat: str) -> dict:
    s = d[d.feature == feat]
    # cells[(condition, pair)] -> ICC.  Missing combinations stay absent rather
    # than becoming NaN, so a gap in the plate means "no estimate", never 0.
    cells = {(r.condition, r.pair): r.ICC for r in s.itertuples()}
    return {'cells': cells}


# Vertical rhythm, in the same data units as everything else.  The hierarchy is
# the point: a family reads as a block, so the gap that separates two families
# has to beat the gap between two rows inside one, and the heading has to sit
# closer to the rows it labels than to the family above.  ROW_PITCH stays wide
# enough for the two sub-lines (+-SESS_DY) plus their markers.
RULE_DROP = 0.30    # family heading -> the rule under it
ROW_PITCH = 0.95    # feature row -> feature row, inside a family
HEAD_GAP = 0.80     # family heading -> its first feature row
FAMILY_GAP = 0.30   # added after a family, so heading-to-heading clears by
                    # ROW_PITCH + FAMILY_GAP = 1.25 against 0.95 within


def build(d: pd.DataFrame):
    rows, y = [], 0.0
    for gname, feats in GROUPS:
        rows.append({'kind': 'group', 'label': gname, 'y': y})
        y -= HEAD_GAP
        for key, label in feats:
            rows.append({'kind': 'feat', 'key': key, 'label': label, 'y': y,
                         **summarise(d, key)})
            y -= ROW_PITCH
        y -= FAMILY_GAP
    return rows


def draw(rows):
    """One panel per stimulation state, two segments per feature."""
    S.apply_rcparams()
    plt.rcParams.update({'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
                         'savefig.facecolor': SURFACE, 'text.color': INK})
    n_panel = len(STIMS)
    x_right = (n_panel - 1) * (PANEL_W + PANEL_GAP) + PANEL_W + R_PAD
    fig_w = TARGET_W_IN / TIGHT_CROP

    y_last = min(r['y'] for r in rows if r['kind'] == 'feat') - 0.52
    y_bottom, y_top = y_last - 1.05, 2.42
    # Height is solved from the y range rather than counted off the rows, so a
    # change to ROW_PITCH or FAMILY_GAP moves the paper height with it and the
    # data units keep a fixed size in inches — point sizes and marker sizes then
    # stay in the same proportion to a row as they are here.
    fig_h = (y_top - y_bottom) * IN_PER_Y * SCALE
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    ax.set_xlim(X_LABEL, x_right)
    ax.set_ylim(y_bottom, y_top)
    ax.axis('off')

    def px(icc: float, i: int) -> float:
        return i * (PANEL_W + PANEL_GAP) + (icc - X0)

    sess_pairs = {name: pairs for name, pairs in SESSIONS}

    for i, (_, slabel) in enumerate(STIMS):
        for lo, hi, tname, colour in TIERS:
            ax.add_patch(Rectangle((px(max(lo, X0), i), y_last),
                                   px(hi, i) - px(max(lo, X0), i), 1.42 - y_last,
                                   facecolor=colour, edgecolor='none', zorder=0))
            if tname in ('poor', 'moderate', 'good'):
                ax.text(px((max(lo, X0) + hi) / 2, i), 1.62, tname, ha='center',
                        va='center', fontsize=SZ_ANNOT, color=MUTED, zorder=3)
        for lo, _, _, _ in TIERS[1:]:
            ax.plot([px(lo, i)] * 2, [y_last, 1.42], color='#FFFFFF', lw=0.9, zorder=1)
        ax.plot([px(0, i)] * 2, [y_last, 1.42], color='#C7CDD4', lw=0.8,
                ls=(0, (3, 3)), zorder=2)
        for t in (0.0, 0.25, 0.5, 0.75, 1.0):
            ax.text(px(t, i), 1.22, f'{t:.2f}', ha='center', va='center',
                    fontsize=SZ_TICK, color=MUTED, zorder=3)
        ax.text(px((X0 + 1.0) / 2, i), 2.05, slabel, ha='center', va='center',
                fontsize=SZ_TITLE, fontweight='bold', color=INK)

    for r in rows:
        y = r['y']
        if r['kind'] == 'group':
            ax.text(X_LABEL, y, r['label'], ha='left', va='center',
                    fontsize=SZ_SECTION, fontweight='bold', color=INK)
            ax.plot([X_LABEL, x_right], [y - RULE_DROP] * 2,
                    color=S.FAINT, lw=0.55, zorder=2)
            continue

        ax.text(X_LABEL + 0.15, y, r['label'], ha='left', va='center',
                fontsize=SZ_LABEL, color=INK, zorder=4)

        for i, (stim, _) in enumerate(STIMS):
            ax.plot([px(X0, i), px(X1, i)], [y, y], color='#FFFFFF', lw=0.7,
                    alpha=0.85, zorder=2.5)
            for sess, _ in SESSIONS:
                vals = [r['cells'][(c, pr)] for c in STIM_CONDS[stim]
                        for pr in sess_pairs[sess] if (c, pr) in r['cells']]
                if not vals:
                    continue
                colour = SESS_COLOUR[sess]
                dy = SESS_DY[sess]
                lo, hi = _clip(min(vals)), _clip(max(vals))
                ax.plot([px(lo, i), px(hi, i)], [y + dy] * 2, color=colour,
                        lw=1.6, solid_capstyle='round', zorder=5)
                for xe in (lo, hi):
                    ax.plot(px(xe, i), y + dy, 'o', ms=4.4, mfc=colour,
                            mec='#FFFFFF', mew=0.7, zorder=6, clip_on=False)

    _draw_legend(ax, fig, [[(SESS_COLOUR[sess], sess) for sess, _ in SESSIONS]],
                 y_last - 0.70, 0.42,
                 (px(X0, 0) + px(X1, n_panel - 1)) / 2)
    return fig


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for side in SIDES:
        fig = draw(build(load(side)))
        stem = f'fig_icc_summary_stim_{side}'
        fig.savefig(OUT / f'{stem}.png', dpi=S.PNG_DPI,
                    facecolor=SURFACE, bbox_inches='tight')
        fig.savefig(OUT / f'{stem}.tiff', dpi=S.PNG_DPI, facecolor=SURFACE,
                    bbox_inches='tight', pil_kwargs={'compression': 'tiff_lzw'})
        plt.close(fig)
        print(f'wrote {OUT / f"{stem}.png"}')


if __name__ == '__main__':
    main()
