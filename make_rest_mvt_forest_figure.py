"""Panel B of the rest-vs-movement figure: the Movement − Rest effect per feature.

NOT A SCRIPT. This module draws Panel B into a cell of a figure it does not own,
through draw_into(); make_rest_mvt_combined_figure.py is the only entry point and
the only thing that writes a file. The standalone fig_rest_mvt_forest_* output
and its CLI were removed — the panel is only ever published beside Panel A, and
two ways of producing it meant two things to keep in step.

Reads Results/stats/lme_rest_mvt/simple_effects_{side}.csv (written by
stats_analysis/lme_rest_mvt_diff.py) and draws one dot-and-interval column per
feature x stim state, against a zero reference.

What the plot shows
-------------------
Each point is the movement effect (mvt − rest, log ratio) within one stim state,
averaged over the three timepoints, with its 95% CI. The zero line is the
no-movement-effect null, so an interval clear of zero is the readable claim.
Pairing the two stim states on one row makes the interaction visible as the gap
between the filled and open markers — the same quantity the model reports as
`x stim ON`, but shown rather than tabulated.

Encoding
--------
Stim state is carried TWICE — by hue and by marker fill (solid = stim OFF, hollow
= stim ON) — so identity never rests on colour alone and the figure still reads
in greyscale or photocopy.

The hues keep the project's stim semantics from stats_analysis/plot_style.py —
warm = stim OFF, cool = stim ON — with brick red (#CC503E)
as the warm pole rather than the orange used elsewhere. Validated against a light
surface: CVD separation ΔE 14.2 (protan) / 30.6 (tritan), and unlike the orange
both hues clear 3:1 contrast outright.

Value text stays in ink, never in the series colour: the marker beside a row
carries identity, and coloured numerals would encode it twice at the cost of
legibility. Ink, type sizes and geometry come from table_figure_style.py, shared
with the table figures.

Significance marks — ink is correction, position is question
-------------------------------------------------------------
Three things are being marked and they carry different evidential weight, so
they must not look alike. Position alone was doing that job and it was not
enough: a reader who counted asterisks read 8 findings from a panel claiming 2.

    ink       BLACK * survived FDR correction   GREY * nominal p < 0.05 only
    position  above an interval: the movement effect WITHIN that stim state
              after a feature name: the movement x stim INTERACTION

Ink rather than hue, so the distinction survives a greyscale print and does not
depend on colour vision. It also stays clear of C_OFF, the brick red already
carrying 'stimulation OFF' on every marker — all four corrected within-condition
marks sit above stim-OFF intervals, and a red mark there would read as part of
the stim encoding. And it needs no key to guess at: the fainter mark is the
weaker evidence.

ABOVE AN INTERVAL: an EXACT sign-flip permutation p — `p_perm` — corrected by BH
across features within its own stimulation state; the bootstrap achieved
significance level (`p_boot`) is used only where no permutation p was written.
`p_perm` is the whole-patient sign-flip of the timepoint-averaged contrast
within that stimulation state. Because the permutation and the plotted bootstrap
interval are different constructions, a grey mark and its interval can disagree
at the margin — the same p-vs-CI split the caption already carries for the
corrected mark. Always above, never below, so it is read without being traced
back to its own bar first.

AFTER A FEATURE NAME: the 'x stim ON' interaction, black when it survives BH and
grey when only the permutation p clears. The name is set bold for the corrected
case only — redundant with the ink on purpose, so the strongest claim survives a
print where a lone asterisk is easy to miss. The mark is drawn separately from
the tick label so the ink lands on the asterisk alone and a grey mark never
leaves the feature name looking de-emphasised. Putting it on the name rather
than in a row of its own keeps the result next to the feature it belongs to; as
a separate row it was the furthest thing on the panel from its own data.

No ***/**/* ladder anywhere. The bootstrap p floors at 1/n_boot, where
several features can tie, and the intercept's permutation p is granular
to 1/2^9 — a graded scale would claim resolution the resampling does not have.

Alpha is the case to watch: a grey mark above its stimOFF interval, a black one
after its name. The interaction survives correction while the within-condition
effect does not. Those are different questions and the panel shows the
disagreement rather than hiding it.

The caption must define both inks and both positions. Without it the grey mark
reads as a faded black one rather than as a weaker claim.

CIs are the cluster bootstrap over patients where the analysis produced them
(resampling whole patients, so within-patient dependence is preserved), and are
left empty otherwise — there is no Wald fallback.

Title and footnote are off by default — the figure is built to sit under a
caption that carries the model, the CI method and the negative-control point.

"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import table_figure_style as S

STATS_DIR = Path('Results') / 'stats' / 'lme_rest_mvt'

# Project stim semantics kept — warm = stim OFF, cool = stim ON, as in plot_style
# — but the warm pole is the darker brick red rather than the
# orange: orange sits at 2.89:1 against white, under the 3:1 guideline, whereas
# this pair clears every check outright (CVD dE 14.2 protan / 30.6 tritan,
# contrast PASS on both).
C_OFF = '#D46A4C'   # warm orange-red — stim OFF
C_ON  = '#4A78A8'   # muted blue      — stim ON

# Row order follows the ICC / LME table figures, so the figures list the same
# features in the same order. The LABELS are abbreviated here and only here:
# this panel sits beside the spectra panel in make_rest_mvt_combined_figure.py,
# where the name gutter is roughly a third the width it gets standalone. The
# table figures keep the long forms, so a reader moving between them sees
# 'Aperiodic power' in one and 'Broadband power' in the other.
GROUPS = [
    ('Band power',                             [
        ('abs_delta', 'Delta'), ('abs_theta', 'Theta'),
        ('abs_alpha', 'Alpha'), ('abs_beta',  'Beta')]),
    ('Tremor-locked power',                     [
        ('abs_tremor_f0', r'$f_0$ power'), ('abs_tremor_2f', r'$2f_0$ power')]),
    ('Aperiodic features',                      [
        ('exponent', 'Exponent'), ('offset', 'Offset'),
        ('broadband_power', 'Aperiodic power')]),
    ('Beta peak',                               [
        ('peak_beta_cf', 'Peak freq.'), ('peak_beta_pw', 'Peak power'),
        ('peak_beta_bw', 'Bandwidth'), ('flat_abs_beta', 'Aperiodic-\ncorrected beta')]),
]

# Diff carries whatever scale the feature was already on. In the main 'ratio'
# panel, log10 features are rescaled to natural log (a pure unit change,
# ln10 = 2.3026) so they share the axis with the ln features; cf and bw are
# frequency shifts in Hz and CANNOT share either axis, so they get their own
# panel rather than being silently plotted 2.3x or 100x out of scale.
LN10 = np.log(10.0)

# Features given a panel of their own even though their unit is log-ratio like the
# first panel's. Flat beta is a FOOOF residual roughly 3x the magnitude of every
# other log-ratio feature (-0.67 against a +/-0.3 spread), so on a common axis it
# sets the scale and squeezes the band-power rows it exists to be compared
# against. Peak power joins it: both are periodic (aperiodic-removed) beta
# measures, so they belong together. Both are log10; since this panel holds no
# ln feature to share an axis with, _scale leaves them in native log10 rather
# than rescaling to ln — which also matches how they are back-transformed
# everywhere else in the pipeline (10**coef, never exp(coef)).
#
# Merging these into the first panel was tried and reverted: it forced that
# panel's limits out to -1.2 to hold flat beta's interval, which left the eight
# features the analysis is mostly about compressed into the middle third of the
# axis and shrank the exponent interval to near-invisible.
SEPARATE_PANEL_FEATURES = {'flat_abs_beta', 'peak_beta_pw'}

# Panels are drawn in this order.
PANEL_ORDER = ['ratio', 'flat', 'hz']

DX        = 0.17    # horizontal offset of each stim mark from the feature centre
# The estimate is the reading; the interval is its uncertainty. Keeping the marker
# clearly heavier than its bar puts the eye on the point first and stops a long
# CI from out-weighting a short one purely by ink.
MS        = 7.4     # marker size (points) — well above the 8px floor at 300 dpi
MS_LEGEND = 5.5      # legend swatch only; data points keep MS
LW_CI     = 1.1
LW_ZERO   = 0.7     # the zero line is a reference, not a data mark
ZERO_COL  = '#CBD5E1'   # desaturated blue-grey; identical in every panel
# Column highlighting is off: the stars, the Stim x movement row and the interval
# positions already carry the emphasis, and a background band competed with them.
# Set True to restore the tint (_is_highlight still selects the columns).
SHOW_HIGHLIGHT = False
HILITE         = '#F4F6F8'

PANEL_TITLE = 'Movement effect by feature and stimulation state'

# Fixed effect-axis limits, by unit kind. Without one the axis is solved from the
# data, so the log-ratio panel's scale moves whenever a CI does and two runs of
# the figure cannot be compared by eye. Only the panels named here are pinned;
# the rest still auto-scale.
PANEL_YLIM = {'ratio': (-0.5, 0.5), 'hz': (-5.0, 5.0)}

# Horizontal grid: the effect axis is the one being read off, and with the
# panels on three different scales a rule at each tick is what lets a reader
# judge a distance without tracking back to the axis. Kept pale enough that it
# never competes with an interval.
GRID_COL   = '#E8EDF1'
GRID_LW    = 0.6

COL_PITCH  = 0.92   # within a feature family
GAP_FAMILY = 0.85   # extra space between families — with the family names
                    # gone this gap is the only thing marking them
GAP_INCHES = 0.78   # physical gap between panels, so the y labels clear

def load_side(side):
    """(simple effects, interaction coefficients, path). The second file supplies
    the Stim x movement column, whose stars come from the FDR-corrected
    interaction term — the inferential claim of this analysis."""
    se_p   = STATS_DIR / f'simple_effects_{side}.csv'
    coef_p = STATS_DIR / f'lme_rest_mvt_diff_{side}.csv'
    if not se_p.exists():
        return None, None, se_p
    se = pd.read_csv(se_p)
    coef = pd.read_csv(coef_p) if coef_p.exists() else None
    return se, coef, se_p


def _ci(row):
    """Cluster-bootstrap CI. Returns (lo, hi, is_boot).

    There is no Wald fallback: lme_rest_mvt_diff reports no Wald quantity, so a
    missing interval means the bootstrap was skipped (--no-boot) and the row has
    no interval to draw rather than a different one to substitute.
    """
    lo, hi = row.get('ci_lo', np.nan), row.get('ci_hi', np.nan)
    if pd.notna(lo) and pd.notna(hi):
        return float(lo), float(hi), True
    return np.nan, np.nan, False


# The glyph carries correction status, the position carries what was tested:
#   *  above an interval   within-condition effect, survived BH in its family
#   †  above an interval   within-condition effect, nominal only
#   *  after a feature name  the movement x stim interaction, survived BH
# A single mark, never a ***/**/* ladder: the bootstrap p floors at 1/n_boot
# (several features can tie there) and the intercept's permutation p is
# granular to 1/2^9, so a graded scale would claim resolution the resampling
# does not have.
MARK = '*'

# Both marks are asterisks; the ink separates them. A luminance difference, not
# a hue one: it survives a greyscale print, does not depend on colour vision,
# and stays clear of C_OFF (#CC503E), which already carries 'stimulation OFF'
# for every marker on the panel — a red mark above a red interval reads as part
# of the stim encoding rather than as significance. The fainter ink also needs
# no key: it is visibly the weaker evidence.
# S.INK and S.MUTED are both pure black, hence the local grey.
COLOR_CORRECTED   = S.INK
COLOR_UNCORRECTED = '#7c8792'

# Vertical offset, in points below the axis, of the interaction mark drawn beside
# a feature name. Matches matplotlib's default x-tick pad so the mark sits on the
# same 45° baseline as the label it follows.
MARK_PAD_PT = 3.5

# Marks are set larger than the value text they used to inherit (SIZE_VALUE,
# 7.4pt) and always bold. An asterisk is mostly whitespace at small sizes, so it
# reads faint next to a solid marker even in black — and the grey one, which has
# less contrast to spend, disappears first. Weight is not part of the
# corrected/uncorrected encoding: every mark is bold, and only the feature NAME
# goes bold, for the corrected interaction.
MARK_FONTSIZE = 11.0

# Whitespace appended to a marked feature name to clear the mark drawn over it.
# Tied to MARK_FONTSIZE by eye: a space is ~0.28em at SIZE_LABEL, so four of
# them cover an 11pt bold asterisk with a hair of separation left over.
MARK_GAP = '    '
# A label containing $...$ is laid out by mathtext, which drops trailing ASCII
# spaces (and \quad) from the width it right-aligns on, so the mark would land
# on the name. Unicode em + thin space survive it: ~1.2em, as the four above.
MARK_GAP_MATH = '\u2003\u2009'


def _mark(p_corrected, p_raw, alpha=0.05):
    """(glyph, colour). Corrected wins; grey is what is left when only p_raw clears."""
    if p_corrected is not None and not pd.isna(p_corrected) \
            and float(p_corrected) < alpha:
        return MARK, COLOR_CORRECTED
    if p_raw is not None and not pd.isna(p_raw) and float(p_raw) < alpha:
        return MARK, COLOR_UNCORRECTED
    return '', COLOR_CORRECTED


def stars(row):
    """
    Significance marker for a simple effect.

    The raw p is the exact sign-flip permutation (`p_perm`) when present, else
    the bootstrap achieved significance level (`p_boot`). q_fdr is that p
    corrected by BH across features within its own stimulation state.

    Returns (glyph, colour). A feature carries at most one mark here: a black
    asterisk when it survives correction, a grey one when only the uncorrected p
    clears. Alpha is the case worth watching — grey above its stimOFF interval
    and black after its name, because the interaction survives correction while
    the within-condition effect does not. Those answer different questions; the
    figure shows the disagreement rather than resolving it.
    """
    # No Wald fallback. The exact sign-flip permutation p is the reported
    # quantity; p_boot covers a --no-perm preview only. If neither was computed
    # (--no-perm --no-boot), no star is drawn rather than substituting a normal
    # approximation this module rejects everywhere else.
    p = row.get('p_perm', np.nan)
    if pd.isna(p):
        p = row.get('p_boot', np.nan)
    if pd.isna(p):
        return '', COLOR_CORRECTED
    return _mark(row.get('q_fdr', np.nan), p)


def interaction_stars(coef, feature):
    """
    (glyph, colour) for the 'Stim ON vs OFF' interaction of one feature.

    Black when it survives BH across the 13 features, grey when only the
    uncorrected permutation p clears — the same ink rule the within-condition
    marks use. A grey mark is an interaction significant on its own but lost
    once the 13 looks are accounted for; marking it makes that visible instead
    of leaving it to the coefficient table.
    """
    if coef is None:
        return '', COLOR_CORRECTED
    r = coef[(coef['feature'] == feature) & (coef['term'] == 'x stim ON')]
    if r.empty:
        return '', COLOR_CORRECTED
    row = r.iloc[0]
    return _mark(row.get('p_fdr', np.nan),
                 row.get('p_used', row.get('p_perm', np.nan)))


def _unit_kind(df, feature):
    """
    Which panel a feature belongs to.

    'hz' is a hard split — a frequency shift cannot share an axis with a log
    ratio. 'flat' is a soft one: the same unit as 'ratio', separated only so that
    one large-magnitude feature does not set the scale for all the others.
    """
    sub = df[df['feature'] == feature]
    if sub.empty:
        return 'ratio'
    if str(sub.iloc[0].get('unit', 'ln')) == 'Hz':
        return 'hz'
    return 'flat' if feature in SEPARATE_PANEL_FEATURES else 'ratio'


def _scale(row, value):
    """
    Put a value on the panel's common scale.

    The FOOOF offset is log10 but shares the main 'ratio' panel with ln
    features (abs_delta, exponent, ...), so it is rescaled to natural log to
    sit on that shared axis — ln10 = 2.3026, a unit change, not a transform of
    the estimate. flat_abs_beta and peak_beta_pw are also log10, but live
    alone on their own 'flat' panel (SEPARATE_PANEL_FEATURES) with no ln
    feature to share it with, so they are left in their native log10 — that
    also matches how they are back-transformed everywhere else in the
    pipeline (10**coef, never exp(coef)).
    """
    if row.get('feature') in SEPARATE_PANEL_FEATURES:
        return value
    return value * LN10 if str(row.get('unit', 'ln')) == 'log10' else value


def _is_highlight(df, coef, feature):
    """
    Rows worth the reader's eye first: either the FDR-corrected interaction is
    significant, or a simple effect is strong (p_boot < .01). Derived from the
    data rather than a hand-typed list, so the emphasis cannot drift out of date
    when the analysis is rerun.
    """
    mark, mark_color = interaction_stars(coef, feature)
    if mark and mark_color == COLOR_CORRECTED:
        return True
    sub = df[df['feature'] == feature]
    if 'p_boot' in sub.columns and (pd.to_numeric(sub['p_boot'],
                                                  errors='coerce') < 0.01).any():
        return True
    return False


def _layout(df, unit_kind):
    """
    Columns with explicit x positions: tight inside a family, looser between
    them, so the gaps do the separating and the group headings can stay quiet.

    Headings sit ABOVE their features now rather than beside them, so unlike the
    horizontal layout they consume no space along the feature axis — their x is
    the centre of the family they name.
    """
    present = set(df['feature'])
    items, x = [], 0.0
    first = True
    for gname, feats in GROUPS:
        keep = [(f, lab) for f, lab in feats
                if f in present and _unit_kind(df, f) == unit_kind]
        if not keep:
            continue
        # The families are no longer named — the extra gap between them is what
        # marks the grouping now, so GROUPS is read for order and for where the
        # breaks fall, not for labels.
        if not first:
            x += GAP_FAMILY
        first = False
        for f, lab in keep:
            items.append(('feat', f, lab, x))
            x += COL_PITCH
    return items


def _draw_panel(ax, df, coef, items, ylim=None):
    """One panel: dot + interval per feature x stim, with group headings."""
    vals = []
    any_boot = False

    for kind, key, label, x in items:
        if kind != 'feat':
            continue
        sub = df[df['feature'] == key]
        for stim, dx, col, face in [('stimOFF', -DX, C_OFF, C_OFF),
                                    ('stimON',  +DX, C_ON,  S.SURFACE)]:
            r = sub[sub['stim'] == stim]
            if r.empty:
                continue
            r = r.iloc[0]
            est = _scale(r, float(r['est']))
            lo, hi, is_boot = _ci(r)
            lo, hi = _scale(r, lo), _scale(r, hi)
            any_boot |= is_boot
            ax.plot([x + dx, x + dx], [lo, hi], color=col, lw=LW_CI,
                    solid_capstyle='round', zorder=3)
            ax.plot([x + dx], [est], marker='o', ms=MS, mfc=face, mec=col,
                    mew=1.5, ls='none', zorder=4)
            vals += [lo, hi]

    span = max(abs(min(vals)), abs(max(vals))) if vals else 0.4
    lim = span * 1.10
    pad = span * 0.045

    if SHOW_HIGHLIGHT:
        for kind, key, label, x in items:
            if kind == 'feat' and _is_highlight(df, coef, key):
                ax.axvspan(x - COL_PITCH * 0.46, x + COL_PITCH * 0.46,
                           color=HILITE, zorder=0, lw=0)

    for kind, key, label, x in items:
        if kind != 'feat':
            continue
        sub = df[df['feature'] == key]
        for stim, dx in [('stimOFF', -DX), ('stimON', +DX)]:
            r = sub[sub['stim'] == stim]
            if r.empty:
                continue
            r = r.iloc[0]
            mark, mark_color = stars(r)
            if not mark:
                continue
            _lo, hi, _ = _ci(r)
            hi = _scale(r, hi)
            # always above the interval, never below: a star that sometimes sat
            # under the bar and sometimes over it had to be traced back to its
            # own interval before it could be read
            ax.text(x + dx, hi + pad, mark, ha='center', va='bottom',
                    fontsize=MARK_FONTSIZE, color=mark_color, fontweight='bold',
                    clip_on=False, zorder=5)

    # zero reference: a guide, not a data mark
    ax.axhline(0.0, color=ZERO_COL, lw=LW_ZERO, zorder=1)

    # The FDR-corrected interaction — the inferential claim of this analysis —
    # rides on the feature name itself: its star follows the name and the name
    # goes bold. It used to be a separate row of marks above the plot, which put
    # the corrected result furthest from the feature it belonged to and needed a
    # rule and a heading of its own to be readable at all. Bold is redundant with
    # the star on purpose, so the emphasis survives a greyscale print where a
    # lone asterisk is easy to miss.
    # The mark is inked independently of the name, so a grey asterisk never
    # makes the feature itself look de-emphasised. Matplotlib cannot colour part
    # of a tick label, so the name is padded with a trailing space to reserve
    # room and the asterisk is drawn separately at the same anchor the label is
    # right-aligned to — the tick, MARK_PAD_PT below the axis along the 45°
    # baseline. Both use rotation_mode='anchor', so they stay joined at any
    # figure size.
    feats = [(key, lab, x) for k, key, lab, x in items if k == 'feat']
    labels, weights, marks = [], [], []
    for key, lab, _x in feats:
        mark, mark_color = interaction_stars(coef, key)
        # Trailing spaces reserve room for the mark, which is drawn separately
        # at the anchor the label is right-aligned to. Sized for MARK_FONTSIZE —
        # a wider mark overruns a narrower gap and collides with the name.
        gap = MARK_GAP_MATH if '$' in lab else MARK_GAP
        # Gap on every line of a wrapped name: the lines are right-aligned to
        # one edge, so padding only the first would shift it off the others.
        labels.append('\n'.join(f'{ln}{gap}' for ln in lab.split('\n'))
                      if mark else lab)
        corrected = mark and mark_color == COLOR_CORRECTED
        weights.append('bold' if corrected else 'normal')
        marks.append((mark, mark_color, 'bold' if corrected else 'normal'))

    xs = [x for _, _, _, x in items]
    ax.set_xticks([x for _key, _lab, x in feats])
    ax.set_xticklabels(labels, fontsize=7.0, rotation=45, ha='right',
                       rotation_mode='anchor')
    for tick, weight in zip(ax.get_xticklabels(), weights):
        tick.set_fontweight(weight)
        tick.set_color(S.INK)

    for (_key, _lab, x), (mark, mark_color, weight) in zip(feats, marks):
        if not mark:
            continue
        ax.annotate(mark, xy=(x, 0), xycoords=('data', 'axes fraction'),
                    xytext=(0, -MARK_PAD_PT), textcoords='offset points',
                    ha='right', va='top', rotation=45, rotation_mode='anchor',
                    fontsize=MARK_FONTSIZE, fontweight='bold',
                    color=mark_color, annotation_clip=False, zorder=6)
    ax.set_xlim(min(xs) - COL_PITCH * 0.75, max(xs) + COL_PITCH * 0.75)
    ax.tick_params(axis='x', length=0, pad=1.5)
    ax.tick_params(axis='y', labelsize=7.0)

    ax.set_ylim(*(ylim if ylim else (-lim, lim)))
    for spine in ('top', 'right', 'bottom'):
        ax.spines[spine].set_visible(False)
    ax.spines['left'].set_color(S.RULE)
    # behind everything, and on the effect axis only — a vertical grid would cut
    # across the intervals it is meant to help measure
    ax.set_axisbelow(True)
    ax.grid(True, axis='y', color=GRID_COL, lw=GRID_LW, zorder=0)
    ax.grid(False, axis='x')
    return any_boot


def panel_layout(df):
    """(panels, spans) — the column blocks and their relative widths."""
    panels = [it for it in (_layout(df, k) for k in PANEL_ORDER) if it]
    spans  = [max(x for _, _, _, x in it) + 1.2 for it in panels]
    return panels, spans


def draw_into(fig, subspec, df, coef, title=None):
    """
    Draw the panel into one cell of an existing figure.

    Split out from draw() so make_rest_mvt_combined_figure.py can place this
    panel below the spectra panel without either script owning the figure.

    Orientation: features run along the X axis and the effect along Y, so the
    panel is wide and short and can sit under the spectra panel at the same page
    width. The zero reference is therefore a horizontal line and the intervals
    are vertical.

    Returns (axes, panels, any_boot), or None when no feature fits any panel.
    """
    panels, spans = panel_layout(df)
    if not panels:
        return None

    # wspace is a fraction of the MEAN axes width, so with panels this unequal
    # (9 features vs 2) any fixed value either crushes the small ones or opens a
    # chasm. Solve it for a constant physical gap instead, measuring the cell the
    # caller actually gave us rather than assuming a whole figure.
    cell_in = subspec.get_position(fig).width * fig.get_size_inches()[0]
    if len(panels) > 1:
        wspace = max(0.12, GAP_INCHES / (cell_in / len(panels)))
    else:
        wspace = 0.0

    gs = subspec.subgridspec(1, len(panels), width_ratios=spans, wspace=wspace)
    axes = [fig.add_subplot(gs[0, i]) for i in range(len(panels))]

    any_boot = False
    for ax, it in zip(axes, panels):
        kind = _unit_kind(df, [k for t, k, _, _ in it if t == 'feat'][0])
        any_boot |= _draw_panel(ax, df, coef, it, ylim=PANEL_YLIM.get(kind))

    # one y label per panel: the panels are deliberately on different scales, so a
    # frequency shift and a log ratio never share an axis. Each spells the
    # quantity out in full — with the log-ratio features now in a single panel
    # there are only two labels, and a bare '(Hz)' left the reader to infer what
    # the axis measured from the panel beside it.
    for ax, it in zip(axes, panels):
        kind = _unit_kind(df, [k for t, k, _, _ in it if t == 'feat'][0])
        unit = {'hz': '(Hz)', 'flat': '(log10)'}.get(kind, '(ln)')
        ax.set_ylabel(f'Movement − Rest   {unit}',
                      fontsize=7.5, labelpad=4)

    # Direction cue on the RIGHT of the last panel — on the left it competed with
    # the y label and the tick numbers for the same gutter. The arrows are drawn
    # rather than set as glyphs in the string, so they run parallel to the text
    # and to the effect axis they describe instead of lying on their side with it.
    axc = axes[-1]
    for y0, y1, lo, hi, lab in ((0.52, 0.96, 0.56, 0.96, 'higher in mvt'),
                                (0.48, 0.04, 0.44, 0.04, 'lower in mvt')):
        axc.text(1.10, (lo + hi) / 2, lab, transform=axc.transAxes,
                 rotation=90, ha='center', va='center', fontsize=6.5,
                 color=S.MUTED, clip_on=False)
        axc.annotate('', xy=(1.19, y1), xytext=(1.19, y0),
                     xycoords='axes fraction', textcoords='axes fraction',
                     arrowprops=dict(
                         arrowstyle='->,head_width=0.12,head_length=0.30',
                         color=S.MUTED, lw=0.5))

    handles = [
        Line2D([], [], marker='o', ms=MS_LEGEND, mfc=C_OFF, mec=C_OFF, mew=1.5,
               ls='none', label='Stim OFF'),
        Line2D([], [], marker='o', ms=MS_LEGEND, mfc=S.SURFACE, mec=C_ON, mew=1.5,
               ls='none', label='Stim ON'),
    ]
    # inside the first panel, top right, where the band-power intervals leave a
    # gap. Above the axes it drifted into the space between this panel and
    # whatever sits over it, and read as that neighbour's legend instead.
    axes[0].legend(handles=handles, loc='upper right', ncol=1, frameon=False,
                   fontsize=7.0, handletextpad=0.5,
                   borderaxespad=0.3)

    if title:
        pos = axes[0].get_position()
        fig.text(pos.x0 - 0.42 / fig.get_size_inches()[0],
                 pos.y1 + 0.15 / fig.get_size_inches()[1], title,
                 fontsize=9.0, fontweight='bold', color=S.INK,
                 ha='left', va='bottom')

    return axes, panels, any_boot
