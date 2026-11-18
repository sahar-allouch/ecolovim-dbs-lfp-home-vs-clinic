"""
Individual trajectories across M3 → home → M6, as small multiples.

One thin line per patient, points at each session, a heavy line for the group
median, one column per recording condition, one row per feature with its own
y-scale. The point is to let a reader check, by eye, the four things a table of
ICCs and betas cannot show:

    * whether patients keep their relative ordering across sessions
      (that is what a high ICC is claiming);
    * whether the at-home value sits systematically off the two clinic values,
      rather than being noisier;
    * whether an effect rides on one or two patients;
    * whether the spread widens under stimulation ON.

The main figure shows six features (FEATURES below): alpha, beta and tremor
f0 power, the aperiodic exponent and offset, and flattened beta power. The
supplement shows all 13 modelled features (ALL_FEATURES).

Values are the same unit the ICC and LME tables report, so a line here is the
quantity those models were fitted on.

Output: Results/Figures/fig_trajectories_{side}.png (main figure; + .tiff for
        bilateral) and
        Results/Figures/fig_trajectories_all_{side}_{1,2}.png (the 13-feature
        supplement, split into two pages — see SUPP_PAGE_SIZES)
Run:    uv run python make_trajectory_figure.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

matplotlib.use('Agg')

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import table_figure_style as S  # noqa: E402
from stats_analysis.data_utils import (  # noqa: E402
    TIMEPOINTS,
    build_aperiodic_df,
    build_bandpower_df,
    build_broadband_power_df,
    build_flattened_bandpower_df,
    build_peak_params_df,
    build_psd_array,
    build_tremor_bandpower_df,
    load_all_sessions,
    load_all_sessions_tremor,
)

OUT = ROOT / 'Results/Figures'
SIDES = ('bilateral', 'L', 'R')
SIDE = 'bilateral'   # the unit being drawn; main() sets it per side

# One column per recording condition, so rest and movement sit in the same
# figure without sharing axes: overlaying them would put ~18 lines in a panel
# and bury the per-patient structure this figure exists to show. Columns are
# ordered so the two stimulation-OFF panels sit together, then the two ON.
PANELS = [('rest_stimOFF', 'rest', 'Rest', 'Stimulation OFF'),
          ('mvt_stimOFF', 'mvt', 'Movement', 'Stimulation OFF'),
          ('rest_stimON', 'rest', 'Rest', 'Stimulation ON'),
          ('mvt_stimON', 'mvt', 'Movement', 'Stimulation ON')]

# (key, source, band/column, display label, y-scale)
#
# Flat (aperiodic-adjusted) power sits on a clipped-log scale that can reach the
# floor, and peak power is already log10 above the fit, so both are drawn on a
# linear axis; raw band power and tremor power are drawn on a log axis, the
# scale their models were fitted on.
FEATURES = [
    ('abs_alpha', 'bandpower', 'alpha',
     'Alpha power', 'log'),
    ('abs_beta', 'bandpower', 'beta',
     'Beta power', 'log'),
    ('abs_tremor_f0', 'tremor', 'tremor_f0',
     'Tremor power f0', 'log'),
    ('exponent', 'aperiodic', 'exponent',
     'Aperiodic exponent', 'linear'),
    ('offset', 'aperiodic', 'offset',
     'Aperiodic offset', 'linear'),
    ('flat_abs_beta', 'flat', 'beta',
     'Flat beta power', 'linear'),
]

# Supplement: the 13 features the LME and ICC tables model, in table order, so
# every panel has a statistic a reader can look up and vice versa.
#
# It used to carry 18, adding abs_gamma1, flat_abs_gamma1 and flat_abs_{delta,
# theta,alpha}.  None of those are modelled: EXCLUDED_BANDS drops gamma1 from
# ANALYSIS_BANDS (config.py) and flat power is beta-only (INCLUDE_FLAT /
# FLAT_ANALYSIS_BANDS in lme_analysis.py).  flat_abs_gamma1 was the worst of
# them — APERIODIC_FREQ_RANGE = [1, 40] truncates the FOOOF grid, so it covered
# only 35–39 Hz of its nominal 35–55 and the panel was a partial-band artefact.
# _assert_matches_analysis_set() below re-derives the set from those constants
# at import, so this list cannot drift from the models again.
ALL_FEATURES = [
    ('abs_delta', 'bandpower', 'delta', 'Delta power', 'log'),
    ('abs_theta', 'bandpower', 'theta', 'Theta power', 'log'),
    ('abs_alpha', 'bandpower', 'alpha', 'Alpha power', 'log'),
    ('abs_beta', 'bandpower', 'beta', 'Beta power', 'log'),
    ('abs_tremor_f0', 'tremor', 'tremor_f0', 'f0 power', 'log'),
    ('abs_tremor_2f', 'tremor', 'tremor_2f', '2f0 power', 'log'),
    ('exponent', 'aperiodic', 'exponent', 'Aperiodic exponent', 'linear'),
    ('offset', 'aperiodic', 'offset', 'Aperiodic offset', 'linear'),
    ('broadband_power', 'broadband', 'broadband_power',
     'Broadband power', 'log'),
    ('flat_abs_beta', 'flat', 'beta', 'Flat beta power', 'linear'),
    ('cf', 'peak', 'cf', 'Beta peak centre frequency', 'linear'),
    ('pw', 'peak', 'pw', 'Beta peak power', 'linear'),
    ('bw', 'peak', 'bw', 'Beta peak bandwidth', 'linear'),
]


def _assert_matches_analysis_set():
    """
    Fail loudly if ALL_FEATURES drifts from the feature set the models fit.

    Re-derives the set from the same constants lme_analysis.py and
    icc_analysis.py read, rather than trusting this file's own list — the
    drift it guards against is exactly how five unmodelled panels ended up in
    the supplement.  Peak metrics are keyed 'cf'/'pw'/'bw' here but
    'peak_beta_*' in the tables, so they are compared on the metric name.
    """
    from config import ANALYSIS_BANDS
    from stats_analysis.data_utils import PEAK_METRICS
    from stats_analysis.lme_analysis import FLAT_ANALYSIS_BANDS, INCLUDE_FLAT

    expected = {f'abs_{b}' for b in ANALYSIS_BANDS}
    expected |= {'abs_tremor_f0', 'abs_tremor_2f',
                 'exponent', 'offset', 'broadband_power'}
    if INCLUDE_FLAT:
        expected |= {f'flat_abs_{b}' for b in FLAT_ANALYSIS_BANDS}
    expected |= set(PEAK_METRICS)

    have = {f[0] for f in ALL_FEATURES}
    if have != expected:
        raise AssertionError(
            'ALL_FEATURES is out of step with the modelled feature set.\n'
            f'  plotted but not modelled: {sorted(have - expected) or "none"}\n'
            f'  modelled but not plotted: {sorted(expected - have) or "none"}')


_assert_matches_analysis_set()

TP_LABEL = {'M3': 'M3', 'home': 'home', 'M6': 'M6'}

# One condition column, sized so the whole plate is delivered at the shared page
# width (table_figure_style.TARGET_W_MM).  It used to be 2.85 in, which put the
# figure at 286 mm; placed in a 170 mm column that scaled everything by 0.59 and
# printed the 8.2 pt labels at 4.9 pt, below the legibility floor and out of step
# with the other plates.  bbox_inches='tight' crops to the artists, so the
# declared width is divided by the measured crop fraction for this layout.
# Width per condition column.  Deliberately NOT tied to
# table_figure_style.TARGET_W_MM: authoring this plate at 170 mm puts each panel
# at 42 mm, and at that size the fixed 7.4-8.2 pt type collides — the y-axis
# labels of adjacent rows overlap and panel A's legend lands on the traces.  A
# 4-column x 6-row grid of line panels needs its own width, so this figure is
# placed at reduced scale in the manuscript and its type is correspondingly
# smaller on the page than the single-axis plates.
# Delivered width, in mm.  The plate is authored at its original geometry and
# then ZOOMED as a whole: figsize, every point size, every line width and every
# inch-valued pad are multiplied by SCALE, so the design is identical to the
# 286 mm version and only its size on the page changes.  Height follows from
# the same factor and is left free.
#
# Scaling the canvas WITHOUT scaling the type is a different operation and a
# worse one: it shrinks the panels while the 7.4-8.2 pt labels stay put, which
# is what made adjacent rows' y-labels collide and dropped panel A's legend on
# to the traces.
TARGET_W_MM = 185.0
TARGET_W_IN = TARGET_W_MM / 25.4

# bbox_inches='tight' crops to the drawn artists, so the delivered width is a
# fixed fraction of the declared figsize.  Measured for this layout; the zoom
# leaves it unchanged because every element scales together.
TIGHT_CROP = 0.98519

PANEL_W_IN = 2.85          # authored width per condition column
ROW_H_IN = 1.80            # authored height per feature row

SCALE = TARGET_W_IN / TIGHT_CROP / (PANEL_W_IN * 4)


def _z(v):
    """Any authored point size, line width or inch pad, at the delivered size."""
    return v * SCALE


# Printed point sizes at TARGET_W_MM.  The plate is authored at its delivered
# size, so these ARE the sizes on the page — no journal rescaling sits between
# them and the reader.  They are set here rather than zoomed off
# table_figure_style's constants because type does not scale with a figure: a
# uniform 0.63 zoom put the tick labels at 4.7 pt and the n annotation at 3.1 pt,
# under the ~5 pt floor most journals set for final-size text.  Geometry still
# scales through _z(); only type is pinned.
FS_SECTION = 9.0     # A / B block headings
FS_TITLE = 7.5       # condition title over each panel
FS_LABEL = 7.5       # axis labels
FS_TICK = 7.0        # tick labels
FS_LEGEND = 7.0      # legend entries
FS_ANNOT = 6.5       # the n annotation in panel A

# 600 dpi for print.  Set here rather than raising table_figure_style.PNG_DPI,
# which 10 other plates share.
PNG_DPI = 900
SECTION_GAP = 0.30  # empty grid row that holds the trajectories heading
PSD_LEGEND_GAP = 0.28  # extra gap row height when the PSD legend sits above it

# Section headings, so each block names what its panels show. The figure is
# cited by block, not by cell — the four columns are the same four conditions
# in both blocks — so the letters go on the two headings and the individual
# panels carry none.
SECTION_PSD = 'A   Group spectral profiles'
SECTION_TRAJ = 'B   Participant-level feature trajectories'

# Ink, surface and grid are set here rather than taken from table_figure_style,
# whose pure black is shared with the tables: on a figure this dense, charcoal
# on white keeps the panels from reading as a grid of black boxes.
INK = '#222222'            # text, axes, labels
MUTED = '#6B7280'          # secondary annotation (the per-panel n)
SURFACE = '#ffffff'
GRID_LINE = '#E6E8EB'

# Recessive grid, behind the data.
GRID_KW = dict(color=GRID_LINE, lw=_z(0.5), zorder=0)

# Hue encodes the recording condition — one condition per column, in column
# order — and is spent ONLY on the group summary: median line, median markers
# and IQR ribbon. Individual patients are drawn in one neutral grey for every
# panel, so hue never has to separate an individual from a group, and the eye
# lands on the summary first. Stim OFF in purples (plum / lavender), stim ON in
# yellows (amber / olive-gold), so the stimulation state reads as a hue family
# and rest vs movement as a shade within it. Each condition owns a whole column
# and is titled above it, so colour is redundant with position and only has to
# stay legible in print and under the common CVD types.
PATIENT_COLOR = '#B8BEC7'
COND_COLOR = {'rest_stimOFF': '#8E4B7A',   # plum
              'mvt_stimOFF': '#6F63B6',    # lavender-purple
              'rest_stimON': '#D99000',    # amber
              'mvt_stimON': '#9A8F2A'}     # olive-gold
COND_LABEL = {'rest_stimOFF': 'Rest, stim OFF', 'mvt_stimOFF': 'Movement, stim OFF',
              'rest_stimON': 'Rest, stim ON', 'mvt_stimON': 'Movement, stim ON'}
ALPHA_PATIENT = 0.42
IQR_ALPHA = 0.15           # across-patient IQR ribbon in the trajectory panels
LW_PATIENT = _z(0.8)
LW_MEDIAN = _z(2.8)
MS_PATIENT = _z(3.0)
MS_MEDIAN = _z(7.0)

# ── Spectra row ──────────────────────────────────────────────────────────────
# Session hues: muted navy / teal / vermilion. Well separated for normal vision
# and under deutan and protan simulation, since these three curves are the only
# place in the figure where colours must be told apart INSIDE one panel. All
# three are drawn solid; set a dash pattern here if a greyscale reproduction is
# ever needed, where navy and vermilion converge.
C_SESSION = {'M3': '#355C9A', 'home': '#2F8F83', 'M6': '#C95C54'}
DASH_SESSION = {'M3': (None, None), 'home': (None, None), 'M6': (None, None)}
LW_PSD = _z(1.9)
BAND_ALPHA = 0.10          # +/- 1 SD ribbon
PSD_FMIN, PSD_FMAX = 1.0, 45.0   # display range, matches Fig 1C (make_figure1C.py)

# The spectra span ~3 decades and are plotted on a log axis, so mean +/- SD is
# taken on log10 power: the mean line is then the geometric mean and the ribbon
# is a multiplicative +/- 1 SD, which cannot reach the negative values a linear
# SD would put below a log axis. Set to 'linear' for an arithmetic mean +/- SD.
PSD_AGG = 'log'


def load_long(features) -> pd.DataFrame:
    """One tidy frame: patient, timepoint, condition, feature, value.

    Each source is read at most once, however many features come from it.
    """
    srcs = {src for _k, src, *_ in features}
    sessions = load_all_sessions() if srcs - {'tremor'} else None
    frames = []

    def _pick(df, col, key, value_name):
        return (df.assign(feature=key)[
            ['patient', 'timepoint', 'condition', 'feature', col]]
            .rename(columns={col: value_name}))

    if {'bandpower'} & srcs:
        bp = build_bandpower_df(sessions, side=SIDE, power_type='abs')
        for key, src, col, *_ in features:
            if src == 'bandpower':
                frames.append(_pick(bp[bp['band'] == col], 'power', key,
                                    'value'))

    if {'flat'} & srcs:
        fl = build_flattened_bandpower_df(sessions, side=SIDE, power_type='abs')
        for key, src, col, *_ in features:
            if src == 'flat':
                frames.append(_pick(fl[fl['band'] == col], 'power', key,
                                    'value'))

    if {'aperiodic'} & srcs:
        ap = build_aperiodic_df(sessions, side=SIDE)
        for key, src, col, *_ in features:
            if src == 'aperiodic':
                frames.append(_pick(ap, col, key, 'value'))

    # Broadband power is its own builder, not a column of the aperiodic frame.
    if {'broadband'} & srcs:
        bb = build_broadband_power_df(sessions, side=SIDE)
        for key, src, col, *_ in features:
            if src == 'broadband':
                frames.append(_pick(bb, col, key, 'value'))

    if {'peak'} & srcs:
        pk = build_peak_params_df(sessions, side=SIDE, band='beta')
        for key, src, col, *_ in features:
            if src == 'peak':
                frames.append(_pick(pk.dropna(subset=[col]), col, key, 'value'))

    if {'tremor'} & srcs:
        tr = build_tremor_bandpower_df(load_all_sessions_tremor(), side=SIDE)
        for key, src, col, *_ in features:
            if src == 'tremor':
                frames.append(_pick(tr[tr['band'] == col], 'power', key,
                                    'value'))

    out = pd.concat(frames, ignore_index=True)
    out['timepoint'] = out['timepoint'].astype(str)
    return out.dropna(subset=['value'])


def _box_spines(ax):
    """A closed frame on all four sides."""
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(_z(0.8))
        sp.set_color(INK)


def _agg_spectra(spec: np.ndarray):
    """(centre, lo, hi) of the per-patient spectra, in linear power.

    In 'log' mode these are the geometric mean and a multiplicative +/- 1 SD;
    in 'linear' mode the arithmetic mean and +/- 1 SD, clipped at zero so the
    ribbon stays drawable on a log axis.
    """
    if spec.size == 0 or spec.shape[0] < 1:
        return None
    if PSD_AGG == 'log':
        lg = np.log10(spec)
        m = lg.mean(axis=0)
        sd = lg.std(axis=0, ddof=1) if spec.shape[0] > 1 else np.zeros_like(m)
        return 10 ** m, 10 ** (m - sd), 10 ** (m + sd)
    m = spec.mean(axis=0)
    sd = spec.std(axis=0, ddof=1) if spec.shape[0] > 1 else np.zeros_like(m)
    return m, np.clip(m - sd, np.finfo(float).tiny, None), m + sd


def _draw_psd_row(axes_row, freqs, spectra):
    """Top row: group spectrum per session, one panel per condition."""
    for c, (cond, _mov, _mov_label, _stim_label) in enumerate(PANELS):
        ax = axes_row[c]
        n_by_tp = []
        for tp in TIMEPOINTS:
            spec = spectra.get((tp, cond), np.empty((0, 0)))
            n_by_tp.append(spec.shape[0] if spec.size else 0)
            agg = _agg_spectra(spec)
            if agg is None:
                continue
            centre, lo, hi = agg
            ax.fill_between(freqs, lo, hi, color=C_SESSION[tp],
                            alpha=BAND_ALPHA, lw=0, zorder=2)
            line, = ax.plot(freqs, centre, color=C_SESSION[tp], lw=LW_PSD,
                            zorder=3, label=TP_LABEL[tp])
            if DASH_SESSION[tp][0] is not None:
                line.set_dashes(DASH_SESSION[tp])

        ax.set_yscale('log')
        ax.set_xlim(PSD_FMIN, PSD_FMAX)
        ax.tick_params(axis='both', labelsize=FS_TICK, length=_z(3))
        _box_spines(ax)
        ax.set_axisbelow(True)
        ax.grid(True, axis='both', **GRID_KW)
        ax.set_xlabel('Frequency (Hz)', fontsize=FS_LABEL,
                      labelpad=_z(1.5))
        ax.text(0.98, 0.96, 'n = ' + '/'.join(str(n) for n in n_by_tp),
                transform=ax.transAxes, ha='right', va='top',
                fontsize=FS_ANNOT, color=MUTED)
        if c == 0:
            ax.set_ylabel('Power (µVp²)', fontsize=FS_LABEL)



def draw(long: pd.DataFrame, features, stem: str, psd=None, heading=None):
    S.apply_rcparams()
    # White surface and charcoal ink everywhere, including the tick labels and
    # the spines matplotlib draws before _box_spines gets to them.
    plt.rcParams.update({
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
        'savefig.facecolor': SURFACE, 'text.color': INK,
        'axes.labelcolor': INK, 'axes.edgecolor': INK,
        'xtick.color': INK, 'ytick.color': INK,
    })
    x = np.arange(len(TIMEPOINTS), dtype=float)

    # The spectra row is added on top when spectra are supplied. It cannot share
    # an x-axis with the trajectory rows below — its x is frequency, theirs is
    # the session index — so the grid is built by hand rather than with
    # plt.subplots(sharex=True), and x-sharing is applied within the trajectory
    # block only.
    n_extra = 1 if psd is not None else 0
    n_row, n_col = len(features) + n_extra, len(PANELS)

    # An empty grid row between the PSD block and/or the page heading and the
    # trajectory rows carries the section heading; without it the heading
    # would sit on top of the first feature's panels. A page with no PSD row
    # but its own heading (a continuation page of a split supplement) still
    # needs that gap row.
    show_gap = n_extra or heading is not None
    gap = SECTION_GAP + (PSD_LEGEND_GAP if n_extra else 0)
    gs_rows = (([1.0] if n_extra else []) + ([gap] if show_gap else [])
               + [1.0] * len(features))
    r_grid = ([0] if n_extra else []) + list(
        range(n_extra + (1 if show_gap else 0), len(gs_rows)))

    fig = plt.figure(figsize=(_z(PANEL_W_IN) * n_col,
                              _z(ROW_H_IN) * n_row
                              + _z(ROW_H_IN) * (gap if show_gap else 0)))
    gs = fig.add_gridspec(len(gs_rows), n_col, height_ratios=gs_rows)
    axes = np.empty((n_row, n_col), dtype=object)
    for c in range(n_col):
        if n_extra:
            axes[0, c] = fig.add_subplot(gs[r_grid[0], c],
                                         sharey=axes[0, 0] if c else None)
        for r in range(n_extra, n_row):
            axes[r, c] = fig.add_subplot(
                gs[r_grid[r], c],
                sharex=axes[n_extra, c] if r > n_extra else None,
                sharey=axes[r, 0] if c else None)

    if psd is not None:
        _draw_psd_row(axes[0], *psd)

    for r0, (key, _src, _col, flabel, scale) in enumerate(features):
        r = r0 + n_extra
        for c, (cond, mov, mov_label, stim_label) in enumerate(PANELS):
            ax = axes[r, c]
            d = long[(long.feature == key) & (long.condition == cond)]

            wide = d.pivot_table(index='patient', columns='timepoint',
                                 values='value', aggfunc='median')
            wide = wide.reindex(columns=TIMEPOINTS)

            col = COND_COLOR[cond]

            for _pat, row in wide.iterrows():
                ok = row.notna().to_numpy()
                if ok.sum() == 0:
                    continue
                ax.plot(x[ok], row.to_numpy()[ok], '-o', color=PATIENT_COLOR,
                        lw=LW_PATIENT, ms=MS_PATIENT, mfc=PATIENT_COLOR,
                        mec='white', mew=0.4, zorder=2, alpha=ALPHA_PATIENT)

            med = wide.median(axis=0, skipna=True)
            ok = med.notna().to_numpy()

            # IQR rather than an SD ribbon: it is the spread that matches the
            # median already drawn, and these n<=9 distributions are skewed
            # (band power especially), so a symmetric band would misstate them.
            q25 = wide.quantile(0.25, axis=0).to_numpy()
            q75 = wide.quantile(0.75, axis=0).to_numpy()
            ax.fill_between(x[ok], q25[ok], q75[ok], color=col,
                            alpha=IQR_ALPHA, lw=0, zorder=1)

            ax.plot(x[ok], med.to_numpy()[ok], '-o', color=col,
                    lw=LW_MEDIAN, ms=MS_MEDIAN, mfc=col, mec='white',
                    mew=1.0, zorder=3)

            if scale == 'log':
                ax.set_yscale('log')
            ax.set_xticks(x)
            ax.set_xticklabels([TP_LABEL[t] for t in TIMEPOINTS],
                               fontsize=FS_TICK)
            ax.tick_params(axis='y', labelsize=FS_TICK, length=_z(3))
            ax.tick_params(axis='x', length=3)
            _box_spines(ax)
            ax.set_axisbelow(True)
            ax.grid(True, axis='y', **GRID_KW)
            ax.set_xlim(-0.35, len(TIMEPOINTS) - 0.65)
            # Session ticks are repeated only on the bottom row; plt.subplots
            # did this via sharex, which the hand-built grid no longer sets.
            if r < n_row - 1:
                ax.tick_params(axis='x', labelbottom=False)

            if c == 0:
                # Wrapped: at FS_LABEL / ROW_H_IN a rotated one-line label
                # longer than ~20 chars ("Tremor 2nd harmonic (2f0 ± 1 Hz)",
                # "Beta peak power (log10 above fit)") runs taller than its
                # own row and bleeds into the neighbouring row's label.
                ax.set_ylabel(_wrap(flabel, 20), fontsize=FS_LABEL,
                              multialignment='center')

    # Every column is one condition for the whole figure, so the title is
    # drawn once, over the top row only, not repeated down every row.
    for c, (cond, _mov, _mov_label, _stim_label) in enumerate(PANELS):
        axes[0, c].set_title(COND_LABEL[cond], fontsize=FS_TITLE,
                             fontweight='bold', color=INK, loc='center',
                             pad=6)

    # One entry per condition (the median line, since that is the heavy mark),
    # plus one for the individual-patient weight — grey in every panel, so
    # repeating it four times would say nothing.
    handles = [Line2D([], [], color=COND_COLOR[cond], lw=LW_MEDIAN, marker='o',
                      ms=MS_MEDIAN, mfc=COND_COLOR[cond], mec='white',
                      label=f'{COND_LABEL[cond]} — median')
               for cond, _mov, _lab, _stim in PANELS]
    handles.append(Line2D([], [], color=PATIENT_COLOR, lw=LW_PATIENT,
                          marker='o', ms=MS_PATIENT, mfc=PATIENT_COLOR,
                          mec='white', alpha=ALPHA_PATIENT,
                          label='Individual patient'))
    handles.append(Patch(facecolor=MUTED, alpha=IQR_ALPHA + 0.10, lw=0,
                         label='Interquartile range across patients'))
    fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False,
               fontsize=FS_LEGEND,
               bbox_to_anchor=(0.5, _z(0.22) / fig.get_figheight()),
               columnspacing=1.8)

    # Margins are pinned in inches, not figure fractions: the supplement is
    # ~5x taller than the main figure, and a fractional reserve would shrink
    # with it until the section heading collided with the figure edge.
    fig_h_in = fig.get_figheight()

    # The legend now wraps to two rows, so the bottom reserve holds both. The
    # top reserve holds the section heading over the first row when there is
    # one (the PSD block and/or a page heading) and shrinks to a bare margin
    # otherwise, e.g. the headingless supplement pages.
    # pad/h_pad/w_pad pulled in from the 1.08 default: that default reserves
    # extra whitespace sized off rcParams['font.size'] (~10 pt), well above
    # any label actually drawn here (the biggest is FS_SECTION), so panels
    # sat further apart than the labels needed once every element got bigger.
    top_pad_in = 0.72 if (n_extra or heading is not None) else 0.12
    fig.tight_layout(rect=(0, _z(0.95) / fig_h_in, 1, 1 - _z(top_pad_in) / fig_h_in),
                     pad=0.4, h_pad=0.4, w_pad=0.4)
    # One x per column of ylabels, so the row labels form a single left edge
    # instead of drifting with each row's tick-label width.
    fig.align_ylabels(list(axes[:, 0]))
    if psd is not None:
        h, l = axes[0, 0].get_legend_handles_labels()
        y0 = axes[0, 0].get_position().y0
        fig.legend(h, l, loc='upper center', ncol=len(h), frameon=False,
                   fontsize=FS_LEGEND, handlelength=2.4, columnspacing=1.8,
                   bbox_to_anchor=(0.5, y0 - _z(0.56) / fig_h_in))
    _section_titles(fig, axes, n_extra, heading)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f'{stem}.png', dpi=PNG_DPI, bbox_inches='tight',
                facecolor=SURFACE)
    if stem == 'fig_trajectories_bilateral':   # main-text figure only
        fig.savefig(OUT / f'{stem}.tiff', dpi=PNG_DPI, bbox_inches='tight',
                    facecolor=SURFACE, pil_kwargs={'compression': 'tiff_lzw'})
    plt.close(fig)
    print(f'saved {OUT / f"{stem}.png"}')


def _section_titles(fig, axes, n_extra, heading):
    """Heading over each block, so a panel says what it shows on its own.

    Positioned from the drawn axes rather than in figure fractions, for the
    same reason as the stimulation bands: the supplement is ~5x taller than the
    main figure and a fractional offset would grow with it.
    """
    if not n_extra and heading is None:
        return
    h_in = fig.get_figheight()

    def _put(row, gap_in, title):
        # Anchored to the left spine of the block it heads, not to a single
        # row's x0: the two blocks can end up with different left margins once
        # tight_layout has sized each row's tick labels.
        pos = axes[row, 0].get_position()
        fig.text(pos.x0, pos.y1 + _z(gap_in + 0.08) / h_in, title, ha='left',
                 va='bottom', fontsize=FS_SECTION, fontweight='bold',
                 color=INK)

    # Section A clears the panel letter and condition title over the top row.
    # The trajectory heading sits in the empty grid row above the first
    # feature row — same row whether or not a PSD block precedes it, so a
    # split-off continuation page (no PSD row, n_extra=0) still gets it right
    # above its own top row.
    if n_extra:
        _put(0, 0.42, SECTION_PSD)
    if heading is not None:
        _put(n_extra, 0.36 if n_extra else 0.42, heading)


def _wrap(text: str, width: int) -> str:
    import textwrap
    return '\n'.join(textwrap.wrap(text, width))


def report(long: pd.DataFrame, features):
    """Printed so the drawn lines can be checked against the numbers."""
    print('\n' + '=' * 78)
    print('MEDIAN BY SESSION (the heavy line in each panel)')
    print('=' * 78)
    for key, _src, _col, flabel, _sc in features:
        print(f'\n  {flabel}')
        for cond, _mov, _lab, _stim in PANELS:
                d = long[(long.feature == key) & (long.condition == cond)]
                if d.empty:
                    continue
                med = d.groupby('timepoint')['value'].median()
                n = d.groupby('timepoint')['patient'].nunique()
                cells = '  '.join(
                    f'{t} {med.get(t, float("nan")):>9.3g} (n={n.get(t, 0)})'
                    for t in TIMEPOINTS)
                print(f'    {cond:<14} {cells}')


# The 13-row supplement plate is too tall for a single page of a supplementary
# materials document, so it is split into two pages instead — the same
# absolute band power + tremor-frequency (6 rows) vs aperiodic + beta-peak
# (7 rows). Neither page carries the spectra row or a section heading — draw() is
# called with psd=None, heading=None — so each page is just its feature grid.
# Row height is fixed (ROW_H_IN * SCALE), so this chunking is the only knob.
SUPP_PAGE_SIZES = (6, 7)


def _paginate(items, sizes):
    out, i = [], 0
    for n in sizes:
        out.append(items[i:i + n])
        i += n
    assert i == len(items), (
        f'SUPP_PAGE_SIZES covers {i} features, ALL_FEATURES has {len(items)}')
    return out


def _run_side(side: str):
    # load_long() and the loaders it calls read the module-level SIDE.
    global SIDE
    SIDE = side
    # Every set carries its side in the filename, bilateral included.
    sfx = f'_{side}'
    print(f'\n── side: {side} ──')

    freqs, spectra, _pat = build_psd_array(load_all_sessions(), side=SIDE,
                                           fmin=PSD_FMIN, fmax=PSD_FMAX)
    psd = (freqs, spectra)

    main_long = load_long(FEATURES)
    draw(main_long, FEATURES, f'fig_trajectories{sfx}', psd=psd,
         heading=SECTION_TRAJ)
    report(main_long, FEATURES)

    all_long = load_long(ALL_FEATURES)
    pages = _paginate(ALL_FEATURES, SUPP_PAGE_SIZES)
    for i, page_features in enumerate(pages):
        draw(all_long, page_features, f'fig_trajectories_all{sfx}_{i + 1}',
             psd=None, heading=None)
    print(f'\n  supplement: {len(ALL_FEATURES)} features x {len(PANELS)} '
          f'conditions, split across {len(pages)} pages '
          f'({"/".join(str(len(p)) for p in pages)} feature rows)')


def main() -> int:
    for side in SIDES:
        _run_side(side)
    return 0


if __name__ == '__main__':
    sys.exit(main())
