"""Panel A of the rest-vs-movement figure: bilateral power spectra by stim state.

NOT A SCRIPT. This module draws Panel A into a cell of a figure it does not own,
through draw_into(); make_rest_mvt_combined_figure.py is the only entry point and
the only thing that writes a file. The standalone fig_rest_mvt_psd_* output and
its CLI were removed — the panel is only ever published beside Panel B, and two
ways of producing it meant two things to keep in step.

Companion to make_rest_mvt_forest_figure.py (Panel B). Panel B reports the
movement effect band by band; this panel shows the spectra those band powers were
integrated from, so a reader can see where in frequency the effect sits before
reading the summary.

Layout
------
    rows          : M3, home, M6 — one recording session each
    columns       : stimulation OFF (left), stimulation ON (right)
    within each   : rest (solid) and movement (dashed)
    shaded        : the two bands carrying the result in Panel B (alpha, beta)

Encoding — deliberately the same grammar as Panel B
---------------------------------------------------
HUE always means stimulation state: brick red = OFF, blue = ON, matching
plot_style's warm/cool stim convention and Panel B's markers. Stim is carried by
the subplot here, so hue is redundant with position — which is the point: nothing
in the combined figure ever uses those two hues for anything else.

Movement is carried by LINE STYLE (rest solid, movement dashed) rather than by a
third and fourth hue, so the figure as a whole holds exactly two colours.

Units — why one session per row, never pooled
---------------------------------------------
Clinic and at-home spectra come from different acquisition modalities (the
in-clinic spectra are flatter), so their absolute levels are not directly
comparable. Giving each session its own row is what makes the figure safe: the comparison the figure exists to show, rest vs movement, happens
WITHIN a row, and any per-session scalar cancels from that comparison exactly.
Averaging the sessions into one curve would not have been safe, and comparing
absolute LEVEL between the home row and the two clinic rows still is not — read
down a column for shape, not for height.

Aggregation — matched to the test in Panel B
--------------------------------------------
The line is the GEOMETRIC mean across patients, i.e. the arithmetic mean taken in
log space and back-transformed; the ribbon is one geometric SD either side
(exp(mean(log) +/- sd(log))).

That is not a cosmetic choice. Panel B's model is fitted on log(mvt) - log(rest)
and its fixed effect is a mean of LOG values, which back-transforms to a
geometric mean. A median of linear power equals the median of log power exactly
(medians survive monotone transforms), and that equals the mean of logs only when
the log-scale distribution is symmetric — which is the model's assumption, not a
guarantee. Using the geometric mean makes the curve the same kind of average the
test reports, so the panels cannot disagree by construction.

--center median restores a median line with an interquartile ribbon, which
describes the patient spread more robustly but is not the estimator Panel B uses.

A caveat no summary of ABSOLUTE power can fix: the model never estimates absolute
level, only the within-session difference. The spectra rows are descriptive
context for Panel B, not a picture of its estimand — which is what the bottom row
is for.

Bottom row — the difference spectrum
------------------------------------
log(mvt) - log(rest) per frequency bin, one curve per stim state, against a zero
reference. Descriptive context for Panel B — it shows WHERE in frequency the
movement effect sits, not a reported statistic.

Sessions can be pooled here, and only here. The per-session scalar the units note
above warns about is a constant multiplier on a patient x session curve, so it
cancels exactly from a within-cell log difference and does not cancel from a
level. That is the same reason the forest panel may average sessions while the
spectra rows may not.

The estimator is a TWO-STAGE MEAN, no model: pairs are matched within patient x
session x stim, then per bin and per stim state

    1. per patient, average log(mvt/rest) over that patient's available sessions
    2. average those per-patient curves over patients

The ribbon is a CLUSTER BOOTSTRAP percentile interval — resample whole patients
with replacement, recompute the two-stage mean, take the 2.5/97.5 percentiles;
one set of resamples is reused across bins so the ribbon varies smoothly with
frequency. This targets the same quantity as Panel B's timepoint-averaged
simple effects (`lme_rest_mvt_diff.simple_effects`, a reduced per-state mixed
model) and agrees with it to well under 1% at band level; the partial pooling
the mixed model adds barely moves a mean of difference scores at n~9, and is not
worth a model per frequency bin for a descriptive curve. Panel B carries the
inference (stars, FDR).

One thing the caption must still say rather than leave implied: the curve does
NOT integrate to Panel B's points. Panel B differences log(sum of power over a
band); averaging this curve over the same band gives the mean over bins of a log
difference. Jensen — related, not equal.

"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

import table_figure_style as S

try:
    from config import FREQ_BANDS
except ImportError:
    FREQ_BANDS = {'alpha': (8, 13), 'beta': (13, 35)}

C_OFF = '#D46A4C'   # warm orange-red — stim OFF   (same as Panel B)
C_ON  = '#4A78A8'   # muted blue      — stim ON

# Bands to shade: the two Panel B reports as FDR-significant. Alpha (8-13) and
# beta (13-35) are adjacent, so they need to be told apart at a glance.
#
# Hues are drawn from the violet / green side of the wheel, deliberately away from
# both stim colours (brick red #CC503E, blue #1D6996): the band tint names a
# FREQUENCY RANGE, and must not be mistaken for a stimulation state. They are
# heavily desaturated so the spectra and their ribbons stay dominant.
HILITE_BANDS = ['alpha', 'beta']
BAND_FILL    = {'alpha': '#EEEAE4',   # light warm gray
                'beta':  '#E7EDF1'}   # light blue-gray
BAND_EDGE    = {'alpha': '#DBD3C8',
                'beta':  '#CCD8E1'}
# The band tint names a frequency range; it is not a surface the data sits on,
# so it does not get to hide the grid the levels are read against.
BAND_ALPHA   = 0.55

DEFAULT_TPS = ['M3', 'home', 'M6']   # one column each — see the units note
TP_LABEL    = {'M3': 'M3', 'home': 'Home', 'M6': 'M6'}

PANEL_TITLE = 'Power spectra by stimulation state and session'

# µVp², not an arbitrary unit: clinic spectra come from extract_psd_welch and
# home spectra from the device, both in Percept-style µVp magnitude, and
# build_psd_array squares them, so clinic and home share one basis.
POWER_UNIT  = 'Power (µVp²)'

# Empty grid column between the sessions and the difference, as a fraction of a
# session column's width. The rendered gap is 2*wspace + SPACER_W*col_width,
# since the uniform wspace lands on both sides of it.
SPACER_W    = 0.08

# Difference column width, as a fraction of a session column's. Under 1.0 because
# it carries one curve rather than two plus their ribbons, so equal width would
# overstate it — but not so far under that its shape is harder to read than the
# spectra it is derived from.
DIFF_W      = 0.95

LW_LINE   = 1.5
RIBBON_A  = 0.16

_CLIP     = 1e-12   # floor before the log, shared by both rows' log space

# Horizontal grid on the spectra columns: three decades of 1/f fall-off are hard
# to read a level off without one, and it is the level that is being compared
# across a row. Pale enough never to compete with a curve or its ribbon.
GRID_COL  = '#E8EDF1'
GRID_LW   = 0.6

# Cluster-bootstrap resamples for the difference row. Lower than the 2000 the
# coefficient table uses, because the cost here is n_boot x n_bins model fits
# rather than n_boot: at 45 bins, 2000 draws is 90k fits. 500 puts 12 draws in
# each 2.5% tail, coarse for a p-value and adequate for a descriptive ribbon —
# and it is the same construction as the tables, which is the point of using it.
N_BOOT_PSD = 500
BOOT_SEED  = 20240517

# The bootstrap is ~15 min at 500 draws and depends only on the data, the bin
# selection, n_boot and the seed — never on anything cosmetic. Caching it keyed
# on exactly those inputs means restyling the figure is free, while any change to
# the data or the resampling produces a different key and recomputes. Delete the
# directory, or pass --refresh-boot, to force a rebuild.
BOOT_CACHE = Path('Results') / 'stats' / 'lme_rest_mvt' / 'psd_diff_boot'

# The zero line in the difference row is a reference, not a data mark — same
# desaturated blue-grey and weight as the forest panel's, so the two figures
# agree on what a null looks like.
ZERO_COL  = '#CBD5E1'
LW_ZERO   = 0.7
Z_95      = 1.959964


def session_spectra(spectra, patients, condition, timepoint):
    """(n_patient, n_freq) for one session x condition, plus the patient ids."""
    arr = spectra.get((timepoint, condition))
    ids = patients.get((timepoint, condition), [])
    if arr is None or not arr.size or not ids:
        return np.empty((0, 0)), []
    return arr, ids


# ── Difference spectrum: Panel B's estimand, per frequency bin ────────────────

def build_diff_rows(spectra, patients, timepoints, sel):
    """
    Matched log(mvt) - log(rest) per patient x session x stim.

    Returns (meta, D): meta is a DataFrame of patient/timepoint/stim, D is
    (n_row, n_freq_selected) of log differences in the same row order.

    A cell contributes only when the SAME patient has both rest and mvt in that
    session x stim — the pairing is the whole point, and an unmatched half
    carries no information about the movement effect. This mirrors the dropna in
    lme_rest_mvt_diff.build_diff_df, so the two analyses discard the same cells.
    """
    rows, D = [], []
    for tp in timepoints:
        for stim in ('stimOFF', 'stimON'):
            a_rest, id_rest = session_spectra(spectra, patients, f'rest_{stim}', tp)
            a_mvt,  id_mvt  = session_spectra(spectra, patients, f'mvt_{stim}',  tp)
            if not a_rest.size or not a_mvt.size:
                continue
            pos_rest = {p: i for i, p in enumerate(id_rest)}
            pos_mvt  = {p: i for i, p in enumerate(id_mvt)}
            for pid in id_rest:
                if pid not in pos_mvt:
                    continue
                lr = np.log(np.clip(a_rest[pos_rest[pid], sel], _CLIP, None))
                lm = np.log(np.clip(a_mvt[pos_mvt[pid], sel], _CLIP, None))
                rows.append((pid, tp, stim))
                D.append(lm - lr)

    meta = pd.DataFrame(rows, columns=['patient', 'timepoint', 'stim'])
    return meta, (np.vstack(D) if D else np.empty((0, 0)))


def _boot_key(meta, D, n_boot, seed):
    """Content hash of everything the ribbon depends on."""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(D, dtype=np.float64).tobytes())
    h.update('|'.join(meta['patient'].astype(str)).encode())
    h.update('|'.join(meta['timepoint'].astype(str)).encode())
    h.update('|'.join(meta['stim'].astype(str)).encode())
    h.update(f'twostage-mean-v1|{n_boot}|{seed}'.encode())
    return h.hexdigest()[:16]


def _patient_curves(meta, D, stims):
    """{stim: {patient: (n_freq,) mean of log(mvt/rest) over that patient's
    available sessions in that stim state}}. Stage 1 of the two-stage mean."""
    pid = meta['patient'].to_numpy()
    st  = meta['stim'].to_numpy()
    out = {}
    for s in stims:
        m = st == s
        sp, sd = pid[m], D[m]
        out[s] = {p: sd[sp == p].mean(axis=0) for p in np.unique(sp)}
    return out


def diff_marginal(meta, D, n_boot=N_BOOT_PSD, seed=BOOT_SEED, refresh=False):
    """
    Per-bin movement effect by stim state, as a TWO-STAGE MEAN (no model):
    {stim: (est, lo, hi)} of arrays over frequency bins.

        est[s]  = mean over patients of ( mean over that patient's sessions of
                  log(mvt/rest) ), per bin
        lo/hi   = 2.5 / 97.5 percentiles of that statistic under a CLUSTER
                  BOOTSTRAP: resample whole patients with replacement, recompute
                  the two-stage mean

    One set of patient resamples is drawn ONCE and reused for every bin, so the
    ribbon varies smoothly with frequency rather than picking up independent
    resampling jitter. Targets the same quantity as Panel B's timepoint-averaged
    simple effects and agrees with them to <1% at band level; the mixed model's
    partial pooling is not worth a fit per frequency bin for a descriptive curve.
    """
    n_freq = D.shape[1] if D.size else 0
    stims  = [s for s in ('stimOFF', 'stimON') if (meta['stim'] == s).any()]
    out    = {s: [np.full(n_freq, np.nan) for _ in range(3)] for s in stims}
    if not n_freq:
        return {s: tuple(v) for s, v in out.items()}, _note(0, False)

    curves = _patient_curves(meta, D, stims)

    # ── observed two-stage mean ─────────────────────────────────────────────
    for s in stims:
        pats = sorted(curves[s])
        if pats:
            out[s][0] = np.vstack([curves[s][p] for p in pats]).mean(axis=0)

    # ── cluster bootstrap: one resample set, reused across bins ─────────────
    patients = sorted(set().union(*(set(curves[s]) for s in stims)))
    cache = BOOT_CACHE / f'{_boot_key(meta, D, n_boot, seed)}.npz'
    cached = False
    if n_boot and cache.exists() and not refresh:
        z = np.load(cache)
        for s in stims:
            if f'{s}_lo' in z:
                out[s][1], out[s][2] = z[f'{s}_lo'], z[f'{s}_hi']
        cached = True

    if n_boot and not cached and len(patients) >= 3:
        rng = np.random.default_rng(seed)
        boot = {s: [] for s in stims}
        for _ in range(n_boot):
            drawn = rng.choice(patients, size=len(patients), replace=True)
            for s in stims:
                pick = [curves[s][p] for p in drawn if p in curves[s]]
                if pick:
                    boot[s].append(np.vstack(pick).mean(axis=0))
        for s in stims:
            if boot[s]:
                arr = np.vstack(boot[s])                 # (n_draw, n_freq)
                out[s][1] = np.nanpercentile(arr, 2.5, axis=0)
                out[s][2] = np.nanpercentile(arr, 97.5, axis=0)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, **{f'{s}_{k}': out[s][i]
                           for s in stims for i, k in ((1, 'lo'), (2, 'hi'))})

    return ({s: tuple(v) for s, v in out.items()},
            _note(n_boot, cached))


def _note(n_boot, cached=False):
    """One line for stdout recording how the difference row was built."""
    ci = (f'cluster bootstrap 95% CI, {n_boot} draws' if n_boot
          else 'no CI (bootstrap skipped)')
    if cached:
        ci += ' [cached]'
    return ('difference row: two-stage mean (per-patient session mean, then '
            f'patient mean), {ci}')


def shade_bands(ax):
    """The two bands Panel B reports, drawn first so every line sits on top."""
    for band in HILITE_BANDS:
        lo, hi = FREQ_BANDS[band]
        ax.axvspan(lo, hi, color=BAND_FILL[band], lw=0, zorder=0,
                   alpha=BAND_ALPHA)
        ax.axvline(lo, color=BAND_EDGE[band], lw=0.6, zorder=0)
        ax.axvline(hi, color=BAND_EDGE[band], lw=0.6, zorder=0)


def draw_into(fig, subspec, freqs, spectra, patients, timepoints, fmin, fmax,
              center='geomean', show_diff=True, wspace=0.19, hspace=0.34,
              title=None, n_boot=N_BOOT_PSD, refresh_boot=False):
    """
    Draw the panel into one cell of an existing figure.

    Split out from draw() so make_rest_mvt_combined_figure.py can place this
    panel above the forest panel without either script owning the figure. The
    caller supplies the cell; everything inside it — the grid, the shared axis
    label, the legend — is this function's business.

    Grid: STIMULATION STATE is the row, SESSION is the column, and the
    difference spectrum is one further column on the right. Hue still means
    stimulation, so a row is now a single colour throughout, and the comparison
    the figure exists to show — rest against movement — stays inside one cell
    where any per-session scale factor cancels. Reading across a row compares
    sessions, which is safe for shape and not for absolute level; see the units
    note in the module docstring.

    Returns (axes, axes_diff, n_used, diff_note).
    """
    sel = (freqs >= fmin) & (freqs <= fmax)
    f = freqs[sel]

    stims = [('stimOFF', 'Stimulation OFF', C_OFF),
             ('stimON',  'Stimulation ON',  C_ON)]

    diff_res, diff_note = (None, None)
    if show_diff:
        meta, D = build_diff_rows(spectra, patients, timepoints, sel)
        if len(meta):
            diff_res, diff_note = diff_marginal(meta, D, n_boot=n_boot,
                                                refresh=refresh_boot)
        else:
            show_diff = False

    n_tp = len(timepoints)
    # An empty spacer column sits before the difference column — gridspec's
    # wspace is uniform, and the break between the sessions and their difference
    # is not the same kind of break as the one between two sessions, so it
    # should not be the same width.
    widths  = [1.0] * n_tp + ([SPACER_W, DIFF_W] if show_diff else [])
    n_cols  = len(widths)
    c_diff  = n_cols - 1

    gs = subspec.subgridspec(2, n_cols, width_ratios=widths,
                             wspace=wspace, hspace=hspace)

    # Spectra share x AND y with each other; the difference column shares x with
    # the spectra and y only within itself, keeping its own linear, zero-centred
    # scale. Its ticks sit on the left like every other column's, so wspace has
    # to be wide enough that they read as its own rather than as M6's.
    axes, base = [], None
    for r in range(2):
        row = []
        for c in range(n_tp):
            kw = dict(sharex=base, sharey=base) if base is not None else {}
            ax = fig.add_subplot(gs[r, c], **kw)
            base = base or ax
            row.append(ax)
        axes.append(row)

    axes_diff = []
    if show_diff:
        for r in range(2):
            kw = dict(sharey=axes_diff[0]) if axes_diff else {}
            axes_diff.append(fig.add_subplot(gs[r, c_diff], sharex=base, **kw))

    n_used = {}
    for r, (stim, stim_label, col) in enumerate(stims):
        for c, tp in enumerate(timepoints):
            ax = axes[r][c]
            shade_bands(ax)

            for movement, ls in [('rest', '-'), ('mvt', '--')]:
                cond = f'{movement}_{stim}'
                arr, pids = session_spectra(spectra, patients, cond, tp)
                if not arr.size:
                    continue
                arr = arr[:, sel]
                if center == 'median':
                    mid = np.median(arr, axis=0)
                    lo_b, hi_b = np.percentile(arr, [25, 75], axis=0)
                else:
                    # geometric mean +/- 1 geometric SD, all in log space
                    lg = np.log(np.clip(arr, _CLIP, None))
                    m, sd = lg.mean(axis=0), lg.std(axis=0, ddof=1)
                    mid = np.exp(m)
                    lo_b, hi_b = np.exp(m - sd), np.exp(m + sd)
                ax.fill_between(f, lo_b, hi_b, color=col, alpha=RIBBON_A, lw=0,
                                zorder=2)
                ax.plot(f, mid, color=col, lw=LW_LINE, ls=ls, zorder=3,
                        solid_capstyle='round')
                n_used[(tp, cond)] = len(pids)

            ax.set_yscale('log')
            ax.set_xlim(fmin, fmax)
            # the default locator drops to 20/40 alone once the panel narrows,
            # which is too coarse to place the band edges by eye
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
            # spectra columns only: the difference column is a single curve
            # against a zero line and reads without one
            ax.set_axisbelow(True)
            ax.grid(True, axis='y', color=GRID_COL, lw=GRID_LW, zorder=0)
            for spine in ('top', 'right'):
                ax.spines[spine].set_visible(False)
            ax.spines['left'].set_color(S.RULE)
            ax.spines['bottom'].set_color(S.RULE)
            ax.tick_params(labelsize=7.0)

            # Session name heads its column, on the top row only. The pad lifts
            # it clear of the row subtitle, so the reading order down the top of
            # the panel is session, then stimulation state, then the axes.
            if r == 0:
                ax.set_title(TP_LABEL.get(tp, tp), fontsize=7.5,
                             fontweight='bold', color=S.INK, pad=8)
                # band names once, on the first column only: every column shares
                # the same frequency axis and shows the same shading, so one
                # labelled instance names them all
                if c == 0:
                    for band in HILITE_BANDS:
                        lo, hi = FREQ_BANDS[band]
                        ax.text((lo + hi) / 2, 0.97, band,
                                transform=ax.get_xaxis_transform(), ha='center',
                                va='top', fontsize=6.5, color=S.MUTED)

            # power scale on every row's first column, rather than one shared
            # label centred on the pair: each row is its own reading
            if c == 0:
                ax.set_ylabel(POWER_UNIT, fontsize=7.5, color=S.INK,
                              labelpad=4)
            else:
                # sharey leaves the inner tick labels drawn when the axes are
                # added one at a time, and they land inside the previous column
                ax.tick_params(labelleft=False)

            if r == 0:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel('Frequency (Hz)', fontsize=7.5)

    # ── difference column ─────────────────────────────────────────────────────
    for r, (stim, _stim_label, col) in enumerate(stims):
        if not axes_diff:
            break
        ax = axes_diff[r]
        shade_bands(ax)
        ax.axhline(0.0, color=ZERO_COL, lw=LW_ZERO, zorder=1)

        est_lo_hi = diff_res.get(stim)
        if est_lo_hi is not None:
            est, lo_b, hi_b = est_lo_hi
            ok = np.isfinite(est)
            if ok.any():
                ax.fill_between(f[ok], lo_b[ok], hi_b[ok], color=col,
                                alpha=RIBBON_A, lw=0, zorder=2)
                ax.plot(f[ok], est[ok], color=col, lw=LW_LINE, zorder=3,
                        solid_capstyle='round')

        ax.set_xlim(fmin, fmax)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)
        ax.spines['left'].set_color(S.RULE)
        ax.spines['bottom'].set_color(S.RULE)
        ax.tick_params(labelsize=7.0)

        if r == 0:
            ax.set_title('Movement − Rest', fontsize=7.5,
                         fontweight='bold', color=S.INK, pad=6)
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel('Frequency (Hz)', fontsize=7.5)
        ax.set_ylabel('Movement − Rest', fontsize=7.5,
                      color=S.MUTED, labelpad=4)

    # One legend under the panel carries both encodings: movement as line style
    # in neutral ink, stimulation as hue. The stim keys replace the row
    # subtitles — the rows are already two flat colours, so naming them once
    # here says the same thing as a heading over each without spending a line of
    # vertical space on it twice.
    handles = [
        Line2D([], [], color=S.INK, lw=LW_LINE, ls='-',  label='Rest'),
        Line2D([], [], color=S.INK, lw=LW_LINE, ls='--', label='Movement'),
        Line2D([], [], color=C_OFF,  lw=LW_LINE, label='Stimulation OFF'),
        Line2D([], [], color=C_ON,   lw=LW_LINE, label='Stimulation ON'),
    ]

    # 'Power (a.u.)' belongs to the spectra columns only — the difference column
    # is in log-ratio units and labels itself on the right. The offset is in
    # inches, not figure fraction, so the label keeps the same clearance from
    # the tick labels whatever width the caller gave the panel.
    fig_w_in, fig_h_in = fig.get_size_inches()
    pos_top  = axes[0][0].get_position()

    # centred under the whole panel, clear of the frequency labels
    last  = (axes_diff[1] if axes_diff else axes[1][-1]).get_position()
    first = axes[1][0].get_position()
    fig.legend(handles=handles, loc='upper center',
               bbox_to_anchor=((first.x0 + last.x1) / 2,
                               first.y0 - 0.46 / fig_h_in),
               ncol=len(handles), frameon=False, fontsize=7.0,
               handlelength=2.2, columnspacing=2.0, handletextpad=0.6)

    if title:
        fig.text(pos_top.x0 - 0.62 / fig.get_size_inches()[0],
                 pos_top.y1 + 0.35 / fig_h_in,
                 title, fontsize=9.0, fontweight='bold', color=S.INK,
                 ha='left', va='bottom')

    return axes, axes_diff, n_used, diff_note
