"""
data_utils.py
=============
Shared data loading and DataFrame builders used by statistical_analysis.py
and icc_analysis.py.
"""

import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
# EXCLUDED_BANDS / ANALYSIS_BANDS are re-exported, not redefined: config.py is
# the single source of truth for which bands enter a statistical model.
from config import BAND_NAMES, EXCLUDED_BANDS, ANALYSIS_BANDS  # noqa: F401
from power import _fix_negative_bins

# ── Configuration ─────────────────────────────────────────────────────────────
BAD_REMOVED = True
DROPPED_LBL = {True: 'badremoved', False: 'badkept'}

CONDITIONS      = ['rest_stimOFF', 'mvt_stimOFF', 'rest_stimON', 'mvt_stimON']
TIMEPOINTS      = ['M3', 'home', 'M6']
NB_PATIENTS     = 10


# P03 (index 2) withdrew after M3 — excluded from all statistical analyses.
EXCLUDED_PATIENTS = {2}
EXCLUSION_NOTE    = 'P03 excluded (withdrew after M3; data available at M3 only).'

# P04 (index 3) has no at-home recordings for stimOFF conditions.
# Excluded from any analysis that involves the home timepoint for those conditions
# (3-way ICC, M3 vs home, home vs M6).  Included in M3 vs M6 pairwise analyses.
HOME_EXCLUDED_BY_COND: dict[str, set[int]] = {
    'rest_stimOFF': {3},
    'mvt_stimOFF':  {3},
}
P04_EXCLUSION_NOTE = (
    'P04 excluded from home-session analyses for rest_stimOFF and mvt_stimOFF '
    '(no at-home recordings available for stimOFF conditions); '
    'P04 is included in M3 vs M6 pairwise comparisons for those conditions.'
)

M3M6_BAND_NAMES = BAND_NAMES
HOME_BAND_NAMES = BAND_NAMES
COMMON_BANDS    = BAND_NAMES
# Flattened (FOOOF-residual) features exclude gamma2: after aperiodic removal
# gamma2 has ~no periodic power → values collapse to the log/rel floor (constant)
# → Pearson r / ICC undefined. Keep gamma2 in abs/rel band power, drop from flat.
FLAT_BANDS      = [b for b in BAND_NAMES if b != 'gamma2']

# Stimulation contact per patient and hemisphere — the single source of truth
# for which channel each analysis reads. Keys are 0-based patient indices, so
# P01 is 0. Resolved to a column of `feat['ch_names']` by get_ch_idx().
# Mapping transcribed from the clinical contact-mapping record.
STIM_CHANNELS = {
    0: {'L': 'L13', 'R': 'R13'},   # Patient 1
    1: {'L': 'L02', 'R': 'R13'},   # Patient 2
    2: {'L': 'L02', 'R': 'R02'},   # Patient 3
    3: {'L': 'L13', 'R': 'R13'},   # Patient 4
    4: {'L': 'L02', 'R': 'R13'},   # Patient 5
    5: {'L': 'L13', 'R': 'R02'},   # Patient 6
    6: {'L': 'L03', 'R': 'R13'},   # Patient 7
    7: {'L': 'L13', 'R': 'R02'},   # Patient 8
    8: {'L': 'L02', 'R': 'R13'},   # Patient 9
    9: {'L': 'L02', 'R': 'R02'},   # Patient 10
}

# ── Load data ─────────────────────────────────────────────────────────────────
def load_session(session):
    if session == 'home':
        path = os.path.join('Results', 'features', 'home_psds_dict.pkl')
    else:
        label = DROPPED_LBL[BAD_REMOVED]
        path  = os.path.join('Results', 'features',
                             f'spect_feats_dict_{session}_{label}.pkl')
    with open(path, 'rb') as f:
        return pickle.load(f)


# ── Sensitivity-analysis patient exclusion ───────────────────────────────────
# Set ECOLOVIM_EXCLUDE_PATIENTS to a comma-separated list of ids ("P07,P08") to
# drop those patients from EVERY long-format builder below, and therefore from
# every downstream model that reads them.  Unset (the default) changes nothing,
# so the published results are unaffected.
#
# This exists for leave-one-out style sensitivity checks where an outlier is
# suspected of driving a reliability estimate.  Anything run with it set writes
# the same filenames as a normal run, so run it in a scratch working directory
# rather than over the real Results tree.
EXCLUDE_PATIENTS = frozenset(
    p.strip() for p in os.environ.get('ECOLOVIM_EXCLUDE_PATIENTS', '').split(',')
    if p.strip()
)


def _drop_excluded(df):
    """Remove EXCLUDE_PATIENTS rows from a long-format builder's output."""
    if df.empty or not EXCLUDE_PATIENTS:
        return df
    return df[~df['patient'].isin(EXCLUDE_PATIENTS)].reset_index(drop=True)


def load_all_sessions():
    return {s: load_session(s) for s in TIMEPOINTS}


def load_all_sessions_tremor():
    return load_all_sessions()


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_ch_idx(feat, patient_idx, side='L'):
    # side can be 'L'/'R' or a dict {patient_idx: 'L'/'R'} for per-patient mapping
    if isinstance(side, dict):
        side = side.get(patient_idx, 'R')
    ch_name = STIM_CHANNELS.get(patient_idx, {}).get(side)
    if ch_name is None or ch_name not in list(feat['ch_names']):
        return None, None
    return list(feat['ch_names']).index(ch_name), ch_name


# ── Bilateral helpers ─────────────────────────────────────────────────────────

def _get_bilateral_bp(feat, i, key, allepochs_key=None):
    """
    (n_bands,) bilateral band power: the equal-weight average of the two
    stimulation contacts, ((L + R) / 2), read from the already-aggregated
    per-channel band power.

    It used to re-derive this from the per-epoch array instead, so that the
    reduction over epochs could be pinned to the same convention FOOOF used.
    That mattered while the reduction was a MEDIAN: reduce-then-sum and
    sum-then-reduce give different numbers for a median, so the stored
    `key` and a recomputation from `allepochs_key` could disagree.

    The reduction is a mean now, and mean and sum commute, so the two routes
    are identical and the recomputation only created a way for the unilateral
    and bilateral values of the same feature to drift apart if one call site
    was changed and the other was not. `allepochs_key` is kept in the
    signature so existing callers still work, and is unused.

    Returns None if either hemisphere channel is missing.
    """
    ch_L, _ = get_ch_idx(feat, i, 'L')
    ch_R, _ = get_ch_idx(feat, i, 'R')
    if ch_L is None or ch_R is None:
        return None
    arr = np.array(feat[key])                            # (n_ch, n_bands)
    if ch_L >= arr.shape[0] or ch_R >= arr.shape[0]:
        return None
    return (arr[ch_L] + arr[ch_R]) / 2.0                 # equal-weight hemispheres


def _get_bilateral_scalar(feat, i, arr):
    """
    Float: (L_value + R_value) / 2 from a 1-D per-channel array.
    Returns None if either channel is missing or out of range.
    """
    ch_L, _ = get_ch_idx(feat, i, 'L')
    ch_R, _ = get_ch_idx(feat, i, 'R')
    if ch_L is None or ch_R is None or ch_L >= len(arr) or ch_R >= len(arr):
        return None
    return (float(arr[ch_L]) + float(arr[ch_R])) / 2.0


def build_tremor_bandpower_df(all_sessions, side='L'):
    """
    Long-format DataFrame: patient, timepoint, condition, band, power, n_bins.

    Absolute power only, in the patient- and hemisphere-specific tremor bands
    written by add_tremor_bands.py. Each hemisphere is integrated over ITS OWN
    frequency window (f0 is taken from the contralateral limb and differs
    between hemispheres in most patients), so the bilateral value deliberately
    mixes two different windows — it is 'tremor-locked power', not power in a
    common band. n_bins travels with the value because a 2 Hz window holds only
    2-3 FFT bins on the 0.977 Hz grid.
    """
    from tremor_bands import TREMOR_BAND_NAMES

    key, ep_key = 'power_abs_tremor_allch', 'power_abs_tremor_allch_allepochs'
    records = []
    for tp in TIMEPOINTS:
        sdict = all_sessions[tp]
        for condition in CONDITIONS:
            for i in range(NB_PATIENTS):
                if i in EXCLUDED_PATIENTS:
                    continue
                if tp == 'home' and i in HOME_EXCLUDED_BY_COND.get(condition, set()):
                    continue
                feat = sdict[condition][i]
                if feat is None or feat.get(key) is None:
                    continue

                nbins = np.array(feat.get('tremor_band_nbins'))
                if side == 'bilateral':
                    bp = _get_bilateral_bp(feat, i, key, ep_key)
                    ch_L, _ = get_ch_idx(feat, i, 'L')
                    ch_R, _ = get_ch_idx(feat, i, 'R')
                    nb = (np.nanmin(np.vstack([nbins[ch_L], nbins[ch_R]]), axis=0)
                          if bp is not None else None)
                else:
                    ch_idx, _ = get_ch_idx(feat, i, side)
                    if ch_idx is None:
                        continue
                    bp = np.array(feat[key])[ch_idx]
                    nb = nbins[ch_idx]
                if bp is None:
                    continue

                for b_idx, band in enumerate(TREMOR_BAND_NAMES):
                    val = float(bp[b_idx])
                    if not np.isfinite(val):
                        continue
                    records.append({
                        'patient':   f'P{i + 1:02d}',
                        'timepoint': tp,
                        'condition': condition,
                        'band':      band,
                        'power':     val,
                        'n_bins':    float(nb[b_idx]) if nb is not None else np.nan,
                    })

    df = _drop_excluded(pd.DataFrame(records))
    if not df.empty:
        df['timepoint'] = pd.Categorical(df['timepoint'],
                                         categories=TIMEPOINTS, ordered=True)
        df['condition'] = pd.Categorical(df['condition'], categories=CONDITIONS)
    return df


# ── DataFrame builders ────────────────────────────────────────────────────────
def build_bandpower_df(all_sessions, side='L', power_type='rel'):
    """
    Long-format DataFrame: patient, timepoint, condition, band, power.
    Only bands common to all three sessions.
    """
    records = []
    for tp in TIMEPOINTS:
        sdict = all_sessions[tp]
        for condition in CONDITIONS:
            for i in range(NB_PATIENTS):
                if i in EXCLUDED_PATIENTS:
                    continue
                if tp == 'home' and i in HOME_EXCLUDED_BY_COND.get(condition, set()):
                    continue
                feat = sdict[condition][i]
                if feat is None:
                    continue
                key = f'power_{"abs" if power_type == "abs" else "rel"}_allch'
                if side == 'bilateral':
                    bp = _get_bilateral_bp(feat, i, key, key + '_allepochs')
                    if bp is None:
                        continue
                else:
                    ch_idx, _ = get_ch_idx(feat, i, side)
                    if ch_idx is None:
                        continue
                    bp = np.array(feat[key])[ch_idx]

                for b_idx, band in enumerate(M3M6_BAND_NAMES):
                    records.append({
                        'patient':   f'P{i+1:02d}',
                        'timepoint': tp,
                        'condition': condition,
                        'band':      band,
                        'power':     float(bp[b_idx]),
                    })

    df = _drop_excluded(pd.DataFrame(records))
    df['timepoint'] = pd.Categorical(df['timepoint'],
                                     categories=TIMEPOINTS, ordered=True)
    df['condition'] = pd.Categorical(df['condition'], categories=CONDITIONS)
    return df


def build_aperiodic_df(all_sessions, side='L'):
    """
    Long-format DataFrame: patient, timepoint, condition, exponent, offset.
    """
    records = []
    for tp in TIMEPOINTS:
        sdict = all_sessions[tp]
        for condition in CONDITIONS:
            for i in range(NB_PATIENTS):
                if i in EXCLUDED_PATIENTS:
                    continue
                if tp == 'home' and i in HOME_EXCLUDED_BY_COND.get(condition, set()):
                    continue
                feat = sdict[condition][i]
                if feat is None:
                    continue
                if feat.get('aperiodic_exponent_allch') is None:
                    continue
                exp_arr = np.array(feat['aperiodic_exponent_allch']).flatten()
                off_arr = np.array(feat['aperiodic_offset_allch']).flatten()
                if side == 'bilateral':
                    exponent = _get_bilateral_scalar(feat, i, exp_arr)
                    offset   = _get_bilateral_scalar(feat, i, off_arr)
                    if exponent is None or offset is None:
                        continue
                else:
                    ch_idx, _ = get_ch_idx(feat, i, side)
                    if ch_idx is None or ch_idx >= len(exp_arr):
                        continue
                    exponent = float(exp_arr[ch_idx])
                    offset   = float(off_arr[ch_idx])
                records.append({
                    'patient':   f'P{i+1:02d}',
                    'timepoint': tp,
                    'condition': condition,
                    'exponent':  exponent,
                    'offset':    offset,
                })

    df = _drop_excluded(pd.DataFrame(records))
    df['timepoint'] = pd.Categorical(df['timepoint'],
                                     categories=TIMEPOINTS, ordered=True)
    df['condition'] = pd.Categorical(df['condition'], categories=CONDITIONS)
    return df


# FOOOF peak-width bounds — see spectral_analysis/frequency_analysis.py:438.
# Bandwidth is hard-censored at these values; a bw sitting on a bound is not a
# measurement, so `bw_at_bound` counts them for the availability report.
PEAK_WIDTH_LIMITS = (2.0, 12.0)
PEAK_METRICS      = ['cf', 'pw', 'bw']
# The one band with enough detected peaks to model — alpha peaks are found too
# rarely. Single source of truth for the modules that fit peak parameters, and
# for the `peak_{band}_{metric}` feature names they emit.
PEAK_BAND         = 'beta'


def _peak_triplet(feat, i, side, band):
    """(cf, pw, bw) for one patient/side/band, or None if the channel is absent.

    None means "no data" (session missing, or this patient has no such channel).
    A tuple of NaN means "channel present, FOOOF found no peak in this band" —
    the caller must keep those apart, because only the second is a detection
    failure and only the second belongs in the detection rate.
    """
    if feat is None:
        return None
    ch_idx, _ = get_ch_idx(feat, i, side)
    if ch_idx is None:
        return None
    band_peaks = feat.get('band_peaks_allch')
    if band_peaks is None or band_peaks.get(band) is None:
        return None

    arr = np.asarray(band_peaks[band])           # (n_epochs, n_ch, 3)
    if arr.shape[0] != 1:
        raise ValueError(
            f"band_peaks_allch['{band}'] has {arr.shape[0]} epochs for "
            f'P{i + 1:02d}/{side} (expected 1 — the PSD is mean-reduced across '
            'epochs before the FOOOF fit).'
        )
    if ch_idx >= arr.shape[1]:
        return None
    return tuple(float(v) for v in arr[0, ch_idx, :])


def build_peak_params_df(all_sessions, side='L', band='beta'):
    """
    Long-format DataFrame: patient, timepoint, condition, cf, pw, bw.

    Same column contract as build_aperiodic_df, so the shared ICC and LME
    drivers accept it unchanged.

        cf  center frequency   (Hz)
        pw  peak power         (log10 above the aperiodic fit — already log)
        bw  bandwidth          (Hz, censored to PEAK_WIDTH_LIMITS by the fit)

    side='bilateral' averages the hemispheres that HAVE a detected peak, and
    keeps the row when only one does. This deliberately differs from
    _get_bilateral_bp / _get_bilateral_scalar, which drop the row unless both
    channels are present: band power exists for every channel, a peak does not,
    so requiring both hemispheres here would discard most of the sample. The
    cost is that a bilateral value is a 2-hemisphere mean for some rows and a
    1-hemisphere value for others — `n_hemi` records which.

    Unlike the other builders, rows are emitted even when no peak was detected
    (cf/pw/bw = NaN) so that detection rates can be computed from this frame.
    Every downstream stats call must .dropna() on the metric it analyses; the
    shared ICC/LME drivers already do.
    """
    records = []
    for tp in TIMEPOINTS:
        sdict = all_sessions[tp]
        for condition in CONDITIONS:
            home_excl = HOME_EXCLUDED_BY_COND.get(condition, set())
            for i in range(NB_PATIENTS):
                if i in EXCLUDED_PATIENTS:
                    continue
                if tp == 'home' and i in home_excl:
                    continue
                feat = sdict[condition][i] if i < len(sdict[condition]) else None

                if side == 'bilateral':
                    triplets = [_peak_triplet(feat, i, s, band) for s in ('L', 'R')]
                    present  = [t for t in triplets if t is not None]
                    if not present:
                        continue
                    det = [t for t in present if np.isfinite(t[0])]
                    vals   = (tuple(float(np.mean([t[m] for t in det]))
                                    for m in range(3)) if det else None)
                    n_hemi = len(det)
                else:
                    triplet = _peak_triplet(feat, i, side, band)
                    if triplet is None:
                        continue
                    vals   = triplet if np.isfinite(triplet[0]) else None
                    n_hemi = 1 if vals is not None else 0

                records.append({
                    'patient':   f'P{i+1:02d}',
                    'timepoint': tp,
                    'condition': condition,
                    'band':      band,
                    'detected':  vals is not None,
                    'n_hemi':    n_hemi,
                    'cf':        vals[0] if vals is not None else np.nan,
                    'pw':        vals[1] if vals is not None else np.nan,
                    'bw':        vals[2] if vals is not None else np.nan,
                })

    df = _drop_excluded(pd.DataFrame(records))
    if df.empty:
        return df
    df['timepoint'] = pd.Categorical(df['timepoint'],
                                     categories=TIMEPOINTS, ordered=True)
    df['condition'] = pd.Categorical(df['condition'], categories=CONDITIONS)
    return df


def build_broadband_power_df(all_sessions, side='L'):
    """
    Long-format DataFrame: patient, timepoint, condition, broadband_power.
    broadband_power = aperiodic_broadband_power_allch (summed aperiodic fit in linear space).
    """
    records = []
    for tp in TIMEPOINTS:
        sdict = all_sessions[tp]
        for condition in CONDITIONS:
            for i in range(NB_PATIENTS):
                if i in EXCLUDED_PATIENTS:
                    continue
                if tp == 'home' and i in HOME_EXCLUDED_BY_COND.get(condition, set()):
                    continue
                feat = sdict[condition][i]
                if feat is None:
                    continue
                if feat.get('aperiodic_broadband_power_allch') is None:
                    continue
                bb_arr = np.array(feat['aperiodic_broadband_power_allch']).flatten()
                if side == 'bilateral':
                    bb_val = _get_bilateral_scalar(feat, i, bb_arr)
                    if bb_val is None:
                        continue
                else:
                    ch_idx, _ = get_ch_idx(feat, i, side)
                    if ch_idx is None or ch_idx >= len(bb_arr):
                        continue
                    bb_val = float(bb_arr[ch_idx])
                records.append({
                    'patient':         f'P{i+1:02d}',
                    'timepoint':       tp,
                    'condition':       condition,
                    'broadband_power': bb_val,
                })

    df = _drop_excluded(pd.DataFrame(records))
    if not df.empty:
        df['timepoint'] = pd.Categorical(df['timepoint'],
                                         categories=TIMEPOINTS, ordered=True)
        df['condition'] = pd.Categorical(df['condition'], categories=CONDITIONS)
    return df


def build_flattened_bandpower_df(all_sessions, side='L', power_type='abs'):
    """
    Long-format DataFrame: patient, timepoint, condition, band, power.
    Uses flattened_power_abs_allch or flattened_power_rel_allch (FOOOF-flattened spectra).
    Shape in pickle: (1, n_ch, n_bands) — the epoch/block dim is squeezed by
    the mean in get_spectral_feat before the FOOOF fit.
    """
    key = f'flattened_power_{"abs" if power_type == "abs" else "rel"}_allch'
    records = []
    for tp in TIMEPOINTS:
        sdict = all_sessions[tp]
        for condition in CONDITIONS:
            for i in range(NB_PATIENTS):
                if i in EXCLUDED_PATIENTS:
                    continue
                if tp == 'home' and i in HOME_EXCLUDED_BY_COND.get(condition, set()):
                    continue
                feat = sdict[condition][i]
                if feat is None:
                    continue
                if feat.get(key) is None:
                    continue
                fp_all = np.array(feat[key])
                if fp_all.ndim == 3:
                    fp_all = np.mean(fp_all, axis=0)  # (n_ch, n_bands)
                if side == 'bilateral':
                    ch_L, _ = get_ch_idx(feat, i, 'L')
                    ch_R, _ = get_ch_idx(feat, i, 'R')
                    if ch_L is None or ch_R is None or ch_L >= fp_all.shape[0] or ch_R >= fp_all.shape[0]:
                        continue
                    fp_vec = (fp_all[ch_L] + fp_all[ch_R]) / 2.0
                else:
                    ch_idx, _ = get_ch_idx(feat, i, side)
                    if ch_idx is None or ch_idx >= fp_all.shape[0]:
                        continue
                    fp_vec = fp_all[ch_idx]

                for b_idx, band in enumerate(BAND_NAMES):
                    if b_idx >= len(fp_vec):
                        continue
                    records.append({
                        'patient':   f'P{i+1:02d}',
                        'timepoint': tp,
                        'condition': condition,
                        'band':      band,
                        'power':     float(fp_vec[b_idx]),
                    })

    df = _drop_excluded(pd.DataFrame(records))
    if not df.empty:
        df['timepoint'] = pd.Categorical(df['timepoint'],
                                         categories=TIMEPOINTS, ordered=True)
        df['condition'] = pd.Categorical(df['condition'], categories=CONDITIONS)
    return df


# The raw spectrum is stored under a different key per modality: the clinic
# pickles keep the per-epoch Welch/FFT amplitude, the at-home pickle keeps the
# Percept FFT bins per recording. Both are amplitude in µVp on the same 100-bin
# axis and both carry the repetition on axis 0, so one code path handles them.
PSD_KEY = {'M3': 'psd_abs_allch', 'M6': 'psd_abs_allch',
           'home': 'fft_bin_data'}


def build_psd_array(all_sessions, side='bilateral', fmin=None, fmax=None):
    """
    Per-patient power spectra, on the same basis as the band-power features.

    Returns (freqs, spectra, patients):
        freqs     (n_freq,)  frequency axis, shared by all three sessions
        spectra   {(timepoint, condition): (n_patient, n_freq) array}
        patients  {(timepoint, condition): [patient id, ...]}  row order

    Amplitude is squared before the repetitions are collapsed — mean(A²) is the
    mean power, mean(A)² underestimates it — and the epoch/recording axis is
    reduced by the mean, matching get_spectral_feat and the band-power
    builders. 'bilateral' then averages the two hemispheres with equal
    weight, the same convention as _get_bilateral_bp, so a spectrum here
    decomposes the band powers the models were fitted on.

    `_fix_negative_bins` runs BEFORE the square, exactly as power.py does before
    the band sums. The device censors the at-home snapshots above ~58 Hz and
    stores the censored bins as negative amplitudes, and squaring turns those into large spurious *positive* power.
    Without this the spectra returned here stopped decomposing the fitted band
    powers anywhere above the censoring point. Clinic spectra hold no negatives,
    so the call is a no-op for them.
    """
    spectra, patients, freq_axis = {}, {}, None

    for tp in TIMEPOINTS:
        sdict = all_sessions[tp]
        key = PSD_KEY[tp]
        for condition in CONDITIONS:
            rows, ids = [], []
            for i in range(NB_PATIENTS):
                if i in EXCLUDED_PATIENTS:
                    continue
                if tp == 'home' and i in HOME_EXCLUDED_BY_COND.get(condition, set()):
                    continue
                feat = sdict[condition][i]
                if feat is None or feat.get(key) is None:
                    continue

                arr = np.asarray(feat[key], dtype=float)   # (n_rep, n_ch, n_freq)
                if arr.ndim != 3:
                    continue
                power = np.mean(_fix_negative_bins(arr) ** 2, axis=0)   # (n_ch, n_freq)

                if side == 'bilateral':
                    ch_L, _ = get_ch_idx(feat, i, 'L')
                    ch_R, _ = get_ch_idx(feat, i, 'R')
                    if (ch_L is None or ch_R is None
                            or max(ch_L, ch_R) >= power.shape[0]):
                        continue
                    spec = (power[ch_L] + power[ch_R]) / 2.0
                else:
                    ch_idx, _ = get_ch_idx(feat, i, side)
                    if ch_idx is None or ch_idx >= power.shape[0]:
                        continue
                    spec = power[ch_idx]

                fr = np.asarray(feat['freqs'], dtype=float)
                if freq_axis is None:
                    freq_axis = fr
                    # Clinic and home store the same grid at different rounding
                    # (they differ by ~0.01 Hz on a 0.98 Hz bin), so the guard
                    # is a fraction of a bin, not floating-point equality. It
                    # still catches a genuinely different grid, where averaging
                    # would blend different frequencies into the same bin.
                    tol = 0.1 * float(np.median(np.diff(freq_axis)))
                elif fr.shape != freq_axis.shape or np.abs(fr - freq_axis).max() > tol:
                    raise ValueError(
                        f'{tp}/{condition}/P{i + 1:02d}: frequency axis differs '
                        'from the other sessions')
                rows.append(spec)
                ids.append(f'P{i + 1:02d}')

            spectra[(tp, condition)] = (np.vstack(rows) if rows
                                        else np.empty((0, 0)))
            patients[(tp, condition)] = ids

    if freq_axis is None:
        raise ValueError('no spectra found in any session')

    if fmin is not None or fmax is not None:
        sel = ((freq_axis >= (fmin if fmin is not None else -np.inf))
               & (freq_axis <= (fmax if fmax is not None else np.inf)))
        freq_axis = freq_axis[sel]
        spectra = {k: (v[:, sel] if v.size else v) for k, v in spectra.items()}

    return freq_axis, spectra, patients
