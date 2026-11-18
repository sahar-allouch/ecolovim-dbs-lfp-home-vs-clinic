"""
icc_analysis.py
===============
Intraclass Correlation Coefficient (ICC) analysis for test-retest reliability
of spectral features across timepoints: M3, home, M6.

ICC type: ICC(2,1) — Two-way random effects, single measures, absolute agreement.
    Also known as ICC(A,1) in McGraw & Wong (1996).
    - Raters  = timepoints
    - Targets = patients

Analysis structure
------------------
Pairwise ICC(2,1) for each of the 3 timepoint pairs (M3 vs home, M3 vs M6,
home vs M6), computed separately per feature, condition and channel side.
Each pair is a complete-case analysis on its own two sessions: a patient
enters the M3-home pair on the strength of those two recordings alone.

ICC(3,1) (two-way mixed, consistency) is reported alongside each ICC(2,1) to
separate reliability lost to systematic session offset from reliability lost
to random variability — the home device has a known calibration offset that
ICC(2,1) penalises and ICC(3,1) does not.

Multiplicity
------------
The `p` column (F-test of H0: ICC = 0) is reported UNCORRECTED and is
deliberately never entered into a Benjamini-Hochberg family: no ICC family
exists anywhere in this project.  Reliability is read from the point estimate,
its bootstrap interval and the Koo & Li tier — `p` only says the ICC is
distinguishable from zero, which is a weak claim at n = 8-9 and not the
question these tables are built to answer.  Anything displaying it must label
it uncorrected.  The LME p-values are the
corrected ones, over the per-contrast families lme_analysis.apply_fdr_lme builds.

Feature set is the same 13 that lme_analysis models — ANALYSIS_BANDS for
absolute power, beta only for flat power, plus tremor bands, the aperiodic
trio and the beta peak parameters — so the reliability and contrast tables
carry the same rows and can be read side by side.  Relative power is still
computed for the exploratory figures on the full band list, and never reaches
the statistical tables.

Interval estimation is the percentile bootstrap only (`N_BOOT_ICC` resamples).
Analytic CIs were removed: at n = 8-9 their asymptotics are the questionable
part, and carrying two estimators let different figures print different
bounds for the same ICC.

Interpretation (Koo & Li, 2016):
    ICC < 0.50  : poor
    0.50–0.75   : moderate
    0.75–0.90   : good
    ICC > 0.90  : excellent

Features analysed (per condition, per channel side):
    - Absolute band power, ANALYSIS_BANDS (delta, theta, alpha, beta)
    - Tremor-locked bands, per patient and hemisphere (abs_tremor_f0/_2f)
    - Aperiodic-adjusted (FOOOF-residual) band power
    - Broadband power
    - Aperiodic exponent and offset
    - FOOOF beta-peak parameters (peak_beta_cf/_pw/_bw) — conditional on
      detection, so their n is thinner than everything above
  Relative power is still computed for the exploratory figures but is excluded
  from the statistical analysis.

Usage:
    python -m stats_analysis.icc_analysis
"""

import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.stats as stats
from plotly.subplots import make_subplots
from scipy.optimize import brentq

from stats_analysis.data_utils import (
    ANALYSIS_BANDS,
    COMMON_BANDS,
    CONDITIONS,
    EXCLUSION_NOTE,
    P04_EXCLUSION_NOTE,
    PEAK_BAND,
    PEAK_METRICS,
    build_aperiodic_df,
    build_bandpower_df,
    build_broadband_power_df,
    build_flattened_bandpower_df,
    build_peak_params_df,
    build_tremor_bandpower_df,
    load_all_sessions,
    load_all_sessions_tremor,
)
from stats_analysis.plot_style import (
    _INK,
    _MID,
    COND_COLORS,
    FONT,
    ICC_COLOR_LIST,
    SIZE_ANNOT,
    SIZE_HEAD,
    SIZE_LABEL,
    SIZE_TICK,
    add_caption,
    apply_style,
)
from stats_analysis.plot_style import (
    COND_LABELS as _COND_LABELS_STYLE,
)
from stats_analysis.plot_style import (
    ICC_COLORSCALE as _ICC_CS_STYLE,
)
from stats_analysis.plot_style import (
    save_fig as _save_fig,
)

RESULTS_DIR = os.path.join('Results', 'stats', 'icc')
os.makedirs(RESULTS_DIR, exist_ok=True)

ICC_THRESHOLDS = [0.50, 0.75, 0.90]
ICC_COLORS     = ICC_COLOR_LIST          # [poor, moderate, good, excellent]
_ICC_COLORSCALE = _ICC_CS_STYLE
_COND_LABELS    = _COND_LABELS_STYLE

TP_PAIRS    = [('M3', 'home'), ('M3', 'M6'), ('home', 'M6')]

# Bands that enter the statistical tables, matching lme_analysis exactly so the
# reliability and contrast tables carry the same rows. ANALYSIS_BANDS is
# config.EXCLUDED_BANDS applied to BAND_NAMES; flat power is beta only, for the
# same reason lme_analysis restricts it (outside beta the aperiodic-removed
# residual is negative for a large share of observations, and flat_abs_gamma2
# is structurally all-zero on the [1, 40] Hz fit grid).
ICC_ANALYSIS_BANDS  = ANALYSIS_BANDS
ICC_FLAT_BANDS      = ['beta']
N_BOOT_ICC  = 5000   # bootstrap resamples for ICC(2,1) and ICC(3,1) CIs
MDI_POWER   = 0.80   # target power for minimum detectable ICC (MDI)

# Fixed so reruns reproduce byte-identical CSVs.  Seeding per call also means ICC(2,1) and ICC(3,1) for
# the same cell are bootstrapped over the *same* resamples, so the two
# intervals are directly comparable rather than differing by Monte Carlo noise.
BOOT_SEED   = 0


def icc_power(rho, n, k, alpha=0.05):
    """
    Power to detect ICC = rho vs ICC = 0 using the F-test.

    Approximation: assumes no systematic between-session offset (MS_cols ≈ MS_error).
    Exact for ICC(3,1); upper bound for ICC(2,1) when session bias is present.
    """
    if np.isnan(rho) or rho <= 0 or rho >= 1 or n < 3 or k < 2:
        return np.nan
    df1    = n - 1
    df2    = (n - 1) * (k - 1)
    f_crit = stats.f.ppf(1 - alpha, df1, df2)
    lam    = 1.0 + k * rho / (1.0 - rho)
    return float(1 - stats.f.cdf(f_crit / lam, df1, df2))


def min_detectable_icc(n, k, target_power=MDI_POWER, alpha=0.05):
    """
    Minimum ICC detectable at `target_power` power given n subjects and k raters.
    Solved numerically via Brent's method.
    Returns NaN if no solution in (0, 0.9999).
    """
    if n < 3 or k < 2:
        return np.nan
    try:
        return float(brentq(
            lambda rho: icc_power(rho, n, k, alpha) - target_power,
            1e-4, 0.9999, xtol=1e-4))
    except ValueError:
        return np.nan


# ── ICC(2,1) implementation ───────────────────────────────────────────────────

def icc_2_1(data):
    """
    ICC(2,1) / ICC(A,1): Two-way random, absolute agreement, single measures.
    Works for any number of raters k ≥ 2.

    Parameters
    ----------
    data : ndarray, shape (n_subjects, n_raters)
        Missing values (NaN) are allowed; subjects with any NaN are excluded.

    Returns
    -------
    icc, p  — both float; NaN if insufficient data.

    Notes
    -----
    Interval estimation is the percentile bootstrap (`icc_2_1_bootstrap`), not
    an analytic formula.  The Satterthwaite approximation of McGraw & Wong
    (1996) was removed: at n = 8-9 it is the asymptotic behaviour that is in
    doubt, and carrying two interval estimators meant different figures could
    print different bounds for the same ICC.
    """
    data = np.array(data, dtype=float)
    data = data[~np.isnan(data).any(axis=1)]

    n, k = data.shape
    if n < 3 or k < 2:
        return np.nan, np.nan

    grand_mean  = data.mean()
    subj_means  = data.mean(axis=1)
    rater_means = data.mean(axis=0)

    SS_total = ((data - grand_mean) ** 2).sum()
    SS_rows  = k * ((subj_means  - grand_mean) ** 2).sum()
    SS_cols  = n * ((rater_means - grand_mean) ** 2).sum()
    SS_error = SS_total - SS_rows - SS_cols

    df_rows  = n - 1
    df_cols  = k - 1
    df_error = (n - 1) * (k - 1)

    MS_rows  = SS_rows  / df_rows
    MS_cols  = SS_cols  / df_cols
    MS_error = SS_error / df_error if df_error > 0 else np.nan

    # Guard: catches both zero and small-negative values from floating-point
    # subtraction (SS_error = SS_total - SS_rows - SS_cols can underflow).
    if np.isnan(MS_error) or MS_error <= 0:
        return np.nan, np.nan

    icc = (MS_rows - MS_error) / (
        MS_rows + (k - 1) * MS_error + k * (MS_cols - MS_error) / n
    )

    # ── p-value: H0: ICC = 0, F = MS_rows / MS_error ─────────────────────────
    F_obs = MS_rows / MS_error
    p     = 1 - stats.f.cdf(F_obs, df_rows, df_error)

    return float(icc), float(p)


def icc_3_1(data):
    """
    ICC(3,1) / ICC(C,1): Two-way mixed effects, consistency, single measures.

    Raters (timepoints) are treated as fixed — systematic session offset does
    NOT reduce ICC(3,1) because consistency only requires preserved ranking.
    Compare with ICC(2,1) (absolute agreement).  ICC(3,1) ≥ ICC(2,1) holds for
    the population parameters, but NOT for these single-measure estimates: the
    ICC(A,1) denominator carries k(MS_cols - MS_error)/n, which goes negative
    whenever the observed session mean square falls below the error mean square,
    and the estimated ICC(2,1) then exceeds the estimated ICC(3,1).  This is
    sampling noise around "no offset", not evidence of one, so a negative
    ICC(3,1) - ICC(2,1) gap must be read as zero.  Recount after any rerun rather than
    trusting this line:
        (pd.read_csv('Results/stats/icc/icc_all_pairwise_bilateral.csv')
           .eval('ICC3 < ICC').sum())

    Use case here: home device has a known calibration offset relative to
    in-clinic recordings → ICC(2,1) penalises this offset, ICC(3,1) does not.
    Reporting both quantifies how much of the reliability loss is due to
    systematic bias vs. random variability.

    Parameters
    ----------
    data : ndarray, shape (n_subjects, n_raters)
        NaN rows excluded listwise.

    Returns
    -------
    icc, p  — both float; NaN if insufficient data.
    Intervals come from `icc_3_1_bootstrap`, matching ICC(2,1).
    """
    data = np.array(data, dtype=float)
    data = data[~np.isnan(data).any(axis=1)]
    n, k = data.shape
    if n < 3 or k < 2:
        return np.nan, np.nan

    grand_mean  = data.mean()
    subj_means  = data.mean(axis=1)
    rater_means = data.mean(axis=0)

    SS_total = ((data - grand_mean) ** 2).sum()
    SS_rows  = k * ((subj_means  - grand_mean) ** 2).sum()
    SS_cols  = n * ((rater_means - grand_mean) ** 2).sum()
    SS_error = SS_total - SS_rows - SS_cols

    df_rows  = n - 1
    df_error = (n - 1) * (k - 1)

    MS_rows  = SS_rows  / df_rows
    MS_error = SS_error / df_error if df_error > 0 else np.nan

    if np.isnan(MS_error) or MS_error <= 0:
        return np.nan, np.nan

    # ICC(3,1) — consistency: between-rater variance not included in denominator
    icc = (MS_rows - MS_error) / (MS_rows + (k - 1) * MS_error)

    F_obs = MS_rows / MS_error
    p     = 1 - stats.f.cdf(F_obs, df_rows, df_error)

    return float(icc), float(p)


def _icc_2_1_batch(batch):
    """Vectorised ICC(2,1) over a batch of matrices. batch: (B, n, k) → (B,)."""
    B, n, k = batch.shape
    gm  = batch.mean(axis=(1, 2))              # (B,)
    srm = batch.mean(axis=2)                   # (B, n)  subject means
    scm = batch.mean(axis=1)                   # (B, k)  rater means
    SS_t = ((batch - gm[:, None, None]) ** 2).sum(axis=(1, 2))
    SS_r = k * ((srm - gm[:, None]) ** 2).sum(axis=1)
    SS_c = n * ((scm - gm[:, None]) ** 2).sum(axis=1)
    SS_e = SS_t - SS_r - SS_c
    MS_r = SS_r / (n - 1)
    MS_c = SS_c / (k - 1)
    MS_e = SS_e / ((n - 1) * (k - 1))
    valid = MS_e > 0
    with np.errstate(divide='ignore', invalid='ignore'):
        icc = np.where(valid,
                       (MS_r - MS_e) / (MS_r + (k - 1) * MS_e + k * (MS_c - MS_e) / n),
                       np.nan)
    return icc


def _icc_3_1_batch(batch):
    """Vectorised ICC(3,1) over a batch of matrices. batch: (B, n, k) → (B,)."""
    B, n, k = batch.shape
    gm  = batch.mean(axis=(1, 2))
    srm = batch.mean(axis=2)
    scm = batch.mean(axis=1)
    SS_t = ((batch - gm[:, None, None]) ** 2).sum(axis=(1, 2))
    SS_r = k * ((srm - gm[:, None]) ** 2).sum(axis=1)
    SS_c = n * ((scm - gm[:, None]) ** 2).sum(axis=1)
    SS_e = SS_t - SS_r - SS_c
    MS_r = SS_r / (n - 1)
    MS_e = SS_e / ((n - 1) * (k - 1))
    valid = MS_e > 0
    with np.errstate(divide='ignore', invalid='ignore'):
        icc = np.where(valid,
                       (MS_r - MS_e) / (MS_r + (k - 1) * MS_e),
                       np.nan)
    return icc


def icc_2_1_bootstrap(data, n_boot=N_BOOT_ICC, rng=None):
    """
    Bootstrap 95% CI for ICC(2,1) — vectorised over all resamples at once.

    Generates all n_boot resampled matrices as a single (n_boot, n, k) array
    and computes ICC for each in one numpy pass. ~50–100x faster than a loop.

    Seeded from BOOT_SEED unless an `rng` is supplied, so reruns reproduce.
    """
    if rng is None:
        rng = np.random.default_rng(BOOT_SEED)
    data = np.array(data, dtype=float)
    data = data[~np.isnan(data).any(axis=1)]
    n = data.shape[0]
    if n < 3:
        return np.nan, np.nan

    idx   = rng.integers(0, n, size=(n_boot, n))  # (n_boot, n)
    batch = data[idx]                               # (n_boot, n, k)
    iccs  = _icc_2_1_batch(batch)                  # (n_boot,)
    iccs  = iccs[~np.isnan(iccs)]

    if len(iccs) < 10:
        return np.nan, np.nan
    return float(np.percentile(iccs, 2.5)), float(np.percentile(iccs, 97.5))


def _icc_label(icc):
    if np.isnan(icc):
        return 'n/a', ICC_COLORS[0]
    if icc < ICC_THRESHOLDS[0]:
        return 'poor', ICC_COLORS[0]
    if icc < ICC_THRESHOLDS[1]:
        return 'moderate', ICC_COLORS[1]
    if icc < ICC_THRESHOLDS[2]:
        return 'good', ICC_COLORS[2]
    return 'excellent', ICC_COLORS[3]


def _pivot_and_icc_boot(sub, value_col, timepoints, n_boot=N_BOOT_ICC):
    """
    Pivot a sub-DataFrame and compute ICC(2,1) with a bootstrap 95% CI.

    Returns
    -------
    icc, p, sd_between, ci_lo_boot, ci_hi_boot, n_sub
        sd_between : SD of per-patient mean values across timepoints.
                     Small sd_between relative to the raw feature SD signals
                     range restriction (ICC may be artifically near zero).
    """
    pivot = sub.pivot_table(index='patient', columns='timepoint',
                            values=value_col, aggfunc='mean')
    cols = [tp for tp in timepoints if tp in pivot.columns]
    if len(cols) < 2:
        nan = np.nan
        return nan, nan, nan, nan, nan, 0
    mat = pivot[cols].values
    mat_clean  = mat[~np.isnan(mat).any(axis=1)]
    n_sub      = mat_clean.shape[0]
    row_means  = np.nanmean(mat, axis=1)
    sd_between = float(np.nanstd(row_means, ddof=1)) if len(row_means) > 1 else np.nan
    icc, p                 = icc_2_1(mat)
    ci_lo_boot, ci_hi_boot = icc_2_1_bootstrap(mat, n_boot=n_boot)
    return icc, p, sd_between, ci_lo_boot, ci_hi_boot, n_sub


def icc_3_1_bootstrap(data, n_boot=N_BOOT_ICC, rng=None):
    """Bootstrap 95% CI for ICC(3,1) — vectorised over all resamples at once.

    Seeded from BOOT_SEED unless an `rng` is supplied, so reruns reproduce.
    """
    if rng is None:
        rng = np.random.default_rng(BOOT_SEED)
    data = np.array(data, dtype=float)
    data = data[~np.isnan(data).any(axis=1)]
    n = data.shape[0]
    if n < 3:
        return np.nan, np.nan

    idx   = rng.integers(0, n, size=(n_boot, n))
    batch = data[idx]
    iccs  = _icc_3_1_batch(batch)
    iccs  = iccs[~np.isnan(iccs)]

    if len(iccs) < 10:
        return np.nan, np.nan
    return float(np.percentile(iccs, 2.5)), float(np.percentile(iccs, 97.5))


def _pivot_and_icc3(sub, value_col, timepoints, n_boot=N_BOOT_ICC):
    """Pivot and compute ICC(3,1) + bootstrap CI — consistency model."""
    pivot = sub.pivot_table(index='patient', columns='timepoint',
                            values=value_col, aggfunc='mean')
    cols = [tp for tp in timepoints if tp in pivot.columns]
    if len(cols) < 2:
        return np.nan, np.nan, np.nan, np.nan
    mat = pivot[cols].values
    icc, p     = icc_3_1(mat)
    lo_b, hi_b = icc_3_1_bootstrap(mat, n_boot=n_boot)
    return icc, p, lo_b, hi_b


# ── Range-restriction / degeneracy flags ───────────────────────────────────────

# A bootstrap CI this narrow is not attainable from 8-9 resampled patients; it
# means the resamples are returning the same value because the input is mostly
# ties (e.g. flattened band power floored at log(1e-12)), not because the ICC is
# precisely estimated.
DEGEN_CI_WIDTH = 0.05


def _range_flags(icc, sd_b, total_sd, lo_b, hi_b):
    """-> (degenerate, range_flag) for one ICC cell.

    `degenerate` catches a collapsed input distribution, which the original
    ratio test (sd_between < 10% of total SD) cannot: once every observation sits
    on the same floor, sd_between and total_sd go to zero together, the ratio is
    undefined, and the `total_sd > 0` guard silently returns False. Degeneracy is
    therefore tested on absolute quantities — a NaN ICC, zero variance, or a
    bootstrap CI that has collapsed onto a single value.

    `range_flag` stays the published meaning (restricted between-patient range)
    and is a superset: any degenerate cell is also flagged, so existing plots and
    tables keyed on range_flag surface these instead of showing a bare ICC.
    """
    ci_collapsed = bool(not np.isnan(lo_b) and not np.isnan(hi_b)
                        and (hi_b - lo_b) < DEGEN_CI_WIDTH)
    degenerate = bool(np.isnan(icc)
                      or (not np.isnan(sd_b) and sd_b == 0)
                      or np.isnan(total_sd) or total_sd <= 0
                      or ci_collapsed)
    restricted = bool(not np.isnan(sd_b) and not np.isnan(total_sd)
                      and total_sd > 0 and sd_b < 0.10 * total_sd)
    return degenerate, bool(degenerate or restricted)


# ── Compute ICC tables ─────────────────────────────────────────────────────────

def compute_bandpower_icc(df_bp, conditions=None, bands=None):
    """
    Compute pairwise ICC(2,1) for each (condition, band).

    conditions : list of condition values to iterate over (default: all four CONDITIONS).
    bands      : list of band names (default: COMMON_BANDS). Pass the tremor
                 band names to run the same estimator on the patient-specific
                 bands, whose edges are not in FREQ_BANDS.

    Returns
    -------
    icc_pairwise : DataFrame — one row per (pair, condition, band), pairwise ICC
    """
    if conditions is None:
        conditions = CONDITIONS
    if bands is None:
        bands = COMMON_BANDS
    records_pair = []

    for condition in conditions:
        for band in bands:
            sub = df_bp[(df_bp['condition'] == condition) &
                        (df_bp['band'] == band)]
            total_sd = float(sub['power'].std(skipna=True))

            for tp_a, tp_b in TP_PAIRS:
                icc_p, p_p, sd_b_p, lo_bp, hi_bp, n_p = _pivot_and_icc_boot(
                    sub, 'power', [tp_a, tp_b])
                icc3_p, _, lo3bp, hi3bp = _pivot_and_icc3(sub, 'power', [tp_a, tp_b])
                k_p   = len([tp for tp in [tp_a, tp_b] if tp in
                             sub.pivot_table(index='patient', columns='timepoint',
                                             values='power', aggfunc='mean').columns])
                pwr_p = icc_power(icc_p, n_p, k_p)
                mdi_p = min_detectable_icc(n_p, k_p)
                label_p, _ = _icc_label(icc_p)
                degen_p, range_flag_p = _range_flags(icc_p, sd_b_p, total_sd, lo_bp, hi_bp)
                records_pair.append({
                    'pair': f'{tp_a} vs {tp_b}',
                    'condition': condition, 'band': band,
                    'ICC': icc_p, 'p': p_p,
                    'CI_lo_boot': lo_bp, 'CI_hi_boot': hi_bp,
                    'ICC3': icc3_p,
                    'CI3_lo_boot': lo3bp, 'CI3_hi_boot': hi3bp,
                    'n_subjects': n_p, 'power': pwr_p, 'mdi': mdi_p,
                    'underpowered': bool(not np.isnan(icc_p) and not np.isnan(mdi_p) and icc_p < mdi_p),
                    'rating': label_p, 'sd_between': sd_b_p, 'range_flag': range_flag_p,
                    'degenerate': degen_p,
                })

    return pd.DataFrame(records_pair)


def compute_feature_icc(df_feat, conditions=None, outcomes=('exponent', 'offset')):
    """
    Compute pairwise ICC(2,1) for each (condition, feature).

    Works on any long frame with patient / timepoint / condition columns plus
    one column per feature — nothing here is aperiodic-specific. Used for the
    aperiodic parameters, broadband power, and the FOOOF peak parameters; the
    default `outcomes` reflects its first caller, not a restriction.

    Note the ICC(2,1) computation drops rows with a missing value at either
    timepoint, so the pairwise tables are pairwise complete cases: a patient
    enters the M3-home pair on the strength of those two sessions alone.

    df_feat    : long DataFrame — patient, timepoint, condition, + outcome columns.
    conditions : list of condition values to iterate over (default: all four CONDITIONS).
    outcomes   : tuple of column names to analyse (default: exponent + offset).
                 Pass ('exponent', 'offset', 'broadband_power') to include broadband.
    Returns
    -------
    icc_pairwise : DataFrame — one row per (pair, condition, feature)
    """
    if conditions is None:
        conditions = CONDITIONS
    records_pair = []

    for condition in conditions:
        for outcome in outcomes:
            sub = df_feat[df_feat['condition'] == condition]
            total_sd = float(sub[outcome].std(skipna=True))

            for tp_a, tp_b in TP_PAIRS:
                icc_p, p_p, sd_b_p, lo_bp, hi_bp, n_p = _pivot_and_icc_boot(
                    sub, outcome, [tp_a, tp_b])
                icc3_p, _, lo3bp, hi3bp = _pivot_and_icc3(sub, outcome, [tp_a, tp_b])
                k_p   = len([tp for tp in [tp_a, tp_b] if tp in
                             sub.pivot_table(index='patient', columns='timepoint',
                                             values=outcome, aggfunc='mean').columns])
                pwr_p = icc_power(icc_p, n_p, k_p)
                mdi_p = min_detectable_icc(n_p, k_p)
                label_p, _ = _icc_label(icc_p)
                degen_p, range_flag_p = _range_flags(icc_p, sd_b_p, total_sd, lo_bp, hi_bp)
                records_pair.append({
                    'pair': f'{tp_a} vs {tp_b}',
                    'condition': condition, 'feature': outcome,
                    'ICC': icc_p, 'p': p_p,
                    'CI_lo_boot': lo_bp, 'CI_hi_boot': hi_bp,
                    'ICC3': icc3_p,
                    'CI3_lo_boot': lo3bp, 'CI3_hi_boot': hi3bp,
                    'n_subjects': n_p, 'power': pwr_p, 'mdi': mdi_p,
                    'underpowered': bool(not np.isnan(icc_p) and not np.isnan(mdi_p) and icc_p < mdi_p),
                    'rating': label_p, 'sd_between': sd_b_p, 'range_flag': range_flag_p,
                    'degenerate': degen_p,
                })

    return pd.DataFrame(records_pair)


# ── Figures (Plotly) ───────────────────────────────────────────────────────────


def plot_icc_scatter_grid(df_long, outcome, features, tp_pair=('M3', 'M6'),
                          cond_order=None, title=None, save_name=None):
    """
    Supplementary scatter grid: rows = conditions, cols = features.
    One figure per timepoint pair.

    Parameters
    ----------
    df_long    : long-format DataFrame with 'band' or 'feature' column, 'condition',
                 'timepoint', 'patient', and <outcome> columns
    outcome    : str — dependent variable column name
    features   : list — ordered list of feature labels (bands or aperiodic outcomes)
    tp_pair    : (str, str) — the two timepoints to compare
    cond_order : list of condition keys for rows (default: all four CONDITIONS).
    """
    if cond_order is None:
        cond_order = list(CONDITIONS)
    tp_a, tp_b = tp_pair
    n_rows = len(cond_order)
    n_cols = len(features)
    title_ = title or f'{outcome}  –  {tp_a} vs {tp_b}'
    feat_col = 'band' if 'band' in df_long.columns else 'feature'

    # Condition labels as row titles (includes contrast conditions)
    _COND_LABELS = {
        'rest_stimOFF': 'Rest / Stim OFF',
        'mvt_stimOFF':  'Mvt / Stim OFF',
        'rest_stimON':  'Rest / Stim ON',
        'mvt_stimON':   'Mvt / Stim ON',
        'stimOFF':      'Stim OFF',
        'stimON':       'Stim ON',
    }
    subplot_titles = features + [''] * ((n_rows - 1) * n_cols)

    v_spacing = min(0.06, 0.9 / max(n_rows - 1, 1))
    h_spacing = min(0.008, 0.3 / max(n_cols - 1, 1))

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=subplot_titles,
        row_titles=[_COND_LABELS.get(c, c) for c in cond_order],
        shared_xaxes=False, shared_yaxes=False,
        horizontal_spacing=h_spacing,
        vertical_spacing=v_spacing,
    )

    for row_idx, condition in enumerate(cond_order):
        r = row_idx + 1

        for col_idx, feat in enumerate(features):
            c          = col_idx + 1
            dot_color  = COND_COLORS.get(condition, '#888888')
            sub        = df_long[(df_long[feat_col] == feat) & (df_long['condition'] == condition)]
            s1         = sub[sub['timepoint'] == tp_a].set_index('patient')[outcome]
            s2         = sub[sub['timepoint'] == tp_b].set_index('patient')[outcome]
            common     = s1.index.intersection(s2.index)
            x, y       = s1[common].values, s2[common].values
            patients   = list(common)

            if len(x) == 0:
                continue

            # Data points — patient IDs in hover only
            fig.add_trace(go.Scatter(
                x=x, y=y,
                mode='markers',
                customdata=patients,
                hovertemplate='<b>%{customdata}</b><br>'
                              f'{tp_a}: %{{x:.3f}}<br>{tp_b}: %{{y:.3f}}<extra></extra>',
                marker=dict(
                    color=dot_color, size=9,
                    line=dict(color='white', width=1.2),
                ),
                showlegend=False,
            ), row=r, col=c)

            # Identity line (perfect agreement diagonal)
            if len(x) >= 2:
                span = max(x.max(), y.max()) - min(x.min(), y.min())
                pad  = max(span * 0.10, 1e-9)
                lo   = float(min(x.min(), y.min()) - pad)
                hi   = float(max(x.max(), y.max()) + pad)
                fig.add_shape(
                    type='line', x0=lo, x1=hi, y0=lo, y1=hi,
                    line=dict(color='#999999', width=1.2, dash='dot'),
                    row=r, col=c,
                )

            # r / ICC annotation — bottom-right corner
            if len(x) >= 3:
                r_val, _ = stats.pearsonr(x, y)
                icc_v, _ = icc_2_1(np.column_stack([x, y]))
                ax_idx = row_idx * n_cols + col_idx + 1
                xref = f'x{ax_idx} domain' if ax_idx > 1 else 'x domain'
                yref = f'y{ax_idx} domain' if ax_idx > 1 else 'y domain'
                fig.add_annotation(
                    x=0.97, y=0.04,
                    xref=xref, yref=yref,
                    xanchor='right', yanchor='bottom',
                    text=f'r = {r_val:.2f}   ICC = {icc_v:.2f}',
                    showarrow=False,
                    font=dict(size=9, family=FONT, color=_MID),
                    align='right',
                    bgcolor='rgba(255,255,255,0.85)',
                    bordercolor='#cccccc',
                    borderwidth=1,
                    borderpad=3,
                )

            # Equal axis scale: y anchored to x so the identity line is at 45°
            ax_idx = row_idx * n_cols + col_idx + 1
            x_axis = f'x{ax_idx}' if ax_idx > 1 else 'x'
            y_axis = f'y{ax_idx}' if ax_idx > 1 else 'y'
            fig.update_layout(**{
                f'yaxis{"" if y_axis == "y" else y_axis[1:]}': dict(
                    scaleanchor=x_axis,
                    scaleratio=1,
                )
            })

            # Axis labels: x only on bottom row, y only on left col
            if row_idx == n_rows - 1:
                fig.update_xaxes(title_text=tp_a,
                                 title_font=dict(size=SIZE_LABEL, family=FONT),
                                 row=r, col=c)
            if col_idx == 0:
                fig.update_yaxes(title_text=tp_b,
                                 title_font=dict(size=SIZE_LABEL, family=FONT),
                                 row=r, col=c)

    # ── Global axis style ─────────────────────────────────────────────────────
    apply_style(fig, title=title_,
                height=max(130 * n_rows + 250, 500),
                width=max(160 * n_cols + 200, 800),
                showlegend=False)

    # Fine-tune after apply_style (overrides its defaults where needed)
    fig.update_xaxes(
        tickfont=dict(size=SIZE_TICK, family=FONT, color=_MID),
        showgrid=True, gridcolor='#eeeeee', gridwidth=1,
        zeroline=False,
    )
    fig.update_yaxes(
        tickfont=dict(size=SIZE_TICK, family=FONT, color=_MID),
        showgrid=True, gridcolor='#eeeeee', gridwidth=1,
        zeroline=False,
    )

    # Column-header annotations (subplot_titles, generated by make_subplots)
    for ann in fig.layout.annotations:
        if ann.text in features:
            ann.font    = dict(size=SIZE_HEAD, family=FONT, color=_INK)
            ann.yanchor = 'bottom'

    height = fig.layout.height or 600
    fig.update_layout(title_y=1 - 20 / height)

    # Place row-title annotations left of y-axis label (rotated vertical)
    fig_width = fig.layout.width or 800
    left_margin = 160
    # x in paper coords: -offset/fig_width puts label in left margin area
    label_x = -(left_margin * 0.55) / fig_width
    _row_title_texts = {_COND_LABELS.get(c, c) for c in cond_order}
    for ann in fig.layout.annotations:
        if ann.text in _row_title_texts:
            ann.x         = label_x
            ann.xanchor   = 'center'
            ann.xref      = 'paper'
            ann.textangle = -90
            ann.font      = dict(family=FONT, size=SIZE_HEAD, color=_INK)
    fig.update_layout(margin=dict(l=left_margin, r=30, t=80, b=70))

    add_caption(fig,
        f'Each panel: {tp_a} (x-axis) vs {tp_b} (y-axis) for one condition (row) and feature (column). '
        'Dot colour encodes condition. Dotted diagonal = identity line (perfect agreement). '
        'r = Pearson correlation; ICC = pairwise ICC(2,1). '
        'Points scattered around the diagonal = random noise; '
        'a vertical/horizontal shift = systematic bias between sessions.<br>'
        + EXCLUSION_NOTE + '<br>' + P04_EXCLUSION_NOTE)

    if save_name:
        _save_fig(fig, RESULTS_DIR, save_name)
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

# ── ICC annotated heatmap table ───────────────────────────────────────────────

_ICC_RATING_Z = {
    'n/a':       0.00,
    'poor':      0.25,
    'moderate':  0.50,
    'good':      0.75,
    'excellent': 1.00,
}

# Pastel backgrounds — same hue semantics as ICC_RATING_COLORS but washed out
# so cell text remains readable. Evenly spaced to match _ICC_RATING_Z.
_ICC_TABLE_COLORSCALE = [
    [0.00, '#F5F5F5'],   # n/a     — near-white gray
    [0.25, '#FFFFFF'],   # poor    — white
    [0.50, '#FEF5DF'],   # moderate — warm cream
    [0.75, '#E8F5EC'],   # good    — pale sage
    [1.00, '#DFF0FA'],   # excellent — pale sky
]


# Outline styles for the two cells that must not be read as ordinary ICCs.
# Drawn as cell borders rather than as text so they survive the annotation
# already packed into each cell (ICC, both CIs, power, MDI, rating).
CELL_FLAG_STYLE = {
    'not_estimable':  dict(dash='solid', width=2.4, color='#1A1A1A'),
    'uninformative':  dict(dash='dash',  width=2.0, color='#1A1A1A'),
}


def classify_cells(side, df_pairwise):
    """Flag cells whose ICC must not be read as a reliability estimate.

    Reads Results/stats/icc/sem_all_sides.csv, written by
    stats_analysis.sem_analysis, which holds the two variance components the ICC
    is a ratio of.  Returns {(pair, condition, feature): flag}.

        not_estimable   sd_between clamped to 0 -- MS_rows fell to or below
                        MS_error, so no between-patient variance is recoverable
                        and the printed ICC estimates nothing.
        uninformative   ICC < 0.50 while the within-patient error is BELOW its
                        upper tertile.  The ICC is low without the measurement
                        being noisy, so it reflects a homogeneous cohort (or a
                        session offset), not unreliability.

    The bar is stats_analysis.sem_analysis.SEM_CV_CUTOFF_MDC100 -- fixed,
    the same for L, R and bilateral -- not the per-side empirical tertile an
    earlier version used.  It is anchored to MDC95 >= 100%, not tuned to this
    dataset's distribution; see sem_analysis.py and ICC_CASE_CLASSIFICATION.md
    for the derivation and why a fixed criterion was chosen over a percentile.

    Returns an empty dict if the CSV is absent, so this module keeps running
    standalone -- run `python -m stats_analysis.sem_analysis` first to get the
    outlines.
    """
    path = os.path.join(RESULTS_DIR, 'sem_all_sides.csv')
    if not os.path.exists(path):
        print(f'! {path} not found — ICC table drawn without cell flags; '
              f'run stats_analysis.sem_analysis to add them')
        return {}
    from stats_analysis.sem_analysis import SEM_CV_CUTOFF_MDC100
    sem = pd.read_csv(path)
    sem = sem[sem['side'] == side]
    if sem.empty:
        return {}
    flags = {}
    for r in sem.itertuples():
        key = (r.pair, r.condition, r.feature)
        if r.sd_between == 0:
            flags[key] = 'not_estimable'
        elif r.ICC < 0.50 and r.cv_sem < SEM_CV_CUTOFF_MDC100:
            flags[key] = 'uninformative'
    return flags


def plot_icc_table(df_pairwise, features,
                   title='ICC(2,1) — results table',
                   conditions=None,
                   cell_flags=None,
                   save_name=None):
    """
    Annotated heatmap showing pairwise ICC(2,1) results.

    Layout (mirrors plot_lme_table):
      - One subplot per condition (stacked).
      - Columns  = features (x-axis, -45° labels).
      - Rows     = session pairs: M3 vs home, M3 vs M6, home vs M6.
      - Background color encodes ICC rating (poor → excellent).
      - Cell text: ICC value + bootstrap 95% CI + rating label.

    Parameters
    ----------
    df_pairwise: DataFrame with columns pair, condition, feature, ICC,
                 CI_lo_boot, CI_hi_boot, rating
    features   : ordered list of feature names (column order)
    conditions : list of condition values to show as subplots
                 (default: all in df_pairwise)
    cell_flags : {(pair, condition, feature): 'not_estimable' | 'uninformative'}
                 from classify_cells(); outlines those cells.  None or {} draws
                 the table exactly as before.
    save_name  : if given, saves svg+webp+html under Results/stats/icc/
    """
    cell_flags = cell_flags or {}
    if conditions is None:
        conditions = df_pairwise['condition'].unique().tolist()

    row_defs = [
        ('M3 vs home',  'M3 vs home'),
        ('M3 vs M6',    'M3 vs M6'),
        ('home vs M6',  'home vs M6'),
    ]
    row_labels = [lbl for lbl, _ in row_defs]
    n_cond     = len(conditions)

    fig = make_subplots(
        rows=n_cond, cols=1,
        subplot_titles=[_COND_LABELS.get(c, c) for c in conditions],
        vertical_spacing=0.20 if n_cond <= 2 else 0.10,
    )

    _nan = np.nan

    def _lookup_pair(pair_lbl, cond, feat):
        sub = df_pairwise[
            (df_pairwise['pair'] == pair_lbl) &
            (df_pairwise['condition'] == cond) &
            (df_pairwise['feature'] == feat)
        ]
        if sub.empty:
            return dict(icc=_nan, lb=_nan, hb=_nan,
                        icc3=_nan, lo3b=_nan, hi3b=_nan,
                        pwr=_nan, mdi=_nan, underpowered=False, rating='n/a')
        r = sub.iloc[0]
        return dict(
            icc=r['ICC'],
            lb=r.get('CI_lo_boot', _nan), hb=r.get('CI_hi_boot', _nan),
            icc3=r.get('ICC3', _nan),
            lo3b=r.get('CI3_lo_boot', _nan), hi3b=r.get('CI3_hi_boot', _nan),
            pwr=r.get('power', _nan), mdi=r.get('mdi', _nan),
            underpowered=bool(r.get('underpowered', False)),
            rating=r['rating'])

    for row_idx, cond in enumerate(conditions):
        z_data     = []
        hover_data = []

        for row_lbl, pair_key in row_defs:
            z_row     = []
            hover_row = []
            for feat in features:
                d = _lookup_pair(pair_key, cond, feat)
                z_row.append(_ICC_RATING_Z.get(d['rating'], 0.0))
                hover_row.append(f'<b>{feat}</b> — {row_lbl}')
            z_data.append(z_row)
            hover_data.append(hover_row)

        fig.add_trace(go.Heatmap(
            x=features,
            y=row_labels,
            z=z_data,
            text=hover_data,
            colorscale=_ICC_TABLE_COLORSCALE,
            showscale=False,
            zmin=0, zmax=1,
            xgap=2, ygap=2,
            hovertemplate='%{text}<extra></extra>',
        ), row=row_idx + 1, col=1)

        for row_lbl, pair_key in row_defs:
            for feat in features:
                d = _lookup_pair(pair_key, cond, feat)
                icc                = d['icc']
                lb,  hb            = d['lb'],   d['hb']
                icc3               = d['icc3']
                lo3b, hi3b         = d['lo3b'], d['hi3b']
                pwr, mdi           = d['pwr'],  d['mdi']
                underpowered       = d['underpowered']
                rating             = d['rating']

                if np.isnan(icc):
                    txt = '<i>n/a</i>'
                    fc  = '#CCCCCC'
                else:
                    _flag = ' ⚠' if underpowered else ''
                    _pwr  = (f'pwr={pwr*100:.0f}% MDI={mdi:.2f}{_flag}<br>'
                             if not (np.isnan(pwr) or np.isnan(mdi)) else '')
                    _ci21 = (f' [{lb:.2f}, {hb:.2f}]'
                             if not (np.isnan(lb) or np.isnan(hb)) else '')
                    _ci31 = (f' [{lo3b:.2f}, {hi3b:.2f}]'
                             if not (np.isnan(lo3b) or np.isnan(hi3b)) else '')
                    _i31  = (f'(3,1):{icc3:.2f}{_ci31}<br>'
                             if not np.isnan(icc3) else '')
                    txt = (f'<b>{icc:.2f}{_ci21}</b><br>'
                           + _i31 + _pwr
                           + f'<i>{rating}</i>')
                    fc  = '#CC3300' if underpowered else '#1A1A1A'

                fig.add_annotation(
                    x=feat, y=row_lbl,
                    text=txt,
                    showarrow=False,
                    font=dict(family=FONT, size=SIZE_ANNOT, color=fc),
                    align='center',
                    row=row_idx + 1, col=1,
                )

                # Cell outline.  Categorical axes place category i at i, so the
                # cell spans i +/- 0.5; the 0.02 inset keeps the stroke off the
                # 2 px heatmap gap and stops adjacent flagged cells sharing an
                # edge that reads as one wide box.
                flag = cell_flags.get((pair_key, cond, feat))
                if flag in CELL_FLAG_STYLE:
                    fi = features.index(feat)
                    ri = row_labels.index(row_lbl)
                    fig.add_shape(
                        type='rect',
                        x0=fi - 0.48, x1=fi + 0.48,
                        y0=ri - 0.48, y1=ri + 0.48,
                        line=CELL_FLAG_STYLE[flag],
                        fillcolor='rgba(0,0,0,0)',
                        layer='above',
                        row=row_idx + 1, col=1,
                    )

    fig.update_xaxes(tickangle=-45,
                     tickfont=dict(family=FONT, size=SIZE_TICK),
                     showgrid=False, zeroline=False)
    fig.update_yaxes(autorange='reversed',
                     tickfont=dict(family=FONT, size=SIZE_TICK),
                     showgrid=False, zeroline=False)

    cond_label_set = {_COND_LABELS.get(c, c) for c in conditions}
    for ann in fig.layout.annotations:
        if ann.text in cond_label_set:
            ann.font     = dict(family=FONT, size=SIZE_HEAD, color=_INK)
            ann.x        = 0.0
            ann.xanchor  = 'left'

    cell_h_px      = 85
    cell_w_px      = 140
    per_subplot_h  = len(row_defs) * cell_h_px + 55
    total_h        = n_cond * per_subplot_h + 130
    total_w        = len(features) * cell_w_px + 180

    apply_style(fig, title=title, height=total_h,
                width=total_w, showlegend=False)
    fig.update_layout(margin=dict(l=110, r=20, t=65, b=180))

    if cell_flags:
        add_caption(fig,
            '<b>Solid outline</b> — between-patient variance estimated as zero '
            '(MS_rows &#8804; MS_error): the ICC shown estimates nothing and is '
            'excluded from any reliability classification.<br>'
            '<b>Dashed outline</b> — ICC &lt; 0.50 while the within-patient error '
            'sits below its upper tertile: the ICC is low without the '
            'measurement being noisy, so it reflects a homogeneous cohort or a '
            'session offset rather than unreliability. Read SEM and MDC95 from '
            'sem_all_sides.csv for these cells instead of the ICC.<br>'
            'Both flags are cell-level; the same feature can be flagged in one '
            'condition or session pair and not another.')

    if save_name:
        _save_fig(fig, RESULTS_DIR, save_name)

    return fig


def main(side='L', side_label=None):
    """
    side       : 'L', 'R', or a dict {patient_idx: 'L'/'R'} for per-patient mapping.
    side_label : clean string used in filenames/titles (e.g. 'worse', 'better').
                 Defaults to str(side) when not provided.
    """
    if side_label is None:
        side_label = side if isinstance(side, str) else str(side)
    _s = side_label
    print(f'\nICC Analysis  |  side={_s}  |  rel + abs + aperiodic\n')

    all_sessions = load_all_sessions()

    # ── Load data ─────────────────────────────────────────────────────────────
    df_bp_rel = build_bandpower_df(all_sessions, side=side, power_type='rel')
    df_bp_abs = build_bandpower_df(all_sessions, side=side, power_type='abs')
    df_ap     = build_aperiodic_df(all_sessions, side=side)
    df_bb     = build_broadband_power_df(all_sessions, side=side)
    df_fl_abs = build_flattened_bandpower_df(all_sessions, side=side, power_type='abs')
    # No 'rel' counterpart: relative flattened power is no longer produced at the
    # feature stage (the FOOOF helper stopped returning 'flattened_power_rel'), and
    # flat_rel never reached the statistical tables.  Building it here yielded an
    # empty frame and crashed compute_bandpower_icc on the missing 'condition'.

    # Tremor-locked bands: same estimator, same tables — but the band edges are
    # per patient AND per hemisphere (f0 of the contralateral limb, which can
    # differ between L and R), so a bilateral value averages two different
    # integration windows on purpose. The tremor keys live in the ordinary
    # feature pickles (pipeline.py adds them); absent keys drop the rows
    # rather than crash.
    try:
        df_tr = build_tremor_bandpower_df(load_all_sessions_tremor(), side=side)
    except (FileNotFoundError, OSError) as _exc:
        print(f'! Tremor bands skipped ({type(_exc).__name__}: {_exc}) — '
              f'run add_tremor_bands.py to include them')
        df_tr = None
    tremor_bands = sorted(df_tr['band'].unique()) if df_tr is not None else []

    # FOOOF beta-peak parameters (cf / pw / bw). Conditional on detection and
    # missing-not-at-random, so n varies per cell and is thinner than for the
    # power features — read n_subjects and `underpowered` before quoting a cell.
    # side='bilateral' averages the hemispheres that HAVE a peak and keeps a
    # patient with only one, unlike the band-power builders which need both.
    df_pk = build_peak_params_df(all_sessions, side=side, band=PEAK_BAND)

    # ── Log-transform absolute power features + aperiodic exponent before ICC ─
    # Absolute power is right-skewed / log-normal → log improves normality.
    # Exponent is a positive quantity with no known distribution — log applied.
    # Offset is already in log-µVp² units — left untransformed.
    # Relative power can be zero — not log-transformed.
    # Flattened power is the FOOOF log-space residual (log of the PSD-to-aperiodic
    # ratio) — already a log quantity, and signed, so it is NOT logged again.
    _CLIP = 1e-12   # guard against log(0)
    for _df, _col in [
        (df_bp_abs, 'power'),
        (df_bb,     'broadband_power'),
        (df_ap,     'exponent'),
    ]:
        _df[_col] = np.log(_df[_col].clip(lower=_CLIP))
    if df_tr is not None:   # absolute power, logged like abs_{band}
        df_tr['power'] = np.log(df_tr['power'].clip(lower=_CLIP))

    # Merge broadband into aperiodic df so compute_feature_icc handles it
    _bb_cols = ['patient', 'timepoint', 'condition', 'broadband_power']
    df_ap    = df_ap.merge(df_bb[_bb_cols], on=['patient', 'timepoint', 'condition'], how='left')

    # ── Compute ICCs ─────────────────────────────────────────────────────────
    _AP_OUTCOMES = ('exponent', 'offset', 'broadband_power')

    # 'rel' is exploratory only and never reaches the statistical tables, so it
    # is left on the full band list; abs and flat carry the analysis band set.
    icc_rel_pair    = compute_bandpower_icc(df_bp_rel)
    icc_abs_pair    = compute_bandpower_icc(df_bp_abs, bands=ICC_ANALYSIS_BANDS)
    icc_ap_pair     = compute_feature_icc(df_ap, outcomes=_AP_OUTCOMES)
    icc_fl_abs_pair = compute_bandpower_icc(df_fl_abs, bands=ICC_FLAT_BANDS)
    icc_tr_pair     = (compute_bandpower_icc(df_tr, bands=tremor_bands)
                       if df_tr is not None else None)
    # No transform on any peak metric: cf/bw are Hz, pw is already log10.
    icc_pk_pair     = (compute_feature_icc(df_pk, outcomes=tuple(PEAK_METRICS))
                       if df_pk is not None and not df_pk.empty else None)
    if icc_pk_pair is not None:      # cf -> peak_beta_cf, matching the LME
        icc_pk_pair = icc_pk_pair.copy()
        icc_pk_pair['feature'] = f'peak_{PEAK_BAND}_' + icc_pk_pair['feature']
    peak_features = ([f'peak_{PEAK_BAND}_{m}' for m in PEAK_METRICS]
                     if icc_pk_pair is not None else [])

    print('Rel. band power ICC (pairwise):')
    print(icc_rel_pair.to_string(index=False))
    print('\nAbs. band power ICC (pairwise):')
    print(icc_abs_pair.to_string(index=False))
    print('\nAperiodic ICC (pairwise):')
    print(icc_ap_pair.to_string(index=False))

    # ── Prefix band names: rel_delta, abs_delta, … ───────────────────────────
    def _prefix(df, prefix):
        df = df.copy()
        df['band'] = prefix + '_' + df['band'].astype(str)
        return df.rename(columns={'band': 'feature'})

    icc_rel_pair_f    = _prefix(icc_rel_pair,    'rel')
    icc_abs_pair_f    = _prefix(icc_abs_pair,    'abs')
    icc_fl_abs_pair_f = _prefix(icc_fl_abs_pair, 'flat_abs')
    # abs_tremor_f0 / abs_tremor_2f — same 'abs_' convention as band power.
    icc_tr_pair_f     = (_prefix(icc_tr_pair, 'abs')
                         if icc_tr_pair is not None else None)
    tremor_features   = [f'abs_{b}' for b in tremor_bands]

    # ── Feature lists for each figure set ────────────────────────────────────
    rel_features      = [f'rel_{b}'      for b in COMMON_BANDS]
    abs_features      = [f'abs_{b}'      for b in ICC_ANALYSIS_BANDS]
    flat_abs_features = [f'flat_abs_{b}' for b in ICC_FLAT_BANDS]
    # Solid separator after the band-power block: abs bands + the two tremor
    # bands, which are band power too.
    n_band = len(abs_features) + len(tremor_features)

    # Set A: abs band power + aperiodic + flattened abs
    df_pair_abs = pd.concat([icc_abs_pair_f, icc_ap_pair, icc_fl_abs_pair_f], ignore_index=True)
    feats_abs   = abs_features + ['exponent', 'offset', 'broadband_power'] + flat_abs_features

    # Set B: rel band power + aperiodic (exploratory; never reaches the tables)
    df_pair_rel = pd.concat([icc_rel_pair_f, icc_ap_pair], ignore_index=True)
    feats_rel   = rel_features + ['exponent', 'offset', 'broadband_power']

    # ── ICC tables — all features combined ───────────────────────────────────
    # 'rel' excluded from statistical analysis; 'flat_rel' no longer exists
    feats_all = (abs_features
                 + tremor_features
                 + ['exponent', 'offset', 'broadband_power']
                 + flat_abs_features
                 + peak_features)
    df_pair_all = pd.concat(
        [d for d in (icc_abs_pair_f, icc_tr_pair_f, icc_ap_pair,
                     icc_fl_abs_pair_f, icc_pk_pair) if d is not None],
        ignore_index=True)

    df_pair_all.to_csv(
        os.path.join(RESULTS_DIR, f'icc_all_pairwise_{_s}.csv'), index=False)

    plot_icc_table(
        df_pair_all, features=feats_all,
        title=f'Test-Retest Reliability — ICC(2,1) Table  |  {_s}',
        cell_flags=classify_cells(_s, df_pair_all),
        save_name=f'icc_table_{_s}')

    # ── Scatter grids — one figure per timepoint pair × power type ────────────
    df_bp_abs_long = (df_bp_abs
                      .assign(feature=lambda x: 'abs_' + x['band'])
                      .rename(columns={'power': 'value'})
                      [['patient', 'timepoint', 'condition', 'feature', 'value']])
    df_fl_abs_long = (df_fl_abs
                      .assign(feature=lambda x: 'flat_abs_' + x['band'])
                      .rename(columns={'power': 'value'})
                      [['patient', 'timepoint', 'condition', 'feature', 'value']])
    df_ap_long = (df_ap.melt(id_vars=['patient', 'timepoint', 'condition'],
                             value_vars=list(_AP_OUTCOMES),
                             var_name='feature', value_name='value'))

    df_tr_long = (None if df_tr is None else
                  df_tr.assign(feature=lambda x: 'abs_' + x['band'])
                       .rename(columns={'power': 'value'})
                       [['patient', 'timepoint', 'condition', 'feature', 'value']])

    df_pk_long = (None if df_pk is None or df_pk.empty else
                  df_pk.melt(id_vars=['patient', 'timepoint', 'condition'],
                             value_vars=list(PEAK_METRICS),
                             var_name='feature', value_name='value')
                       .assign(feature=lambda x: f'peak_{PEAK_BAND}_' + x['feature']))

    df_scatter_all = pd.concat(
        [d for d in (df_bp_abs_long, df_tr_long, df_ap_long, df_fl_abs_long,
                     df_pk_long) if d is not None],
        ignore_index=True)

    _tp_labels = {'M3': 'M3', 'home': 'Home', 'M6': 'M6'}
    for tp_a, tp_b in TP_PAIRS:
        la, lb = _tp_labels[tp_a], _tp_labels[tp_b]
        plot_icc_scatter_grid(
            df_scatter_all, outcome='value',
            features=feats_all,
            tp_pair=(tp_a, tp_b),
            title=f'Test-Retest Scatter: {la} vs {lb}  |  {_s}',
            # `_s`, not `side`: for the worse/better runs `side` is the
            # per-patient mapping dict, whose repr became the filename.
            save_name=f'icc_scatter_{tp_a}_{tp_b}_{_s}')

    return (
        icc_rel_pair,
        icc_abs_pair,
        icc_ap_pair,
        {
            'df_pair_abs': df_pair_abs,
            'df_pair_rel': df_pair_rel,
            'feats_abs':   feats_abs,
            'feats_rel':   feats_rel,
            'n_band':      n_band,
        },
    )


if __name__ == '__main__':
    # ── Sides run for ICC ─────────────────────────────────────────────────────
    # 'L', 'R'  : standard hemispheres
    # bilateral : L+R pooled into one value per patient
    #
    # Deliberately NOT run here:
    #   - FTM worse/better side definitions;
    #   - the per-stim-contact variant, which stacked L and R as independent
    #     subjects to reach n=18. Two hemispheres of one patient are not two
    #     independent subjects, so that n was inflated.
    # Both remain available to the other analyses via data_utils.
    _SIDES = [
        ('L',         'L',         'Left'),
        ('R',         'R',         'Right'),
        ('bilateral', 'bilateral', 'Bilateral (L+R)'),
    ]
    side_results = {}
    for _side, _slabel, _label in _SIDES:
        print(f'\n{"─"*60}\nRunning ICC for: {_label}\n{"─"*60}')
        side_results[_label] = main(side=_side, side_label=_slabel)

    print('Done.')
