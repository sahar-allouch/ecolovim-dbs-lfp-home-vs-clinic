"""
lme_rest_mvt_diff.py
====================
Movement effect as a within-session difference score, modelled across sessions.

Design
------
For every patient x timepoint x stim cell, collapse the rest/mvt pair to one
number:

    Diff = mvt - rest        (both already log-transformed, so Diff = log(mvt/rest))

then fit one model per feature:

    Diff ~ C(timepoint, Treatment("home")) + C(stim, Treatment("stimOFF")) + (1 | patient)

Why difference scores rather than a factorial model on the raw values
---------------------------------------------------------------------
Rest and mvt inside one cell come from the same recording session, so they
share the patient's baseline power, the electrode configuration, the impedance
state and the session amplitude scale.  Subtracting cancels all of it exactly.
A factorial model has to *estimate* those nuisance sources from the same data,
and at 9 patients that noise inflates the standard error of every movement
contrast.  Empirically the two give the same point estimates but very
different precision.

Reading the coefficients — every term except the intercept is an interaction
-----------------------------------------------------------------------------
    Intercept        the movement effect itself, at the reference cell
                     (home, stimOFF).  H0: no rest/mvt difference.
    timepoint[M3]    movement x timepoint: does the movement effect change
    timepoint[M6]    between the at-home session and each clinic visit?
    stim[stimON]     movement x stim: does stimulation change the movement
                     effect?

Because Diff is a log ratio, exp(coef) is a multiplicative fold change:
exp(intercept) = 0.8 would mean that movement lowers the feature by 20%.

The random intercept is not the same object as in lme_analysis.py.  There it
absorbs between-patient differences in baseline level; here that has already
been differenced away, so (1|patient) captures between-patient variance in
*movement responsiveness* — the equivalent of a random movement slope in the
factorial model, fittable at this sample size because the nuisance variance is
gone.  Session-level nesting is unnecessary for the same reason: the shared
session effect cancels algebraically rather than needing a variance component.

Inference
---------
Wald p-values assume normal residuals, which 9 patients cannot support.  Each
term therefore also gets a permutation p-value, computed under the null that
belongs to that term and stratified so the *other* factor stays fixed:

    Intercept    sign-flip Diff for whole patients.  Under H0 the movement
                 effect is zero, so each patient's block of differences is
                 symmetric about 0.  Flipping whole patients (not individual
                 rows) preserves within-patient dependence.  With <= 13
                 patients all 2^n sign vectors are enumerated, making the test
                 exact rather than Monte Carlo.
    timepoint    permute timepoint labels within patient x stim (3 rows per
                 stratum), leaving the stim structure intact.
    stim         permute stim labels within patient x timepoint (2 rows per
                 stratum), leaving the timepoint structure intact.

The statistic is |z| for the term, recomputed by refitting the model on each
permuted data set.

Intervals follow the same logic: every CI reported here is a CLUSTER BOOTSTRAP
percentile interval over patients, for model terms and simple effects alike.
Wald intervals rest on the very normal approximation the permutation p-values
exist to avoid, so reporting one beside the other was incoherent — and could
contradict outright (a Wald CI excluding zero beside a non-significant
permutation p).  No Wald quantity is reported by this module any more —
not the p, not the interval.  Everything inferential here is a permutation p or
a cluster-bootstrap interval.

One bootstrap per feature serves both tables: a model term and a simple effect
are both linear contrasts of the same fixed-effect vector, so they are derived
from one set of draws rather than two independent ones.

A percentile CI and a permutation p are still different constructions — the CI
is not the inversion of the test — so they can disagree at the margin.  The
switch was made for coherence: it removes the position of rejecting Wald for
the p-value while relying on it for the interval.  If the two must agree by
construction, p_boot is the p that belongs to these intervals.

Why no Wald fallback at all.  On a true null built by sign-flipping whole
patients, the permutation p was exact, the bootstrap p ran at about twice its
nominal rate (so p_boot is a fast preview and p_perm is the result), and a Wald
p extrapolated to tail probabilities orders of magnitude smaller than the exact
p, in a region that nine patients never sample.

Multiple comparisons
--------------------
BH FDR across features, separately within each term and each side.  Correcting
a term against itself across features is the family that matches how these are
read ("which bands show a movement x stim interaction?").

The simple effects carry their own correction, BH across features separately
within each stimulation state (two families of 13 — see
apply_fdr_simple_effects for why not one of 26).  This answers a question no
term family covers: whether a movement effect exists *within* a given state.

BH needs independence or positive dependence; the Diff values of these 13
features are not all positively correlated, so PRDS is not verified and BH is used as the field convention rather than because its
assumption holds here.

No correction is applied across the side definitions.  They are the same
recordings reshuffled rather than independent tests, so a correction across them
would penalise a re-read of one dataset as if it were three.  bilateral is the
pre-specified primary side; L and R are reported as supporting detail.

Outputs
-------
    Results/stats/lme_rest_mvt/lme_rest_mvt_diff_{side}.csv    coefficient table
    Results/stats/lme_rest_mvt/diff_long_{side}.csv            the Diff values

Usage
-----
    uv run python stats_analysis/lme_rest_mvt_diff.py
    uv run python stats_analysis/lme_rest_mvt_diff.py --sides bilateral --n-perm 2000
    uv run python stats_analysis/lme_rest_mvt_diff.py --no-perm      # bootstrap only, fast
"""

from __future__ import annotations

import os
import sys
import argparse
import itertools
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from statsmodels.stats.multitest import multipletests
from joblib import Parallel, delayed, parallel_backend

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from stats_analysis.data_utils import (
    load_all_sessions,
    load_all_sessions_tremor,
    build_bandpower_df,
    build_aperiodic_df,
    build_broadband_power_df,
    build_flattened_bandpower_df,
    build_tremor_bandpower_df,
    build_peak_params_df,
)

RESULTS_DIR = os.path.join('Results', 'stats', 'lme_rest_mvt')
os.makedirs(RESULTS_DIR, exist_ok=True)

ALPHA   = 0.05
_CLIP   = 1e-12

# Flattened (FOOOF-residual) power is a log-space residual and is negative
# wherever the spectrum sits below the aperiodic fit, so it is never
# log-transformed again.  Only beta is modelled (FLAT_ANALYSIS_BANDS): it is the
# one band with a reliably detected periodic peak.
INCLUDE_FLAT = True     # beta only — see FLAT_ANALYSIS_BANDS

# Same feature set as lme_analysis.py and icc_analysis.py, so the three result
# tables report the same rows and can be read side by side.
ABS_BANDS = ['delta', 'theta', 'alpha', 'beta']

# Flattened power for BETA ONLY: beta is the band where a peak is actually
# detected; in the other bands the aperiodic-removed residual is dominated by
# noise around zero.
FLAT_ANALYSIS_BANDS = ['beta']

PEAK_BAND    = 'beta'
PEAK_METRICS = ['cf', 'pw', 'bw']

# Groups, in the order the ICC/LME table figures use them.
FEATURE_GROUPS = [
    ('Absolute band power',                    [f'abs_{b}' for b in ABS_BANDS]),
    ('Tremor-locked bands (patient-specific)', ['abs_tremor_f0', 'abs_tremor_2f']),
    ('Aperiodic & broadband',                  ['exponent', 'offset',
                                                'broadband_power']),
    ('Beta periodic features (FOOOF)',         [f'peak_{PEAK_BAND}_{m}'
                                                for m in PEAK_METRICS]
                                               + [f'flat_abs_{b}'
                                                  for b in FLAT_ANALYSIS_BANDS]),
]

# Diff is a difference of whatever scale the feature is already on, so the units
# are NOT interchangeable and must not share one axis unexamined:
#   ln      natural-log ratio  — log(mvt/rest); every power feature
#   log10   base-10 log ratio  — FOOOF offset and peak power are already log10
#   Hz      a frequency shift  — peak centre frequency and bandwidth
# The CSV keeps each feature in its native scale; converting is the figure's job.
FEATURE_UNITS = {
    **{f'abs_{b}': 'ln' for b in ABS_BANDS},
    'abs_tremor_f0': 'ln', 'abs_tremor_2f': 'ln',
    'exponent': 'ln', 'broadband_power': 'ln',
    # log10, not ln: flattened power is the FOOOF log-space residual
    # (fm.power_spectrum - fm._ap_fit) and FOOOF works in log10.  It is no
    # longer log-transformed by this module, so the stored value keeps FOOOF's
    # base — the same situation as the aperiodic offset below.
    **{f'flat_abs_{b}': 'log10' for b in FLAT_ANALYSIS_BANDS},
    'offset': 'log10',
    f'peak_{PEAK_BAND}_pw': 'log10',
    f'peak_{PEAK_BAND}_cf': 'Hz',
    f'peak_{PEAK_BAND}_bw': 'Hz',
}


def feature_group(feature):
    """Which GROUPS section a feature belongs to ('' if unlisted)."""
    for name, feats in FEATURE_GROUPS:
        if feature in feats:
            return name
    return ''

N_PERM  = 5000          # Monte-Carlo draws for the label-permutation schemes
PERM_SEED = 0           # fixed so reruns reproduce identical CSVs
EXHAUSTIVE_MAX_PATIENTS = 13   # 2^13 = 8192 sign vectors; above this, sample

FORMULA = ('diff ~ C(timepoint, Treatment("home")) '
           '+ C(stim, Treatment("stimOFF"))')

# model term -> (short label, permutation scheme)
TERMS = {
    'Intercept':                               ('movement (mvt-rest)', 'signflip'),
    'C(timepoint, Treatment("home"))[T.M3]':   ('x timepoint M3',      'timepoint'),
    'C(timepoint, Treatment("home"))[T.M6]':   ('x timepoint M6',      'timepoint'),
    'C(stim, Treatment("stimOFF"))[T.stimON]': ('x stim ON',           'stim'),
}

# The FTM worse/better side definitions are deliberately NOT run here.  They are
# a per-patient relabelling of L and R, not independent data: 'better' reproduces
# L exactly for every patient whose worse side is R, and vice versa.  Reporting
# them beside L and R multiply-counts the same recordings.  Four of ten patients
# have equal FTM subscores, so their assignment is a tie-break rather than a
# clinical fact.  The mappings stay in data_utils for any analysis that wants them.
SIDES = [
    ('L',         'L',         'Left hemisphere'),
    ('R',         'R',         'Right hemisphere'),
    ('bilateral', 'bilateral', 'Bilateral ((L+R)/2)'),
]


# ── Building the difference scores ────────────────────────────────────────────

def build_diff_df(df, outcome):
    """
    Collapse a long condition-level frame to one Diff row per
    patient x timepoint x stim.

    A cell contributes only when BOTH rest and mvt are present — the pairing is
    the whole point, so an unmatched half carries no information about the
    movement effect and is dropped.
    """
    d = df.copy()
    d['cond_str'] = d['condition'].astype(str)
    d['movement'] = d['cond_str'].str.split('_').str[0]
    d['stim']     = d['cond_str'].str.split('_').str[1]

    wide = d.pivot_table(index=['patient', 'timepoint', 'stim'],
                         columns='movement', values=outcome, observed=True)
    if not {'rest', 'mvt'}.issubset(wide.columns):
        return pd.DataFrame()
    wide = wide.dropna(subset=['rest', 'mvt'])
    wide['diff'] = wide['mvt'] - wide['rest']
    out = wide.reset_index()[['patient', 'timepoint', 'stim', 'rest', 'mvt', 'diff']]
    out['timepoint'] = out['timepoint'].astype(str)
    return out


# ── Model fitting ─────────────────────────────────────────────────────────────

def _fit(diff_df):
    """Fit the difference model, quietly. Returns the result or None."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            return smf.mixedlm(FORMULA, diff_df,
                               groups=diff_df['patient']).fit(reml=True)
        except Exception:
            return None


def _fit_ols(diff_df):
    """OLS sensitivity fit — the random intercept is weakly identified at n=9,
    so a reader should be able to see whether it moved the fixed effects."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            return smf.ols(FORMULA, diff_df).fit()
        except Exception:
            return None


def _z_stats(result):
    """{term: |z|} for the modelled terms, nan where a term is absent."""
    out = {}
    for term in TERMS:
        try:
            out[term] = abs(float(result.params[term] / result.bse[term]))
        except (KeyError, TypeError, ZeroDivisionError):
            out[term] = np.nan
    return out


# ── Permutation ───────────────────────────────────────────────────────────────

def _sign_flip_vectors(patients, rng, n_perm):
    """
    Sign vectors over whole patients.  Enumerated exhaustively when there are
    few enough patients (making the test exact), otherwise sampled.
    The all-positive vector is the observed data and is included, as it must be
    for the permutation p-value to be valid.
    """
    n = len(patients)
    if n <= EXHAUSTIVE_MAX_PATIENTS:
        return [np.array(v, dtype=float) for v in itertools.product([1.0, -1.0], repeat=n)]
    vecs = [np.ones(n)]
    vecs += [rng.choice([1.0, -1.0], size=n) for _ in range(n_perm - 1)]
    return vecs


def _permute_labels(df, col, strata_cols, rng):
    """
    Shuffle `col` within each stratum, holding the other factor fixed.
    Returns a copy with the permuted column.
    """
    out = df.copy()
    vals = out[col].to_numpy(copy=True)
    for _, idx in out.groupby(strata_cols, observed=True, sort=False).indices.items():
        # size-1 strata (e.g. P04's stim-OFF cell exists only at M3/M6, so its
        # home row is absent) have nothing to permute and are left as-is — the
        # label there stays fixed across every draw.
        if len(idx) > 1:
            vals[idx] = rng.permutation(vals[idx])
    out[col] = vals
    return out


def permutation_pvalues(diff_df, observed_z, n_perm=N_PERM, seed=PERM_SEED):
    """
    Permutation p for every term, each under the null that belongs to it.

    Sign-flip block (Intercept): the 2^n sign vectors are enumerated in full
    (n <= EXHAUSTIVE_MAX_PATIENTS), the observed all-+1 vector among them, so

        p = #{|z_flip| >= |z_obs|} / 2^n          (exact; >= 2 / 2^n two-sided)

    Label-permutation blocks (interactions): the observed data is seeded as one
    draw and n_perm-1 random permutations are added, so

        p = (1 + #{|z_perm| >= |z_obs|}) / (1 + #{fits that converged})

    Either way the observed data counts as one of its own permutations, which
    keeps the test valid and stops p from being exactly 0.

    Returns ({term: p}, {term: n_draws_used}).
    """
    rng = np.random.default_rng(seed)
    patients = sorted(diff_df['patient'].unique())
    p_out, n_out = {}, {}

    # ── sign-flip block: tests the Intercept (the movement effect itself) ─────
    sign_terms = [t for t, (_, s) in TERMS.items() if s == 'signflip']
    if sign_terms:
        vecs  = _sign_flip_vectors(patients, rng, n_perm)
        pos   = {p: (diff_df['patient'] == p).to_numpy() for p in patients}
        count = {t: 0 for t in sign_terms}
        used  = 0
        for vec in vecs:
            perm = diff_df.copy()
            d    = perm['diff'].to_numpy(copy=True)
            for p_i, s in zip(patients, vec):
                d[pos[p_i]] *= s
            perm['diff'] = d
            res = _fit(perm)
            if res is None:
                continue
            used += 1
            zs = _z_stats(res)
            for t in sign_terms:
                if not np.isnan(zs[t]) and not np.isnan(observed_z[t]) \
                        and zs[t] >= observed_z[t] - 1e-12:
                    count[t] += 1
        for t in sign_terms:
            p_out[t] = count[t] / used if used else np.nan
            n_out[t] = used

    # ── label-permutation blocks: test the interactions ───────────────────────
    for scheme, col, strata in [('timepoint', 'timepoint', ['patient', 'stim']),
                                ('stim',      'stim',      ['patient', 'timepoint'])]:
        terms = [t for t, (_, s) in TERMS.items() if s == scheme]
        if not terms:
            continue
        count = {t: 1 for t in terms}     # observed data counts as one draw
        used  = 1
        for _ in range(n_perm - 1):
            perm = _permute_labels(diff_df, col, strata, rng)
            res  = _fit(perm)
            if res is None:
                continue
            used += 1
            zs = _z_stats(res)
            for t in terms:
                if not np.isnan(zs[t]) and not np.isnan(observed_z[t]) \
                        and zs[t] >= observed_z[t] - 1e-12:
                    count[t] += 1
        for t in terms:
            p_out[t] = count[t] / used if used else np.nan
            n_out[t] = used

    return p_out, n_out


# ── Per-feature driver ────────────────────────────────────────────────────────

def analyse_feature(diff_df, feature, side_label, do_perm=True,
                    n_perm=N_PERM, seed=PERM_SEED, draws=None):
    """
    Fit one feature and return one row per model term.

    `ci_lo`/`ci_hi` are the CLUSTER BOOTSTRAP percentile interval, matching the
    permutation p beside them in refusing the normal approximation that 9
    patients cannot support.  They used to be Wald, which was incoherent twice
    over: the module rejects Wald for the p-value on exactly the assumption the
    Wald interval rests on, and the two could contradict each other outright.  No Wald interval is reported any more.

    A percentile CI and a permutation p are still different constructions and can
    disagree at the margin; that is a real limitation, not a bug, and belongs in
    the caption.
    """
    n_obs = len(diff_df)
    n_pat = diff_df['patient'].nunique()
    if n_obs < 6 or n_pat < 3:
        return []

    res = _fit(diff_df)
    if res is None:
        return []
    ols = _fit_ols(diff_df)

    try:
        group_var = float(res.cov_re.iloc[0, 0])
    except Exception:
        group_var = np.nan
    scale = float(getattr(res, 'scale', np.nan))
    icc   = (group_var / (group_var + scale)
             if np.isfinite(group_var) and np.isfinite(scale) and (group_var + scale) > 0
             else np.nan)

    obs_z = _z_stats(res)
    if do_perm:
        p_perm, n_draws = permutation_pvalues(diff_df, obs_z,
                                              n_perm=n_perm, seed=seed)
    else:
        p_perm, n_draws = {t: np.nan for t in TERMS}, {t: 0 for t in TERMS}

    rows = []
    for term, (label, scheme) in TERMS.items():
        if term not in res.params.index:
            continue
        coef = float(res.params[term])
        se   = float(res.bse[term])
        # the contrast that selects this one coefficient out of the draws
        b_lo, b_hi, n_boot_ok, p_boot = ci_from_draws(draws, {term: 1.0})
        rows.append({
            'feature':     feature,
            'group':       feature_group(feature),
            'unit':        FEATURE_UNITS.get(feature, 'ln'),
            'side_label':  side_label,
            'term':        label,
            'coef':        round(coef, 5),
            'se':          round(se, 5),
            # the permutation test statistic (signed coef/se); every permuted
            # refit compares |z_stat| to the observed one. Not reported as a
            # Wald quantity — the p and CI beside it are permutation/bootstrap.
            'z_stat':      round(coef / se, 3) if se else np.nan,
            'ci_lo':       b_lo,
            'ci_hi':       b_hi,
            'ci_kind':     'cluster bootstrap' if np.isfinite(b_lo) else 'none',
            'n_boot_ok':   n_boot_ok,
            'p_boot':      p_boot,
            'exp_coef':    _fold(coef, FEATURE_UNITS.get(feature, 'ln')),
            'p_perm':      (round(float(p_perm[term]), 5)
                            if np.isfinite(p_perm.get(term, np.nan)) else np.nan),
            'perm_scheme': scheme,
            'n_perm':      n_draws.get(term, 0),
            'coef_ols':    (round(float(ols.params[term]), 5)
                            if ols is not None and term in ols.params.index else np.nan),
            'n_obs':       n_obs,
            'n_patients':  n_pat,
            'group_var':   round(group_var, 5) if np.isfinite(group_var) else np.nan,
            'scale':       round(scale, 5) if np.isfinite(scale) else np.nan,
            'icc':         round(icc, 3) if np.isfinite(icc) else np.nan,
        })
    return rows


# ── Feature assembly ──────────────────────────────────────────────────────────

# _log_flat_power() was removed here.  It log-transformed flattened power and
# folded negative residuals to |x|, with a guard that refused to run when a
# negative was larger in magnitude than the smallest positive value.  Both the
# transform and the guard are obsolete: flattened power is now the FOOOF
# log-space residual (log PSD − log aperiodic), so it is already a log quantity
# and a negative value means "below the aperiodic fit", which is meaningful and
# must keep its sign rather than be folded or clipped.


def _clip_warn(series, label):
    """Flattened power is a FOOOF residual and can be negative; logging it sends
    those values to log(1e-12) = -27.6. Say so loudly rather than silently."""
    bad = int((series.dropna() <= _CLIP).sum())
    if bad:
        print(f'    ! {label}: {bad} value(s) at/below the log clip floor '
              f'({_CLIP:g}) -> approx -27.6 after log')


def build_all_diffs(all_sessions, side):
    """{feature_name: diff DataFrame} for every modelled feature."""
    out = {}

    # ── 1. Absolute band power ────────────────────────────────────────────────
    df_bp = build_bandpower_df(all_sessions, side=side, power_type='abs')
    _clip_warn(df_bp['power'], 'abs band power')
    df_bp['power'] = np.log(df_bp['power'].clip(lower=_CLIP))
    for band in [b for b in ABS_BANDS if b in df_bp['band'].unique()]:
        sub = df_bp[df_bp['band'] == band].dropna(subset=['power'])
        out[f'abs_{band}'] = build_diff_df(sub, 'power')

    # ── 2. Tremor-locked bands ────────────────────────────────────────────────
    # The tremor keys live in the ordinary feature pickles (pipeline.py adds
    # them); skipped with a notice rather than a crash when absent.
    try:
        tremor_sessions = load_all_sessions_tremor()
        df_tr = build_tremor_bandpower_df(tremor_sessions, side=side)
        if not df_tr.empty:
            _clip_warn(df_tr['power'], 'tremor band power')
            df_tr['power'] = np.log(df_tr['power'].clip(lower=_CLIP))
            for band in df_tr['band'].unique():
                sub = df_tr[df_tr['band'] == band].dropna(subset=['power'])
                out[f'abs_{band}'] = build_diff_df(sub, 'power')
    except Exception as exc:
        print(f'    ! tremor bands skipped ({type(exc).__name__}: {exc}) — '
              f'run add_tremor_bands.py to include them', flush=True)

    # ── 3. Aperiodic & broadband ──────────────────────────────────────────────
    # exponent is logged like the power features; offset is left raw because
    # FOOOF already reports it in log10(power) units.
    df_ap = build_aperiodic_df(all_sessions, side=side)
    _clip_warn(df_ap['exponent'], 'aperiodic exponent')
    df_ap['exponent'] = np.log(df_ap['exponent'].clip(lower=_CLIP))
    for outcome in ('exponent', 'offset'):
        out[outcome] = build_diff_df(df_ap.dropna(subset=[outcome]), outcome)

    df_bb = build_broadband_power_df(all_sessions, side=side)
    _clip_warn(df_bb['broadband_power'], 'broadband power')
    df_bb['broadband_power'] = np.log(df_bb['broadband_power'].clip(lower=_CLIP))
    out['broadband_power'] = build_diff_df(
        df_bb.dropna(subset=['broadband_power']), 'broadband_power')

    # ── 4. Beta periodic features (FOOOF) ─────────────────────────────────────
    # cf and bw are in Hz and pw is already log10 — none of them are logged here,
    # and FEATURE_UNITS records the scale so a figure cannot silently put them on
    # the same axis as the log-ratio features.
    try:
        df_pk = build_peak_params_df(all_sessions, side=side, band=PEAK_BAND)
        for m in PEAK_METRICS:
            if m in df_pk.columns:
                out[f'peak_{PEAK_BAND}_{m}'] = build_diff_df(
                    df_pk.dropna(subset=[m]), m)
    except Exception as exc:
        print(f'    ! peak parameters skipped ({type(exc).__name__}: {exc})',
              flush=True)

    # Flattened power — beta only, see FLAT_ANALYSIS_BANDS.
    # Not log-transformed: flattened power is the FOOOF log-space residual
    # (log PSD − log aperiodic), so Diff = mvt − rest is already a log ratio,
    # exactly like the logged features above.
    if INCLUDE_FLAT:
        df_fl = build_flattened_bandpower_df(all_sessions, side=side, power_type='abs')
        df_fl = df_fl[df_fl['band'].isin(FLAT_ANALYSIS_BANDS)].copy()
        for band in [b for b in FLAT_ANALYSIS_BANDS if b in df_fl['band'].unique()]:
            sub = df_fl[df_fl['band'] == band].dropna(subset=['power'])
            out[f'flat_abs_{band}'] = build_diff_df(sub, 'power')

    return {k: v for k, v in out.items() if v is not None and not v.empty}


# ── Simple effects: the movement effect within each stim state ────────────────

N_BOOT = 2000   # cluster-bootstrap resamples per feature, shared by the
                # coefficient table and the simple effects


def _contrast(result, weights):
    """
    Linear contrast c'beta of the fixed effects, with SE = sqrt(c'Vc).

    The covariance term is the reason this cannot be done from the coefficient
    table: the movement effect under stim ON is Intercept + stimON, and the SE of
    that sum needs Cov(Intercept, stimON), which a table of marginal SEs does not
    carry.  Returns (estimate, se).
    """
    names = list(result.params.index)
    c = np.array([weights.get(n, 0.0) for n in names])
    V = result.cov_params().values
    return float(c @ result.params.values), float(np.sqrt(c @ V @ c))


def _fold(value, unit):
    """
    Back-transform an estimate to a multiplicative fold change, or NaN when the
    feature is not on a log scale.

    This used to be exp() applied unconditionally, which silently produced
    nonsense for the features that are not log ratios: peak centre frequency is a
    shift in Hz, so exp(1.57 Hz) came out as a 4.82x "fold change", and peak
    power is in log10, so exp() under-reported it by a factor of ln(10).  A Hz
    shift has no fold-change form at all — it returns NaN rather than a number
    that reads as one.
    """
    if not np.isfinite(value):
        return np.nan
    if unit == 'ln':
        return round(float(np.exp(value)), 4)
    if unit == 'log10':
        return round(float(10.0 ** value), 4)
    return np.nan          # 'Hz' — a frequency shift, not a ratio


def cluster_bootstrap_draws(diff_df, n_boot=N_BOOT, seed=PERM_SEED, fit_fn=_fit):
    """
    Refit the model on patient-resampled data; return the fixed effects per draw.

    `fit_fn` is the model fitter: `_fit` (full model) for the coefficient-table
    CIs, `_fit_reduced` for a simple effect fitted on one stim state's rows.

    Resampling whole patients (not rows) keeps the within-patient dependence that
    the random intercept models.  Duplicated patients are relabelled so statsmodels
    treats each draw as its own group rather than merging them.

    The draws are returned rather than reduced, because EVERY interval this module
    reports is a linear function of the same fixed-effect vector: a model term is
    the contrast that selects one coefficient, a simple effect is the contrast
    that averages the sessions.  Bootstrapping once and deriving both means the
    coefficient table and the figure quote the same 2000 resamples instead of two
    independent sets, and it halves the fitting work.

    Returns a DataFrame, one row per successful draw, one column per fixed effect
    (empty if the model could not be fitted).
    """
    patients = sorted(diff_df['patient'].unique())
    if len(patients) < 3:
        return pd.DataFrame()
    rng    = np.random.default_rng(seed)
    blocks = {p: diff_df[diff_df['patient'] == p] for p in patients}

    draws = []
    for _ in range(n_boot):
        drawn = rng.choice(patients, size=len(patients), replace=True)
        parts = []
        for k, p in enumerate(drawn):
            b = blocks[p].copy()
            b['patient'] = f'{p}__{k}'      # unique group per draw
            parts.append(b)
        samp = pd.concat(parts, ignore_index=True)
        res  = fit_fn(samp)
        if res is None:
            continue
        draws.append(res.params)

    return pd.DataFrame(draws).reset_index(drop=True) if draws else pd.DataFrame()


def ci_from_draws(draws, weights, alpha=ALPHA):
    """
    Public: make_rest_mvt_psd_figure imports this so its difference-spectrum
    ribbon is built by the same code as the intervals in this module's tables.

    Percentile CI for a contrast of the bootstrap draws, plus its achieved
    significance level

        p = 2 * min( frac(boot <= 0), frac(boot >= 0) )

    the two-sided p implied by the same resampling distribution the CI comes
    from, so a star and the interval beneath it can never disagree.  Its floor is
    1/n_boot, so p is never reported as 0.

    Returns (lo, hi, n_ok, p_boot).
    """
    if draws is None or draws.empty:
        return np.nan, np.nan, 0, np.nan
    c   = np.array([weights.get(n, 0.0) for n in draws.columns])
    arr = draws.to_numpy(dtype=float) @ c
    arr = arr[np.isfinite(arr)]
    if not arr.size:
        return np.nan, np.nan, 0, np.nan

    lo, hi  = np.percentile(arr, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p_boot  = min(1.0, 2.0 * min(float(np.mean(arr <= 0)), float(np.mean(arr >= 0))))
    p_boot  = max(p_boot, 1.0 / len(arr))       # resolution floor, never 0
    return round(float(lo), 5), round(float(hi), 5), len(arr), round(p_boot, 5)


_SE_REDUCED_FORMULA = 'diff ~ C(timepoint, Treatment("home"))'


def _fit_reduced(df):
    """Fit the within-stim-state model on one subset. None if it fails."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return smf.mixedlm(_SE_REDUCED_FORMULA, df,
                               groups=df['patient']).fit(reml=True)
    except Exception:
        return None


def _reduced_weights(res):
    """Contrast weights for the timepoint-averaged movement effect from the
    reduced (one-stim-state) model `diff ~ C(timepoint)`: the mean of the three
    per-timepoint effects (home = Intercept, M3/M6 = Intercept + beta). The
    sign-flip p (`_se_stat`) and the reported estimate/CI (`simple_effects`)
    both build the contrast from here, so they cannot drift apart."""
    tp = [n for n in res.params.index if n.startswith('C(timepoint')]
    return {'Intercept': 1.0, **{t: 1.0 / (len(tp) + 1) for t in tp}}


def _se_stat(res):
    """(estimate, |z|) for the timepoint-averaged movement effect."""
    w = _reduced_weights(res)
    names = list(res.params.index)
    c = np.array([w.get(n, 0.0) for n in names], dtype=float)
    est = float(c @ res.params.values)
    var = float(c @ res.cov_params().values @ c)
    if not np.isfinite(var) or var <= 0:
        return est, np.nan
    return est, abs(est / np.sqrt(var))


def permutation_simple_effects(diff_df, seed=PERM_SEED):
    """
    Exact permutation p for the movement effect within each stimulation state.

    Scheme.  Restrict to one stim state, fit the REDUCED model on that subset
    (`diff ~ C(timepoint)`, ~27 rows), take the timepoint-averaged contrast
    (`_reduced_weights` — the same one `simple_effects` reports the estimate and
    CI from), and sign-flip WHOLE PATIENTS exhaustively: 2^n vectors, so the p
    is exact rather than Monte Carlo.

    Why the subset and not the full model.  Under H0 "no movement effect in
    THIS stim state" the diffs are symmetric about zero within that subset, so
    whole-patient sign flips are exactly exchangeable and no nuisance parameter
    is shared with the other state.  Sign-flipping inside the full model would
    hit the classic nuisance-parameter problem — the other state's rows carry a
    real effect into the shared timepoint terms — and would need an approximate
    scheme (Freedman-Lane), forfeiting exactness, which is the whole point.

    Since fix A `simple_effects` takes the estimate and the CI from this same
    reduced model on the same rows, so the point estimate, interval and p are
    one model. (The interval is still a bootstrap percentile and the p a
    sign-flip, i.e. different *constructions* of the same model — they can
    disagree at the very margin, which is why the inferential claim for
    "does the modulation differ across states" stays with the `x stim ON`
    interaction term, not with these.)

    Why the sign-flip and not p_boot as the FDR input.  On a true null built by
    sign-flipping whole patients, the bootstrap achieved-significance level
    rejected at about twice its nominal rate while the permutation was exact.

    Returns {stim_label: (p, n_perm_used)}.
    """
    out = {}
    for stim_on, label in [(False, 'stimOFF'), (True, 'stimON')]:
        sub = diff_df[diff_df['stim'] == ('stimON' if stim_on else 'stimOFF')]
        pats = sorted(sub['patient'].unique())
        if len(sub) < 6 or len(pats) < 3:
            out[label] = (np.nan, 0)
            continue
        res0 = _fit_reduced(sub)
        if res0 is None:
            out[label] = (np.nan, 0)
            continue
        obs_z = _se_stat(res0)[1]
        if not np.isfinite(obs_z):
            out[label] = (np.nan, 0)
            continue

        pos = {p: (sub['patient'] == p).to_numpy() for p in pats}
        cnt = used = 0
        for vec in itertools.product([1.0, -1.0], repeat=len(pats)):
            perm = sub.copy()
            v = perm['diff'].to_numpy(copy=True)
            for p_i, sgn in zip(pats, vec):
                v[pos[p_i]] *= sgn
            perm['diff'] = v
            r = _fit_reduced(perm)
            if r is None:
                continue
            used += 1
            z = _se_stat(r)[1]
            if np.isfinite(z) and z >= obs_z - 1e-12:
                cnt += 1
        out[label] = ((cnt / used if used else np.nan), used)
    return out


def simple_effects(diff_df, feature, side_label, perm=None,
                   n_boot=N_BOOT, seed=PERM_SEED, do_boot=True):
    """
    Movement effect (mvt - rest) within each stimulation state — estimate, CI
    and p ALL from the REDUCED model `diff ~ C(timepoint)` fitted on that stim
    state's rows.

        estimate  timepoint-averaged contrast of the reduced fit (`_reduced_weights`)
        CI        percentile cluster bootstrap of the reduced fit on the same
                  subset (whole-patient resampling)
        p         exact whole-patient sign-flip within the subset
                  (`permutation_simple_effects`), passed in via `perm`

    One construction on one set of rows: the interval and the star are the same
    model. The previous version took estimate/CI from the FULL-model contrast
    (`Intercept + 1/3(bM3+bM6)[+bON]`, all 53 rows) while the p was this reduced
    sign-flip — two constructions that agreed to <1% but were not the same
    quantity. The cost of the switch is that the interval no longer borrows the
    other stim state's rows for the shared timepoint terms; for a state-specific
    claim that is arguably the more honest interval.

    These are ESTIMATES; the inferential claim for "does the modulation differ
    across states" stays with the FDR-corrected `x stim ON` interaction term.
    """
    rows = []
    unit = FEATURE_UNITS.get(feature, 'ln')
    for stim_on, label in [(False, 'stimOFF'), (True, 'stimON')]:
        sub = diff_df[diff_df['stim'] == ('stimON' if stim_on else 'stimOFF')]
        if len(sub) < 6 or sub['patient'].nunique() < 3:
            continue
        res = _fit_reduced(sub)
        if res is None:
            continue
        w = _reduced_weights(res)
        try:
            est, se = _contrast(res, w)
        except Exception:
            continue

        b_draws = (cluster_bootstrap_draws(sub, n_boot=n_boot, seed=seed,
                                           fit_fn=_fit_reduced)
                   if do_boot else pd.DataFrame())
        ci_lo, ci_hi, n_ok, p_boot = ci_from_draws(b_draws, w)

        p_perm = perm.get(label, (np.nan,))[0] if perm else np.nan
        n_perm = perm.get(label, (np.nan, 0))[1] if perm else 0

        rows.append({
            'feature':    feature,
            'group':      feature_group(feature),
            'unit':       unit,
            'side_label': side_label,
            'stim':       label,
            'est':        round(est, 5),
            'se':         round(se, 5),
            'ci_lo':      ci_lo,
            'ci_hi':      ci_hi,
            'ci_kind':    'cluster bootstrap (reduced)' if np.isfinite(ci_lo) else 'none',
            'p_boot':     p_boot,
            # exact sign-flip permutation — what apply_fdr_simple_effects
            # corrects.  NaN when --no-perm skipped it.
            'p_perm':     round(float(p_perm), 5) if np.isfinite(p_perm) else np.nan,
            'n_perm':     n_perm,
            'fold':       _fold(est, unit),
            'fold_lo':    _fold(ci_lo, unit),
            'fold_hi':    _fold(ci_hi, unit),
            'n_boot_ok':  n_ok,
            'n_obs':      len(sub),
            'n_patients': sub['patient'].nunique(),
        })
    return rows


def apply_fdr_simple_effects(se_df):
    """BH across features, separately within each stimulation state.

    CORRECTS THE PERMUTATION p.  `p_perm` is the exact whole-patient sign-flip
    within that stim state (permutation_simple_effects); `p_boot` is used only
    when the permutation was skipped (--no-perm), and `p_used` records which one
    carried the correction so a table can never be read as exact when it was not.

    p_boot is not the right input.  On a true null it rejects at roughly twice
    its nominal rate at this cluster count, and across the 78 simple effects it
    was the more liberal of the two in 70; BH on p_boot returned 19 significant
    cells where BH on the exact p returns 6.  The full numbers are in
    permutation_simple_effects' docstring.

    Two families of 13 rather than one of 26. The stimOFF and stimON effects of
    one feature are linear contrasts of the same fitted vector — they share the
    intercept and both timepoint terms and differ only by the stim coefficient —
    so pooling them would build a family out of 13 near-duplicate pairs.
    Splitting by stimulation state gives each feature one test per family.

    This is a family the term-level correction does not cover: `movement
    (mvt-rest)` tests the movement effect at the reference cell (home, stimOFF)
    and `x stim ON` tests whether the two states differ, but neither asks
    whether a movement effect exists within a given state. Being a separate
    family, its FDR is controlled on its own; a feature significant here and in
    the interaction family is the same fitted coefficients read two ways, not
    two independent confirmations.

    BH assumes independence or positive dependence. Across these features the
    Diff values are not uniformly positively correlated, so PRDS is not clearly met and BH is used as
    the field convention rather than because its assumption is verified.
    """
    if se_df.empty:
        return se_df

    df = se_df.copy()
    if 'p_perm' not in df.columns:
        df['p_perm'] = np.nan
    if 'p_boot' not in df.columns:
        df['p_boot'] = np.nan

    df['p_used'] = df['p_perm'].where(df['p_perm'].notna(), df['p_boot'])
    df['p_kind'] = np.where(df['p_perm'].notna(), 'permutation', 'bootstrap')
    df['q_fdr'] = np.nan
    df['sig_q'] = False
    for _stim, grp in df.groupby('stim', sort=False):
        p  = grp['p_used'].to_numpy(dtype=float)
        ok = ~np.isnan(p)
        if ok.sum() == 0:
            continue
        idx = grp.index[ok]
        df.loc[idx, 'q_fdr'] = np.round(
            multipletests(p[ok], method='fdr_bh')[1], 5)
        df.loc[idx, 'sig_q'] = df.loc[idx, 'q_fdr'] < ALPHA
    return df


def apply_fdr(df):
    """BH across features, within each term. p_perm when available, else p_boot."""
    df = df.copy()
    # Fallback is the BOOTSTRAP p, not a Wald p — this module reports no Wald
    # quantity at all.  p_boot is liberal (about 2x nominal type-I error at
    # n = 9 clusters), so a
    # --no-perm pass is a fast preview and not a result.
    df['p_used']  = df['p_perm'].where(df['p_perm'].notna(), df['p_boot'])
    df['p_fdr']   = np.nan
    df['sig']     = df['p_used'] < ALPHA
    df['sig_fdr'] = False
    for term, grp in df.groupby('term', sort=False):
        p = grp['p_used'].to_numpy(dtype=float)
        ok = ~np.isnan(p)
        if ok.sum() > 1:
            _, corr, _, _ = multipletests(p[ok], method='fdr_bh')
            idx = grp.index[ok]
            df.loc[idx, 'p_fdr']   = np.round(corr, 5)
            df.loc[idx, 'sig_fdr'] = corr < ALPHA
        elif ok.sum() == 1:
            idx = grp.index[ok]
            df.loc[idx, 'p_fdr']   = np.round(p[ok], 5)
            df.loc[idx, 'sig_fdr'] = p[ok] < ALPHA
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def _run_feature(feature, ddf, side_label, do_perm, n_perm, seed,
                 do_boot, n_boot):
    """One feature: bootstrap draws, coefficient rows, simple effects.

    Module-level and self-contained so it can be shipped to a worker process.
    `cluster_bootstrap_draws` and `permutation_pvalues` each build their own
    `default_rng(seed)` from the fixed seed on entry, so a feature's numbers
    depend only on its own data — never on execution order.  That is what makes
    `n_jobs > 1` bit-identical to the serial run rather than merely equivalent.

    The full-model draws stay inside the worker: they feed the coefficient
    table's CIs, are 2000 rows wide and nothing downstream needs them, so
    returning them would only pay to pickle them back. The simple effects build
    their own REDUCED-model draws per stim state (fix A).
    """
    draws = (cluster_bootstrap_draws(ddf, n_boot=n_boot, seed=seed)
             if do_boot else pd.DataFrame())
    rows = analyse_feature(ddf, feature, side_label, do_perm=do_perm,
                           n_perm=n_perm, seed=seed, draws=draws)
    # Exhaustive 2^n sign-flip per stim state; skipped with --no-perm, in which
    # case apply_fdr_simple_effects falls back to p_boot and says so.
    se_perm = permutation_simple_effects(ddf, seed=seed) if do_perm else None
    se_rows = simple_effects(ddf, feature, side_label, perm=se_perm,
                             n_boot=n_boot, seed=seed, do_boot=do_boot)
    return feature, rows, se_rows, len(ddf), len(draws)


def run_side(all_sessions, side, side_label, do_perm=True,
             n_perm=N_PERM, seed=PERM_SEED, do_boot=True, n_boot=N_BOOT,
             n_jobs=1):
    diffs = build_all_diffs(all_sessions, side)

    tasks = [(feature, ddf) for feature, ddf in diffs.items()
             if ddf is not None and not ddf.empty]

    rows, long_parts, se_rows = [], [], []
    for feature, ddf in tasks:
        long_parts.append(ddf.assign(feature=feature, side_label=side_label))

    args = (side_label, do_perm, n_perm, seed, do_boot, n_boot)

    def _collect(results):
        # Consumed INSIDE the backend context below: return_as='generator'
        # dispatches lazily, so draining it after the context exited would run
        # the features under whatever backend config happened to be current.
        for feature, f_rows, f_se_rows, n_obs, n_draws in results:
            rows.extend(f_rows)
            se_rows.extend(f_se_rows)
            print(f'    {feature:<22s} n_obs={n_obs:3d}  '
                  f'boot={n_draws}', flush=True)

    if n_jobs == 1:
        # Serial path kept explicit: identical numbers, simpler tracebacks, no
        # worker start-up cost on a --no-perm --no-boot preview.
        _collect(_run_feature(ft, dd, *args) for ft, dd in tasks)
    else:
        # inner_max_num_threads=1: these are ~60-row models, so BLAS threading
        # buys nothing and one thread pool per worker only oversubscribes.
        # return_as='generator' keeps submission order, so the progress log is
        # byte-identical to the serial run.
        with parallel_backend('loky', inner_max_num_threads=1):
            _collect(Parallel(n_jobs=n_jobs, return_as='generator')(
                delayed(_run_feature)(ft, dd, *args) for ft, dd in tasks))

    if not rows:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df = apply_fdr(pd.DataFrame(rows))
    col_order = ['feature', 'group', 'unit', 'side_label', 'term', 'coef', 'se', 'z_stat',
                 'ci_lo', 'ci_hi', 'ci_kind',
                 'exp_coef',
                 'p_perm', 'p_boot', 'p_used', 'p_fdr', 'sig', 'sig_fdr',
                 'perm_scheme', 'n_perm', 'n_boot_ok', 'coef_ols',
                 'n_obs', 'n_patients', 'group_var', 'scale', 'icc']
    df = df[[c for c in col_order if c in df.columns]]

    long_df = (pd.concat(long_parts, ignore_index=True)
               if long_parts else pd.DataFrame())
    se_df = apply_fdr_simple_effects(pd.DataFrame(se_rows)) if se_rows \
        else pd.DataFrame()
    return df, long_df, se_df


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--sides', nargs='*', default=None,
                    help='side labels to run (default: all of SIDES — L, R, bilateral)')
    ap.add_argument('--n-perm', type=int, default=N_PERM,
                    help=f'Monte-Carlo draws per label-permutation scheme (default {N_PERM})')
    ap.add_argument('--no-perm', action='store_true',
                    help='skip permutation; p_boot carries the FDR instead (fast preview, '
                         'not a result — p_boot runs at ~2x nominal type-I error here)')
    ap.add_argument('--no-boot', action='store_true',
                    help='skip the cluster bootstrap; ci_lo/ci_hi and p_boot come back '
                         'empty (there is no Wald fallback) (fast)')
    ap.add_argument('--n-boot', type=int, default=N_BOOT,
                    help=f'cluster-bootstrap resamples per feature (default {N_BOOT})')
    ap.add_argument('--seed', type=int, default=PERM_SEED)
    ap.add_argument('--jobs', type=int, default=1,
                    help='worker processes over features (13 per side, each '
                         'independent and self-seeded, so this is wall-clock '
                         'only — the CSV is bit-identical).  -1 uses every core '
                         '(default 1)')
    args = ap.parse_args(argv)

    wanted = SIDES if not args.sides else [s for s in SIDES if s[1] in args.sides]
    if not wanted:
        print(f'No side matched {args.sides}. Valid: {[s[1] for s in SIDES]}')
        return

    all_sessions = load_all_sessions()

    for side, side_label, pretty in wanted:
        print(f'\n{"-" * 66}\n{pretty}  (side_label={side_label})\n{"-" * 66}', flush=True)
        df, long_df, se_df = run_side(all_sessions, side, side_label,
                                      do_perm=not args.no_perm,
                                      n_perm=args.n_perm, seed=args.seed,
                                      do_boot=not args.no_boot,
                                      n_boot=args.n_boot,
                                      n_jobs=args.jobs)
        if df.empty:
            print('  No data — skipped.')
            continue
        path = os.path.join(RESULTS_DIR, f'lme_rest_mvt_diff_{side_label}.csv')
        df.to_csv(path, index=False)
        long_path = os.path.join(RESULTS_DIR, f'diff_long_{side_label}.csv')
        long_df.to_csv(long_path, index=False)
        n_sig = int(df['sig'].sum())
        n_fdr = int(df['sig_fdr'].sum())
        print(f'  Saved -> {path}  ({len(df)} rows, {n_sig} p<{ALPHA}, {n_fdr} FDR-sig)',
              flush=True)
        print(f'  Saved -> {long_path}  ({len(long_df)} Diff values)', flush=True)
        if not se_df.empty:
            se_path = os.path.join(RESULTS_DIR, f'simple_effects_{side_label}.csv')
            se_df.to_csv(se_path, index=False)
            print(f'  Saved -> {se_path}  ({len(se_df)} simple effects)', flush=True)


if __name__ == '__main__':
    main()
