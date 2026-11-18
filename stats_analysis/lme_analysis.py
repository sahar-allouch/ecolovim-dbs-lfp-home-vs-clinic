"""
lme_analysis.py
===============
Test whether M3, home, and M6 spectral features differ using Linear
Mixed-Effects Models (LME) fitted separately per condition.

Scientific question
-------------------
Are recordings acquired at home (Percept FFT streaming) equivalent to
in-clinic recordings at 3 months (M3) and 6 months (M6) post-surgery?
Does the feature level change across the three sessions within each
experimental condition?

Model (per condition)
---------------------
    feature ~ C(timepoint, Treatment("home")) + (1 | patient)

One model is fitted per condition × feature combination:
    Conditions : rest_stimOFF, mvt_stimOFF, rest_stimON, mvt_stimON
    Reference  : home (at-home recording)
    Contrasts  : M3 vs home  |  M6 vs home  (model coefficients)
                 M3 vs M6                   (post-hoc linear contrast:
                                             M3_coef − M6_coef)

Centering on home directly answers the substitution question:
"Does the at-home recording differ from each clinic visit?"
The post-hoc M3 vs M6 contrast additionally characterises longitudinal
change between the two in-clinic sessions.

The random intercept (1 | patient) accounts for between-patient
baseline differences, leaving only within-patient session changes as
the fixed effect of interest.

Rationale for per-condition approach
-------------------------------------
The primary question is whether a given recording session (home) can
substitute for a clinic visit for a given experimental context.  Pooling
conditions in a factorial model would introduce movement × stim nuisance
structure that is secondary to this equivalence question and would dilute
the focused per-condition contrasts.  ICC addresses the agreement question
directly; LME provides the hypothesis test of mean differences.

Note on power
-------------
Each per-condition model has N ≈ 27–30 observations (≤10 patients ×
3 timepoints).  This is sufficient to detect large effects but may miss
moderate ones.  Non-significant results should be interpreted alongside
the ICC analysis rather than as evidence of equivalence: a non-significant
p-value cannot distinguish evidence of absence from absence of evidence,
and this project runs no test that can.

Features analysed
-----------------
    - Absolute band power — delta, theta, alpha, beta (ANALYSIS_BANDS in
      config.py; sub-bands and gamma are excluded there, see EXCLUDED_BANDS)
    - Flattened (FOOOF-residual) absolute band power — beta only, see
      INCLUDE_FLAT / FLAT_ANALYSIS_BANDS below
    - Aperiodic exponent and offset (FOOOF), broadband power

Relative power is computed by the pipeline but not modelled here.

Usage
-----
    python -m stats_analysis.lme_analysis

Results saved to Results/stats/lme/.
"""

import os
import io
import sys
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.stats.multitest import multipletests
from joblib import Parallel, delayed, parallel_backend

from stats_analysis.data_utils import (
    load_all_sessions, load_all_sessions_tremor,
    build_bandpower_df, build_aperiodic_df,
    build_broadband_power_df,
    build_flattened_bandpower_df,
    build_tremor_bandpower_df,
    build_peak_params_df,
    CONDITIONS, EXCLUDED_BANDS, PEAK_BAND, PEAK_METRICS,
)
from stats_analysis.plot_style import (
    FONT, SIZE_HEAD, SIZE_TICK, SIZE_ANNOT, _INK, COND_LABELS, apply_style, save_fig as _save_fig,
)

# ── Configuration ──────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join('Results', 'stats', 'lme')
os.makedirs(RESULTS_DIR, exist_ok=True)

ALPHA = 0.05
_CLIP = 1e-12  # guard against log(0) in log-transforms
_RE_SLOPES = '~C(timepoint, Treatment("home"))'  # random slopes formula

# |d| below this: sign flips are noise, not fragility.  0.2 is Cohen's
# negligible boundary and `cohen_d_lme` now standardises by total SD, so the
# cutoff is on the scale those benchmarks were calibrated for.  It used to be
# applied to β/σ_resid, i.e. values inflated by √2·(σ_total/σ_resid), which
# made the rule fire on effects it was meant to excuse as noise.
FRAGILE_D_THRESH = 0.2
_D_CSV_DECIMALS  = 6    # cohen_d precision in CSVs; flags are computed post-rounding

# Flattened (FOOOF-residual) power, same policy as lme_rest_mvt_diff: fitted,
# but for beta only.  Beta is the band where a peak is actually detected; in the
# other bands the aperiodic-removed residual is negative for a large share of
# observations, so the
# band sum is dominated by residual noise around zero rather than by periodic
# power, and each such feature would still occupy a slot in every BH family
# apply_fdr_lme builds.  Set INCLUDE_FLAT False to drop the family entirely.
INCLUDE_FLAT = True     # beta only — see FLAT_ANALYSIS_BANDS
FLAT_ANALYSIS_BANDS = ['beta']


# ── Helpers ────────────────────────────────────────────────────────────────────
def _analysis_bands(df):
    """Bands of `df` that enter the models, in the order they appear.

    EXCLUDED_BANDS (config.py) is applied here rather than at feature
    extraction: the excluded bands stay available to plotting, they are just
    not modelled.  This matters for multiplicity, not only for row count — the
    sub-bands partition their parent, so fitting both put two near-duplicate
    tests of one signal into each BH family `apply_fdr_lme` builds and made
    the parent band pay for its own partition.  Same band set as
    lme_rest_mvt_diff, so the modules' families match.
    """
    return [b for b in df['band'].unique() if b not in EXCLUDED_BANDS]


def _flat_analysis_bands(df):
    """Flattened-power bands of `df` that enter the models — beta only.

    Narrower than _analysis_bands by design; see INCLUDE_FLAT.  Ordered by
    FLAT_ANALYSIS_BANDS rather than by appearance so the feature order does not
    depend on how the DataFrame was built.
    """
    if not INCLUDE_FLAT:
        return []
    present = set(df['band'].unique())
    return [b for b in FLAT_ANALYSIS_BANDS
            if b in present and b not in EXCLUDED_BANDS]




def _n_groups(res):
    """Number of patients (random-effect groups) behind a fit, or NaN.

    statsmodels exposes this as n_groups on the fitted result; falling back to
    the group labels keeps the column populated if that attribute moves."""
    n = getattr(res, 'n_groups', None)
    if n is None:
        labels = getattr(getattr(res, 'model', None), 'group_labels', None)
        n = len(labels) if labels is not None else None
    return int(n) if n is not None else np.nan


def _round_d(d):
    """Cohen's d as written to CSV. Flags must be computed from this, not the
    raw float, so a consumer recomputing off the CSV reaches the same verdict."""
    try:
        v = float(d)
    except (TypeError, ValueError):
        return np.nan
    return np.nan if np.isnan(v) else round(v, _D_CSV_DECIMALS)


def is_fragile(sign_stable_n, n_loo, cohen_d, thresh=FRAGILE_D_THRESH):
    """Canonical 'fragile?' rule — use this everywhere (app + CSV writers).

    fragile = (loo_sign_stable < loo_n) and (|d| >= threshold)

    An estimate is fragile when at least one leave-one-patient-out refit flipped
    the sign of beta relative to the full model, AND the effect is big enough
    that the flip is not just noise around zero.

    Missing/unparseable LOO counts → not fragile (nothing to judge on).
    """
    try:
        sn = int(sign_stable_n)
        nl = int(n_loo)
    except (TypeError, ValueError):
        return False
    if pd.isna(sign_stable_n) or pd.isna(n_loo):
        return False
    try:
        d_abs = abs(float(cohen_d))
    except (TypeError, ValueError):
        d_abs = 0.0
    if np.isnan(d_abs):
        d_abs = 0.0
    t = FRAGILE_D_THRESH if thresh is None else float(thresh)
    return (sn < nl) and (d_abs >= t)


def _fit_mixedlm(formula, df, group_var='patient', vc_formula=None,
                 re_formula=None, strict=False, quiet=False):
    """
    Fit a MixedLM model and return the result, suppressing noisy output
    when quiet=True.  If strict=True, ConvergenceWarning is raised as an
    exception rather than suppressed (used when attempting random slopes).

    Attaches ._singular_re = True to the result if statsmodels emits
    'Random effects covariance is singular' during fitting.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model = smf.mixedlm(formula, df, groups=df[group_var],
                            vc_formula=vc_formula, re_formula=re_formula)
        if quiet:
            buf = io.StringIO()
            old = sys.stdout
            sys.stdout = buf
            try:
                result = model.fit(reml=True)
            finally:
                sys.stdout = old
        else:
            result = model.fit(reml=True)

    singular_re = False
    for w in caught:
        if strict and issubclass(w.category, ConvergenceWarning):
            raise ConvergenceWarning(str(w.message))
        if issubclass(w.category, UserWarning) and 'singular' in str(w.message).lower():
            singular_re = True

    result._singular_re = singular_re
    return result


def cohen_d_lme(beta, result):
    """
    Cohen's d for LME: standardised effect size using *total* SD.

    d = β / σ_total,  where σ_total = sqrt(τ² + σ²)
        τ² = result.cov_re.iloc[0, 0]  (between-patient, random intercept)
        σ² = result.scale              (within-patient residual)

    Total SD, not residual SD.  The denominator then spans the same variance
    decomposition as the reported ICCs (ICC = τ²/(τ² + σ²)), so d and the
    reliability figures can be read against each other, and Cohen's 1988
    benchmarks — calibrated on a standardiser that *includes* between-subject
    variance — actually apply.

    Standardising by σ_resid alone (the previous definition here) gives
    β/σ_resid = √2 · d_z, which is not d_z despite the name, carries no
    information beyond the p-value (Spearman |d| vs -log10 p = 0.99 across
    this pipeline's fits, because β/σ_resid = t/√n at fixed n), and cannot be
    compared with Cohen's thresholds.  σ_total fixes all three.

    Note this is *not* the quantity to feed a repeated-measures power
    calculation — that wants d_z = β/(σ_resid·√2).

    Where the random intercept is singular (τ² ≈ 0, tracked as
    `full_singular_re`) σ_total collapses onto σ_resid and d is not
    identified separately; read those rows with that flag.

    Thresholds (Cohen 1988):  |d| < 0.2 negligible  ·  0.2–0.5 small
                               0.5–0.8 medium          · ≥ 0.8 large
    """
    try:
        sigma_sq = float(result.scale)
        tau_sq   = _random_effect_var(result)
        if not np.isfinite(tau_sq) or tau_sq < 0:
            tau_sq = 0.0
        sigma = np.sqrt(tau_sq + sigma_sq)
        if sigma == 0 or np.isnan(sigma):
            return np.nan
        return float(beta) / sigma
    except Exception:
        return np.nan


def _random_effect_var(result):
    """
    Total random-effect variance of a fitted MixedLM: cov_re diagonal + vcomp.

    Both terms are read.  A random intercept lands in `cov_re`; a model that
    carries the patient effect as a variance component (`vc_formula`) reports it
    in `result.vcomp` and leaves `cov_re` empty.  Reading only `cov_re` would
    give tau2 = 0 for such a model and standardise d by the residual SD.

    Returns 0.0 when neither is present, so a model genuinely without random
    effects degrades to beta/sigma_resid rather than raising.
    """
    total = 0.0
    try:
        cov_re = getattr(result, 'cov_re', None)
        if cov_re is not None and getattr(cov_re, 'size', 0):
            total += float(np.sum(np.diag(np.asarray(cov_re, dtype=float))))
    except Exception:
        pass
    try:
        vcomp = getattr(result, 'vcomp', None)
        if vcomp is not None and len(np.atleast_1d(vcomp)):
            total += float(np.nansum(np.asarray(vcomp, dtype=float)))
    except Exception:
        pass
    return total if np.isfinite(total) else 0.0


def _variance_components(result):
    """
    τ², σ² and the model ICC for a fitted MixedLM — the denominator of
    `cohen_d_lme` laid open so a reader can recompute d from the CSV.

    `icc` here is the model's own τ²/(τ² + σ²) over all three timepoints under
    compound symmetry.  It is NOT the pairwise ICC(2,1)/ICC(3,1) from
    icc_analysis.py, which is computed per session pair on complete cases only;
    the two will not agree exactly and should not be quoted for each other.
    """
    try:
        sigma_sq = float(result.scale)
    except Exception:
        return {'group_var': np.nan, 'scale': np.nan, 'model_icc': np.nan}
    tau_sq = _random_effect_var(result)   # cov_re diagonal + vcomp; see helper
    tot = (tau_sq + sigma_sq) if np.isfinite(tau_sq) and np.isfinite(sigma_sq) else np.nan
    return {
        'group_var': round(tau_sq,   6) if np.isfinite(tau_sq)   else np.nan,
        'scale':     round(sigma_sq, 6) if np.isfinite(sigma_sq) else np.nan,
        'model_icc': round(tau_sq / tot, 4) if np.isfinite(tot) and tot > 0 else np.nan,
    }


def _d_label(d):
    """One-word magnitude label for Cohen's d."""
    if np.isnan(d):
        return ''
    a = abs(d)
    if a < 0.2:
        return 'neg'
    if a < 0.5:
        return 'sm'
    if a < 0.8:
        return 'med'
    return 'lg'


def _m3_vs_m6(result):
    """
    Post-hoc contrast: M3 − M6 = (M3 vs home) − (M6 vs home).
    Computed from the fixed-effect estimates of an already-fitted model
    with home as reference — no re-fitting required.

    Direction follows the contrast's own name, matching the other two rows:
    'M3 vs home' is M3 − home and 'M6 vs home' is M6 − home, so 'M3 vs M6'
    is M3 − M6.  It used to be M6 − M3, which broke that convention and had
    to be undone by a hand-written "M6−M3" header in every consumer.

    Note the sign is therefore *backwards in time*: a positive coefficient
    means M3 exceeded M6, i.e. the feature DECREASED over the follow-up.

    Returns (coef, lower_ci, upper_ci, pvalue), or (nan, nan, nan, nan)
    if either coefficient is absent.
    """
    k_m3 = 'C(timepoint, Treatment("home"))[T.M3]'
    k_m6 = 'C(timepoint, Treatment("home"))[T.M6]'
    params = list(result.fe_params.index)
    if k_m3 not in params or k_m6 not in params:
        return np.nan, np.nan, np.nan, np.nan

    contrast = np.zeros(len(params))
    contrast[params.index(k_m3)] =  1.0
    contrast[params.index(k_m6)] = -1.0

    t_res = result.t_test(contrast.reshape(1, -1))
    coef  = float(t_res.effect[0])
    lo    = float(t_res.conf_int()[0, 0])
    hi    = float(t_res.conf_int()[0, 1])
    p     = float(t_res.pvalue[0] if np.ndim(t_res.pvalue) > 0 else t_res.pvalue)
    return coef, lo, hi, p


# ══════════════════════════════════════════════════════════════════════════════
# Resampling inference: permutation p-values and cluster-bootstrap intervals
# ══════════════════════════════════════════════════════════════════════════════
#
# This module reports NO Wald quantity.  statsmodels MixedLM sets use_t = False,
# so `result.pvalues` compares beta/se against N(0, 1) — it treats the standard
# error as a known constant and extrapolates a normal tail indefinitely.  With
# 9-10 patients neither is defensible for a reported number: against an exact
# sign-flip permutation, a Wald p was well calibrated near alpha but
# extrapolated to tail probabilities orders of magnitude below the exact p.
#
# What replaces it, matching lme_rest_mvt_diff so the two modules' tables can be
# read row for row:
#
#   p        permutation p — timepoint labels shuffled WITHIN patient, which is
#            the null "this patient's three sessions are exchangeable".  The
#            within-patient restriction is what keeps the between-patient
#            variation (the random intercept) intact under the null.
#            6 orderings per patient, 6^10 total, so this is Monte Carlo rather
#            than exhaustive; p floors at 1/N_PERM_LME.
#
#   ci_lo /  cluster bootstrap — whole patients resampled with replacement,
#   ci_hi    percentile interval.  Duplicated patients are relabelled so
#            statsmodels treats each copy as its own group instead of merging
#            them into one artificially large cluster.
#
#   p_boot   the two-sided p implied by those same draws, so a CI and the p
#            beside it can never disagree.  Reported for audit only: on a true
#            null it runs at roughly twice its nominal rate at this cluster
#            count (Results/stats/lme_rest_mvt/null_calibration.csv), so it is
#            a preview, and `p` (the permutation) is the result.
#
# All three timepoint contrasts are linear contrasts of one fixed-effect vector,
# so ONE bootstrap per (feature, condition) serves all of them — the same reason
# lme_rest_mvt_diff.cluster_bootstrap_draws returns draws rather than intervals.

N_PERM_LME    = 5000   # Monte-Carlo label permutations per (feature, condition)
N_BOOT_LME    = 2000   # cluster-bootstrap resamples, shared by all 3 contrasts
RESAMPLE_SEED = 0      # fixed so reruns reproduce identical CSVs

_K_M3_ = 'C(timepoint, Treatment("home"))[T.M3]'
_K_M6_ = 'C(timepoint, Treatment("home"))[T.M6]'

# contrast label -> weights on the fixed-effect vector
CONTRAST_WEIGHTS = {
    'M3 vs home': {_K_M3_:  1.0},
    'M6 vs home': {_K_M6_:  1.0},
    'M3 vs M6':   {_K_M3_:  1.0, _K_M6_: -1.0},
}


def _contrast_est_z(result, weights):
    """(estimate, |z|) for a linear contrast c'beta, SE = sqrt(c'Vc).

    The covariance term is why this cannot be done from marginal SEs: with home
    as the reference cell, beta_M3 and beta_M6 share that baseline and are
    positively correlated, so Var(M3 - M6) = se3^2 + se6^2 - 2*Cov.  Dropping
    the covariance inflates the SE by sqrt(2) on a balanced design.
    """
    try:
        names = list(result.fe_params.index)
        c = np.array([weights.get(n, 0.0) for n in names], dtype=float)
        if not c.any():
            return np.nan, np.nan
        V = result.cov_params().values[:len(names), :len(names)]
        est = float(c @ result.fe_params.values)
        var = float(c @ V @ c)
        if not np.isfinite(var) or var <= 0:
            return est, np.nan
        return est, abs(est / np.sqrt(var))
    except Exception:
        return np.nan, np.nan


def _permute_timepoint(df, rng, strata):
    """Shuffle `timepoint` within each stratum, leaving everything else fixed."""
    out = df.copy()
    vals = out['timepoint'].to_numpy(copy=True)
    for _, idx in out.groupby(strata, observed=True, sort=False).indices.items():
        if len(idx) > 1:
            vals[idx] = rng.permutation(vals[idx])
    out['timepoint'] = vals
    return out


def permutation_p_lme(df, formula, group_var, vc_formula, observed_z,
                      n_perm=N_PERM_LME, seed=RESAMPLE_SEED, strata=None):
    """
    Permutation p per contrast:  p = (#{|z_perm| >= |z_obs|} + 1) / (n_used + 1)

    The +1 on both sides counts the observed data as one of its own
    permutations, which keeps the test valid and stops p from ever being 0.
    Returns ({label: p}, n_used).
    """
    if strata is None:
        strata = ['patient']
    rng = np.random.default_rng(seed)
    counts = {lbl: 0 for lbl in CONTRAST_WEIGHTS}
    used = 0
    for _ in range(n_perm):
        perm = _permute_timepoint(df, rng, strata)
        try:
            res = _fit_mixedlm(formula, perm, group_var=group_var,
                               vc_formula=vc_formula, quiet=True)
        except Exception:
            continue
        used += 1
        for lbl, w in CONTRAST_WEIGHTS.items():
            _, z = _contrast_est_z(res, w)
            obs = observed_z.get(lbl, np.nan)
            if np.isfinite(z) and np.isfinite(obs) and z >= obs - 1e-12:
                counts[lbl] += 1
    if used == 0:
        return {lbl: np.nan for lbl in CONTRAST_WEIGHTS}, 0
    return {lbl: (counts[lbl] + 1) / (used + 1) for lbl in CONTRAST_WEIGHTS}, used


def cluster_bootstrap_lme(df, formula, group_var, vc_formula,
                          n_boot=N_BOOT_LME, seed=RESAMPLE_SEED,
                          cluster_col='patient'):
    """
    Refit on patient-resampled data; return the fixed effects per draw.

    Whole clusters are resampled, not rows, so the within-patient dependence the
    random intercept models is preserved.  Duplicated patients are relabelled
    (`p__0`, `p__1`, ...) so statsmodels does not merge the copies.
    """
    clusters = sorted(df[cluster_col].unique())
    if len(clusters) < 3:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    blocks = {c: df[df[cluster_col] == c] for c in clusters}

    draws = []
    for _ in range(n_boot):
        drawn = rng.choice(clusters, size=len(clusters), replace=True)
        parts = []
        for k, c in enumerate(drawn):
            b = blocks[c].copy()
            b[cluster_col] = f'{c}__{k}'
            if 'patient_side' in b.columns:
                b['patient_side'] = b['patient_side'].astype(str) + f'__{k}'
            parts.append(b)
        samp = pd.concat(parts, ignore_index=True)
        try:
            res = _fit_mixedlm(formula, samp, group_var=group_var,
                               vc_formula=vc_formula, quiet=True)
        except Exception:
            continue
        draws.append(res.fe_params)
    return pd.DataFrame(draws).reset_index(drop=True) if draws else pd.DataFrame()


def ci_from_draws_lme(draws, weights, alpha=ALPHA):
    """
    Percentile CI for a contrast of the bootstrap draws, plus the two-sided p
    implied by the same resampling distribution:

        p_boot = 2 * min( frac(draw <= 0), frac(draw >= 0) ),  floored at 1/n

    Returns (lo, hi, n_ok, p_boot).
    """
    if draws is None or draws.empty:
        return np.nan, np.nan, 0, np.nan
    c = np.array([weights.get(n, 0.0) for n in draws.columns], dtype=float)
    if not c.any():
        return np.nan, np.nan, 0, np.nan
    arr = draws.to_numpy(dtype=float) @ c
    arr = arr[np.isfinite(arr)]
    if not arr.size:
        return np.nan, np.nan, 0, np.nan
    lo, hi = np.percentile(arr, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p = min(1.0, 2.0 * min(float(np.mean(arr <= 0)), float(np.mean(arr >= 0))))
    p = max(p, 1.0 / len(arr))
    return float(lo), float(hi), len(arr), float(p)


def _resample_cell(feat, condition, sub, formula, obs_beta, obs_z,
                   group_var, vc_formula, n_perm, n_boot, seed,
                   strata, cluster_col):
    """One (feature, condition) cell: permutation p + cluster bootstrap.

    Module-level and self-contained so it can be shipped to a worker process.
    Both resamplers build their own `default_rng(seed)` from the fixed seed on
    entry, so a cell's numbers depend only on its own data — never on execution
    order.  That is what makes `n_jobs > 1` bit-identical to the serial run
    rather than merely equivalent.
    """
    if n_perm:
        p_perm, n_used = permutation_p_lme(
            sub, formula, group_var, vc_formula, obs_z,
            n_perm=n_perm, seed=seed, strata=strata)
    else:
        p_perm, n_used = {lbl: np.nan for lbl in CONTRAST_WEIGHTS}, 0

    draws = (cluster_bootstrap_lme(sub, formula, group_var, vc_formula,
                                   n_boot=n_boot, seed=seed,
                                   cluster_col=cluster_col)
             if n_boot else pd.DataFrame())

    cell = {}
    for lbl, w in CONTRAST_WEIGHTS.items():
        lo, hi, n_ok, p_b = ci_from_draws_lme(draws, w)
        cell[lbl] = {
            'beta':      obs_beta[lbl],
            'ci_lo':     lo,
            'ci_hi':     hi,
            'p':         p_perm.get(lbl, np.nan),
            'p_boot':    p_b,
            'n_perm':    n_used,
            'n_boot_ok': n_ok,
        }
    return feat, condition, cell, n_used, len(draws)


def resample_lme(results_nested, dfs, outcome_map, conditions=None,
                 extra_fixed=None, vc_formula=None, group_var='patient',
                 strata=None, cluster_col='patient',
                 n_perm=N_PERM_LME, n_boot=N_BOOT_LME, seed=RESAMPLE_SEED,
                 n_jobs=1, quiet=False):
    """
    Permutation p and cluster-bootstrap interval for every
    (feature, condition, contrast).

    Returns {feature: {condition: {contrast: {...}}}} with keys
        beta, ci_lo, ci_hi, p, p_boot, n_perm, n_boot_ok
    where `p` is the permutation p — the reported one.

    This is the expensive step: (n_perm + n_boot) model fits per
    (feature, condition).  Pass n_perm=0 / n_boot=0 for a fast preview; the
    corresponding columns then come back NaN rather than falling back to a Wald
    quantity, because this module reports none.

    `n_jobs` spreads the cells over processes (13 features x 4 conditions = 52
    independent cells per analysis unit).  Cells share nothing and each reseeds
    from the fixed seed, so this changes wall-clock only — the CSV is
    bit-identical.  A single fit is ~11 ms on a 30-row model, i.e. Python and
    scipy.optimize overhead rather than linear algebra, so there is no BLAS
    parallelism to win and process-level scaling is near-linear.  Default 1;
    set ECOLOVIM_LME_JOBS to raise it for the __main__ run.
    """
    if conditions is None:
        conditions = CONDITIONS
    fixed_part = f'{extra_fixed} + ' if extra_fixed else ''
    out = {}
    tasks = []

    for feat, cond_results in results_nested.items():
        outcome = outcome_map.get(feat)
        df_feat = dfs.get(feat)
        if outcome is None or df_feat is None:
            continue
        formula = f'{outcome} ~ {fixed_part}C(timepoint, Treatment("home"))'

        for condition in conditions:
            res = cond_results.get(condition)
            if res is None:
                continue
            sub = df_feat[df_feat['condition'] == condition].dropna(
                subset=[outcome]).copy()
            if len(sub) < 6:
                continue

            obs = {lbl: _contrast_est_z(res, w)
                   for lbl, w in CONTRAST_WEIGHTS.items()}
            obs_z = {lbl: v[1] for lbl, v in obs.items()}

            tasks.append((feat, condition, sub, formula,
                          {lbl: v[0] for lbl, v in obs.items()}, obs_z))

    args = (group_var, vc_formula, n_perm, n_boot, seed, strata, cluster_col)

    def _collect(results):
        # Consumed INSIDE the backend context below: return_as='generator'
        # dispatches lazily, so draining it after the context exited would run
        # the cells under whatever backend config happened to be current.
        for feat, condition, cell, n_used, n_draws in results:
            out.setdefault(feat, {})[condition] = cell
            if not quiet:
                print(f'  resample {feat} | {condition}: '
                      f'perm={n_used} boot={n_draws}', flush=True)

    if n_jobs == 1:
        # Serial path kept explicit: identical numbers, simpler tracebacks, no
        # worker start-up cost on a preview run.
        _collect(_resample_cell(f_, c_, s_, fm_, b_, z_, *args)
                 for f_, c_, s_, fm_, b_, z_ in tasks)
    else:
        # inner_max_num_threads=1: the models are ~30x4, so BLAS threading buys
        # nothing and one thread pool per worker only oversubscribes.
        # return_as='generator' keeps submission order, so the progress log is
        # byte-identical to the serial run.
        with parallel_backend('loky', inner_max_num_threads=1):
            _collect(Parallel(n_jobs=n_jobs, return_as='generator')(
                delayed(_resample_cell)(f_, c_, s_, fm_, b_, z_, *args)
                for f_, c_, s_, fm_, b_, z_ in tasks))
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Per-condition LME
# ══════════════════════════════════════════════════════════════════════════════

def run_lme_per_condition(df, outcome, group_var='patient',
                          conditions=None, save_name_prefix=None,
                          extra_fixed=None, vc_formula=None,
                          try_random_slopes=False, quiet=False):
    """
    Fit one LME per condition:
        outcome ~ [extra_fixed +] C(timepoint, Treatment("home")) + (1 | patient)

    If try_random_slopes=True, first attempts random slopes model
        re_formula='~C(timepoint, Treatment("home"))'
    and falls back to random intercept only on convergence failure.

    Home is the reference; coefficients give the mean difference of M3 and
    M6 relative to home within each condition, controlling for random patient
    baseline.  A post-hoc M3 vs M6 contrast is available via _m3_vs_m6().

    Parameters
    ----------
    df               : long-format DataFrame with columns patient, timepoint,
                       condition, <outcome>
    outcome          : str — dependent variable column name
    group_var        : str — random-effects grouping column (patients)
    conditions       : list or None — condition values to iterate over;
                       defaults to the four experimental CONDITIONS
    save_name_prefix : str or None — if given, saves one .txt per condition
    extra_fixed        : str or None — extra fixed-effect terms prepended to
                         the formula (e.g. 'side')
    vc_formula         : dict or None — variance components for nested RE
                         (e.g. {"patient_side": "0 + C(patient_side)"})
    try_random_slopes  : bool — if True, first attempts random slopes model
                         (1+timepoint|patient); falls back to random intercept
                         on convergence failure (default True)
    quiet              : bool — suppress stdout printing

    Returns
    -------
    dict  {condition: MixedLMResults}
    """
    if conditions is None:
        conditions = CONDITIONS
    fixed_part = f'{extra_fixed} + ' if extra_fixed else ''
    formula = f'{outcome} ~ {fixed_part}C(timepoint, Treatment("home"))'
    results = {}

    for condition in conditions:
        sub = df[df['condition'] == condition].dropna(subset=[outcome]).copy()

        if len(sub) < 6:
            if not quiet:
                print(f'  Skipping {condition}: insufficient data ({len(sub)} rows)')
            continue

        used_slopes = False
        result = None

        if try_random_slopes:
            try:
                result = _fit_mixedlm(formula, sub, group_var=group_var,
                                      vc_formula=vc_formula,
                                      re_formula=_RE_SLOPES,
                                      strict=True, quiet=quiet)
                used_slopes = True
            except Exception:
                if not quiet:
                    print(f'  {condition}: random slopes failed, falling back to intercept-only')

        if result is None:
            try:
                result = _fit_mixedlm(formula, sub, group_var=group_var,
                                      vc_formula=vc_formula,
                                      re_formula=None,
                                      strict=False, quiet=quiet)
            except Exception as e:
                if not quiet:
                    print(f'  Failed for {condition}: {e}')
                continue

        results[condition] = result
        result._used_slopes = used_slopes
        re_tag = 'slopes' if used_slopes else 'intercept'

        if not quiet:
            singular_tag = '  ⚠ singular RE' if getattr(result, '_singular_re', False) else ''
            print('=' * 70)
            print(f'LME — outcome: {outcome}  |  condition: {condition}  |  RE: {re_tag}{singular_tag}')
            print('=' * 70)
            print(result.summary())

        if save_name_prefix:
            out_path = os.path.join(
                RESULTS_DIR, f'{save_name_prefix}_{condition}.txt')
            with open(out_path, 'w') as f:
                f.write(f'Model: {formula}\n')
                f.write(f'Condition: {condition}\n')
                f.write(f'Random effects: {re_tag}\n\n')
                f.write(result.summary().as_text())
            if not quiet:
                print(f'Saved → {out_path}')

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Leave-one-out sensitivity
# ══════════════════════════════════════════════════════════════════════════════

def run_lme_loo(results_nested, dfs, outcome_map, conditions=None,
                extra_fixed=None, vc_formula=None, alpha=ALPHA, quiet=True,
                save_name=None):
    """
    Leave-one-out sensitivity for per-condition LME results.

    For each (feature, condition, patient): refit on n-1 patients and record
    whether the p-value and sign of each contrast are stable.

    Parameters
    ----------
    results_nested : {feature: {condition: MixedLMResults}}
    dfs            : {feature: DataFrame} — data used to fit each feature
    outcome_map    : {feature: str} — outcome column name per feature
    conditions     : list or None
    extra_fixed    : str or None — fixed-effect terms (e.g. 'side')
    vc_formula     : dict or None — variance components
    alpha          : float — significance threshold
    quiet          : bool

    Returns
    -------
    dict  {feature: {condition: {contrast_label: {'p_sig_n': int,
                                                   'sign_stable_n': int,
                                                   'n_loo': int}}}}
    """
    if conditions is None:
        conditions = CONDITIONS

    fixed_part = f'{extra_fixed} + ' if extra_fixed else ''
    _tp_keys = [
        ('C(timepoint, Treatment("home"))[T.M3]', 'M3 vs home'),
        ('C(timepoint, Treatment("home"))[T.M6]', 'M6 vs home'),
        (None,                                     'M3 vs M6'),
    ]

    loo_out = {}

    for feat, cond_results in results_nested.items():
        df_feat = dfs.get(feat)
        outcome = outcome_map.get(feat)
        if df_feat is None or outcome is None:
            continue

        formula = f'{outcome} ~ {fixed_part}C(timepoint, Treatment("home"))'
        loo_out[feat] = {}

        for condition in conditions:
            main_res = cond_results.get(condition)
            if main_res is None:
                continue

            sub = df_feat[df_feat['condition'] == condition].dropna(
                subset=[outcome]).copy()
            patients = sub['patient'].unique()

            main_signs = {}
            main_sig   = {}
            main_d     = {}
            for key, c_lbl in _tp_keys:
                if key is not None:
                    # NOTE: this is statsmodels' Wald p, used ONLY as a
                    # leave-one-out stability diagnostic — it is never reported,
                    # never FDR-corrected, and never reaches a CSV as a p-value.
                    # Re-permuting inside every LOO fold would cost ~10M model
                    # fits (10 folds x 5000 permutations x 208 cells, ~30 h) to
                    # refine a fold-agreement count, so the cheap statistic is
                    # kept here and the reported p stays the permutation one.
                    p_main = main_res.pvalues.get(key, np.nan)
                    coef_m = main_res.fe_params.get(key, np.nan)
                    main_signs[c_lbl] = np.sign(coef_m)
                else:
                    coef_m, _, _, p_main = _m3_vs_m6(main_res)
                    main_signs[c_lbl] = np.sign(coef_m)
                main_sig[c_lbl] = (not np.isnan(p_main)) and (p_main < alpha)
                # full-model |d| — needed for the fragility guard
                main_d[c_lbl] = cohen_d_lme(coef_m, main_res)

            counts = {c_lbl: {'consistent': 0, 'sign_stable': 0}
                      for _, c_lbl in _tp_keys}
            n_converged  = 0
            n_singular   = 0

            for pat in patients:
                sub_loo = sub[sub['patient'] != pat]
                if len(sub_loo) < 6:
                    continue
                try:
                    res_loo = _fit_mixedlm(formula, sub_loo,
                                           vc_formula=vc_formula, quiet=True)
                    n_converged += 1
                    if getattr(res_loo, '_singular_re', False):
                        n_singular += 1
                except Exception:
                    continue

                for key, c_lbl in _tp_keys:
                    if key is not None:
                        p    = res_loo.pvalues.get(key, np.nan)
                        coef = res_loo.fe_params.get(key, np.nan)
                    else:
                        coef, _, _, p = _m3_vs_m6(res_loo)

                    if not np.isnan(p):
                        fold_sig = p < alpha
                        if fold_sig == main_sig[c_lbl]:
                            counts[c_lbl]['consistent'] += 1
                    if (not np.isnan(coef)
                            and not np.isnan(main_signs[c_lbl])
                            and np.sign(coef) == main_signs[c_lbl]):
                        counts[c_lbl]['sign_stable'] += 1

            loo_out[feat][condition] = {
                c_lbl: {
                    'consistent_n':  counts[c_lbl]['consistent'],
                    'sign_stable_n': counts[c_lbl]['sign_stable'],
                    'n_loo':         n_converged,
                    'singular_re_n': n_singular,
                    'cohen_d':       main_d[c_lbl],
                }
                for _, c_lbl in _tp_keys
            }

            if not quiet:
                print(f'  LOO {feat} | {condition}: '
                      f'{n_converged}/{len(patients)} folds converged')

    if save_name:
        rows = []
        for feat, cond_dict in loo_out.items():
            for cond, contrast_dict in cond_dict.items():
                for contrast, vals in contrast_dict.items():
                    # Round first, then flag from the ROUNDED d, so anything
                    # recomputing is_fragile() off this CSV agrees exactly.
                    d_csv = _round_d(vals.get('cohen_d', np.nan))
                    rows.append({
                        'feature':        feat,
                        'condition':      cond,
                        'contrast':       contrast,
                        'consistent_n':   vals['consistent_n'],
                        'sign_stable_n':  vals['sign_stable_n'],
                        'n_loo':          vals['n_loo'],
                        'singular_re_n':  vals.get('singular_re_n', 0),
                        'cohen_d':        d_csv,
                        'fragile':        is_fragile(vals['sign_stable_n'],
                                                     vals['n_loo'], d_csv),
                    })
        df_loo = pd.DataFrame(rows)
        path = os.path.join(RESULTS_DIR, f'{save_name}.csv')
        df_loo.to_csv(path, index=False)
        print(f'Saved → {path}')

    return loo_out


# ══════════════════════════════════════════════════════════════════════════════
# FDR correction
# ══════════════════════════════════════════════════════════════════════════════

def apply_fdr_lme(results_nested, resample_results, alpha=ALPHA,
                  conditions=None):
    """
    Benjamini-Hochberg FDR correction, ONE FAMILY PER TIMEPOINT CONTRAST.

    Three families per call — `M3 vs home`, `M6 vs home`, `M3 vs M6` — each
    spanning every feature x condition, so a family holds exactly one test per
    (feature, condition).

    Why not one pool of all three
    -----------------------------
    The three contrasts of a model are linear functions of the same two fitted
    coefficients: `M3 vs M6` IS `M3_coef - M6_coef`.  Pooling them builds a
    family largely out of near-duplicates, and the two home-referenced
    contrasts — the substitution question this study asks — end up paying for a
    post-hoc contrast that carries no independent information.  This is the same
    rule `lme_rest_mvt_diff.apply_fdr_simple_effects` applies when it splits by
    stimulation state rather than pooling contrasts of one fitted vector, and
    the same rule `lme_rest_mvt_diff.apply_fdr` applies when it corrects within
    each term.  All three now agree: a family = features x conditions, within
    one question.

    Splitting is not uniformly more permissive.  BH depends on rank position,
    not family size alone: in a shared pool, `M3 vs M6` cells can ride on the
    small p-values of the home contrasts.

    Sides are still never pooled: `bilateral` is a function of L and R, so a
    family spanning sides would be built from overlapping data.

    Parameters
    ----------
    results_nested : {feature: {condition: MixedLMResults}}
    alpha          : float
    conditions     : list or None

    Returns
    -------
    dict  {feature: {condition: {contrast_label: fdr_corrected_p}}}
    """
    if conditions is None:
        conditions = CONDITIONS

    _tp_keys = [
        ('C(timepoint, Treatment("home"))[T.M3]', 'M3 vs home'),
        ('C(timepoint, Treatment("home"))[T.M6]', 'M6 vs home'),
        (None,                                     'M3 vs M6'),
    ]

    entries = []  # (feat, cond, c_lbl, p_raw)
    for feat, cond_results in results_nested.items():
        for cond in conditions:
            res = cond_results.get(cond)
            if res is None:
                continue
            for _key, c_lbl in _tp_keys:
                # PERMUTATION p, never res.pvalues — this module reports no Wald
                # quantity (see the resampling section above).  A cell without a
                # resampled p is left out of the family rather than silently
                # back-filled with a Wald value.
                p = (resample_results.get(feat, {}).get(cond, {})
                     .get(c_lbl, {}).get('p', np.nan)
                     if resample_results else np.nan)
                entries.append((feat, cond, c_lbl, float(p)))

    if not entries:
        return {}

    pvals = np.array([e[3] for e in entries])
    labels = np.array([e[2] for e in entries])
    p_fdr  = np.full_like(pvals, np.nan)
    # One BH call per contrast label — the families are independent pools.
    for c_lbl in {lbl for _, _, lbl, _ in entries}:
        sel = (labels == c_lbl) & ~np.isnan(pvals)
        if sel.sum() == 0:
            continue
        _, p_fdr[sel], _, _ = multipletests(pvals[sel], alpha=alpha,
                                            method='fdr_bh')

    fdr_dict = {}
    for (feat, cond, c_lbl, _), pq in zip(entries, p_fdr):
        fdr_dict.setdefault(feat, {}).setdefault(cond, {})[c_lbl] = float(pq)

    return fdr_dict


# ══════════════════════════════════════════════════════════════════════════════
# Summary table
# ══════════════════════════════════════════════════════════════════════════════

def summary_table_per_condition(results_nested, resample_results=None,
                                alpha=ALPHA, conditions=None, fdr_results=None,
                                save_name=None):
    """
    P-value summary table for per-condition LME results.

    Rows = bands/features.
    Columns = one per (condition, timepoint contrast):
        <condition> | M3 vs home
        <condition> | M6 vs home
        <condition> | M3 vs M6   (post-hoc)

    Parameters
    ----------
    results_nested : dict  {feature: {condition: MixedLMResults}}
    alpha          : float — significance threshold (default 0.05)
    conditions     : list or None — ordered list of conditions to include;
                     defaults to the four experimental CONDITIONS
    save_name      : str or None — CSV filename (without extension)

    Returns
    -------
    DataFrame
    """
    if conditions is None:
        conditions = CONDITIONS
    tp_keys  = [
        'C(timepoint, Treatment("home"))[T.M3]',
        'C(timepoint, Treatment("home"))[T.M6]',
    ]
    tp_short = ['M3 vs home', 'M6 vs home']

    def _fmt(p, q=np.nan):
        if np.isnan(p):
            return 'n/a'
        s = f'{p:.3f}{"*" if p < alpha else ""}'
        if not np.isnan(q):
            s += f' (q={q:.3f}{"*" if q < alpha else ""})'
        return s

    def _fmt_d(d):
        return f'{d:+.2f}({_d_label(d)})' if not np.isnan(d) else 'n/a'

    rows = []
    for feat, cond_results in results_nested.items():
        row = {'feature': feat}
        for condition in conditions:
            res = cond_results.get(condition)
            for key, short in zip(tp_keys, tp_short):
                col   = f'{condition} | {short}'
                col_d = f'{condition} | {short} d'
                if res is None:
                    row[col]   = 'n/a'
                    row[col_d] = 'n/a'
                else:
                    p    = (resample_results.get(feat, {}).get(condition, {})
                            .get(short, {}).get('p', np.nan)
                            if resample_results else np.nan)
                    coef = res.fe_params.get(key, np.nan)
                    q    = (fdr_results.get(feat, {}).get(condition, {}).get(short, np.nan)
                            if fdr_results else np.nan)
                    row[col]   = _fmt(p, q)
                    row[col_d] = _fmt_d(cohen_d_lme(coef, res))
            # Post-hoc: M3 vs M6
            col_m3m6   = f'{condition} | M3 vs M6'
            col_d_m3m6 = f'{condition} | M3 vs M6 d'
            if res is None:
                row[col_m3m6]   = 'n/a'
                row[col_d_m3m6] = 'n/a'
            else:
                coef_m3m6 = _contrast_est_z(res, CONTRAST_WEIGHTS['M3 vs M6'])[0]
                p_m3m6 = (resample_results.get(feat, {}).get(condition, {})
                          .get('M3 vs M6', {}).get('p', np.nan)
                          if resample_results else np.nan)
                q_m3m6 = (fdr_results.get(feat, {}).get(condition, {}).get('M3 vs M6', np.nan)
                          if fdr_results else np.nan)
                row[col_m3m6]   = _fmt(p_m3m6, q_m3m6)
                row[col_d_m3m6] = _fmt_d(cohen_d_lme(coef_m3m6, res))
        rows.append(row)

    df = pd.DataFrame(rows).set_index('feature')
    print('\n[Per-condition LME] P-value summary (* = p < 0.05):')
    print(df.to_string())

    if save_name:
        # path = os.path.join(RESULTS_DIR, f'{save_name}.csv')
        # df.to_csv(path)
        # print(f'Saved → {path}')
        pass
    return df


def save_m3_m6_summary(results_nested, resample_results=None, fdr_results=None,
                        loo_results=None, conditions=None, save_name=None):
    """
    Extract all three timepoint contrasts for every (feature, condition).

    Saves tidy CSV with one row per (feature, condition, contrast):
        contrast: 'M3 vs home' | 'M6 vs home' | 'M3 vs M6'
        feature, condition, contrast,
        beta, ci_lo, ci_hi, p, p_fdr, cohen_d, d_label,
        sig, sig_fdr, n_obs, n_patients, converged,
        loo_consistent, loo_sign_stable, loo_n, loo_singular_re, loo_fragile

    n_obs / n_patients / converged are carried so a consumer never has to open
    the per-model .txt reports to state them.  They matter most for the
    detection-conditional features (peak_beta_*), where n varies per cell and a
    non-converged fit must not be quoted.
    """
    _K_M3  = 'C(timepoint, Treatment("home"))[T.M3]'
    _K_M6  = 'C(timepoint, Treatment("home"))[T.M6]'
    _CONTRASTS = [
        ('M3 vs home', _K_M3),
        ('M6 vs home', _K_M6),
        ('M3 vs M6',   None),   # post-hoc
    ]

    if conditions is None:
        conditions = CONDITIONS

    def _coef_ci_p(res, feat, condition, contrast_label):
        """(coef, ci_lo, ci_hi, p, p_boot) for one contrast.

        beta comes from the fit; the interval is the CLUSTER BOOTSTRAP percentile
        and p is the PERMUTATION p.  Neither res.pvalues nor res.conf_int() is
        read — both are Wald quantities on a normal reference, which this module
        no longer reports.  A cell with no resampling result yields NaN rather
        than a Wald substitute.
        """
        r = (resample_results.get(feat, {}).get(condition, {}).get(contrast_label, {})
             if resample_results else {})
        beta = r.get('beta', np.nan)
        if not np.isfinite(beta):
            # fall back to the fitted coefficient for beta only — never for p/CI
            w = CONTRAST_WEIGHTS.get(contrast_label, {})
            beta = _contrast_est_z(res, w)[0]
        return (beta, r.get('ci_lo', np.nan), r.get('ci_hi', np.nan),
                r.get('p', np.nan), r.get('p_boot', np.nan))

    rows = []
    for feat, cond_results in results_nested.items():
        for condition in conditions:
            res = cond_results.get(condition)
            if res is None:
                continue
            for contrast_label, _key in _CONTRASTS:
                coef, lo, hi, p, p_boot = _coef_ci_p(res, feat, condition,
                                                     contrast_label)
                # Round first, then flag from the ROUNDED d (see _round_d)
                d     = _round_d(cohen_d_lme(coef, res))
                p_fdr = (fdr_results.get(feat, {}).get(condition, {}).get(contrast_label, np.nan)
                         if fdr_results else np.nan)
                loo   = (loo_results.get(feat, {}).get(condition, {}).get(contrast_label, {})
                         if loo_results else {})
                rows.append({
                    'feature':         feat,
                    'condition':       condition,
                    'contrast':        contrast_label,
                    'beta':            round(float(coef), 4)   if not np.isnan(coef)  else np.nan,
                    'ci_lo':           round(float(lo),   4)   if not np.isnan(lo)    else np.nan,
                    'ci_hi':           round(float(hi),   4)   if not np.isnan(hi)    else np.nan,
                    # `p` is the PERMUTATION p and `ci_*` the cluster-bootstrap
                    # percentile interval; p_boot is the p implied by those same
                    # draws, carried for audit.  p_boot runs at ~2x nominal
                    # type-I error at this cluster count, so it is a preview and
                    # `p` is the result — see the resampling section's header.
                    'p':               round(float(p),    4)   if not np.isnan(p)     else np.nan,
                    'p_boot':          round(float(p_boot), 4) if not np.isnan(p_boot) else np.nan,
                    'p_fdr':           round(float(p_fdr), 4)  if not np.isnan(p_fdr) else np.nan,
                    'ci_kind':         'cluster bootstrap' if not np.isnan(lo) else 'none',
                    'cohen_d':         d,
                    'd_label':         _d_label(d),
                    # Variance components behind cohen_d, so d = beta/sqrt(tau2+sigma2)
                    # is reproducible from the CSV and the model ICC is visible
                    # beside it (same convention as lme_rest_mvt_diff.py).
                    **_variance_components(res),
                    'sig':             bool(p < 0.05)     if not np.isnan(p)     else False,
                    'sig_fdr':         bool(p_fdr < 0.05) if not np.isnan(p_fdr) else False,
                    'n_obs':           int(res.nobs),
                    'n_patients':      _n_groups(res),
                    'converged':       bool(getattr(res, 'converged', True)),
                    'full_singular_re': bool(getattr(res, '_singular_re', False)),
                    'loo_consistent':  loo.get('consistent_n',  np.nan),
                    'loo_sign_stable': loo.get('sign_stable_n', np.nan),
                    'loo_n':           loo.get('n_loo',         np.nan),
                    'loo_singular_re': loo.get('singular_re_n', np.nan),
                    'loo_fragile':     is_fragile(loo.get('sign_stable_n', np.nan),
                                                  loo.get('n_loo', np.nan), d),
                })
    df = pd.DataFrame(rows)
    if save_name and not df.empty:
        path = os.path.join(RESULTS_DIR, f'{save_name}.csv')
        df.to_csv(path, index=False)
        print(f'Saved → {path}')
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Results table
# ══════════════════════════════════════════════════════════════════════════════

_COND_SHORT = {
    'rest_stimOFF': 'Rest / OFF',
    'mvt_stimOFF':  'Mvt / OFF',
    'rest_stimON':  'Rest / ON',
    'mvt_stimON':   'Mvt / ON',
    'stimOFF':      'Stim OFF',
    'stimON':       'Stim ON',
}
_MUTED_FILL = {                  # sig trivial (p<0.05, |d|<0.5) — pale tint
    'rest_stimOFF': '#FEF3C7',
    'mvt_stimOFF':  '#FDE8C0',
    'rest_stimON':  '#C0E8E7',
    'mvt_stimON':   '#B8D4E8',
    'stimOFF':      '#E0E0E0',
    'stimON':       '#C8C8C8',
}
_SOLID_FILL = {                  # sig meaningful (p<0.05, |d|≥0.5) — richer tint
    'rest_stimOFF': '#FCD34D',
    'mvt_stimOFF':  '#FDBA74',
    'rest_stimON':  '#34D399',
    'mvt_stimON':   '#60A5FA',
    'stimOFF':      '#D1D5DB',
    'stimON':       '#9CA3AF',
}
_HDR_FONT = {
    'rest_stimOFF': '#1A1A1A',
    'mvt_stimOFF':  '#1A1A1A',
    'rest_stimON':  'white',
    'mvt_stimON':   'white',
    'stimOFF':      '#1A1A1A',
    'stimON':       'white',
}
_HDR_FILL_FALLBACK = '#AAAAAA'
_UNDERPOW_COLOR    = '#FDE68A'  # amber-200 — p≥0.05 but |d|≥0.5

# Single 5-level pastel colorscale used for all conditions
_LME_COLORSCALE = [
    [0.00, '#FFFFFF'],    [0.24, '#FFFFFF'],   # z=0.00 — truly null        (white)
    [0.25, '#F1F5F9'],    [0.49, '#F1F5F9'],   # z=0.25 — small non-sig    (slate-100)
    [0.50, '#FEF3C7'],    [0.74, '#FEF3C7'],   # z=0.50 — underpowered     (amber-100)
    [0.75, '#DBEAFE'],    [0.99, '#DBEAFE'],   # z=0.75 — sig trivial      (blue-100)
    [1.00, '#93C5FD'],                          # z=1.00 — sig+meaningful   (blue-300)
]


def _cell_z(p, d, alpha=ALPHA):
    """
    Map (p-value, Cohen's d) to a discrete z level for heatmap coloring.

    z=1.0  sig + meaningful  (p<alpha, |d|≥0.5)  → solid condition color
    z=0.75 sig trivial       (p<alpha, |d|<0.5)  → pale condition color
    z=0.5  underpowered ⚠   (p≥alpha, |d|≥0.5)  → amber
    z=0.25 small non-sig     (p≥alpha, 0.2≤|d|<0.5) → light gray
    z=0.0  truly null        (p≥alpha, |d|<0.2)  → white
    """
    if np.isnan(p):
        return 0.0
    d_abs = abs(d) if not np.isnan(d) else 0.0
    if p < alpha:
        return 1.0 if d_abs >= 0.5 else 0.75
    if d_abs >= 0.5:
        return 0.5
    if d_abs >= 0.2:
        return 0.25
    return 0.0


def _lme_colorscale(_cond=None):
    """Universal 5-level colorscale — same for all conditions."""
    return _LME_COLORSCALE


def plot_lme_table(results_nested,
                   resample_results=None,
                   title='Per-condition LME — results table',
                   conditions=None,
                   alpha=ALPHA,
                   fdr_results=None,
                   loo_results=None,
                   save_name=None):
    """
    Annotated heatmap figure — one subplot per condition, stacked vertically.

    Layout per subplot:
      x-axis  = features (column names, rotated -45 deg).
      y-axis  = 3 session contrasts (M3 vs home, M6 vs home, M3 vs M6).
      Color   = muted condition fill if p < alpha, white if n.s.
      Text    = per-cell annotation: beta*, [lo, hi], p=.xxx (3 lines).

    go.Heatmap drives background color; fig.add_annotation handles multi-line
    HTML text (go.Heatmap texttemplate does not support line breaks).

    Parameters
    ----------
    results_nested : dict {feature_label: {condition: MixedLMResults}}
    title          : str
    conditions     : list or None -- defaults to four experimental CONDITIONS
    alpha          : float -- significance threshold
    save_name      : str or None -- base filename (single HTML saved)
    """
    if conditions is None:
        conditions = CONDITIONS
    features = list(results_nested.keys())
    if not features:
        return None

    tp_keys = [
        'C(timepoint, Treatment("home"))[T.M3]',
        'C(timepoint, Treatment("home"))[T.M6]',
    ]
    contrasts = [
        (tp_keys[0], 'M3 vs home'),
        (tp_keys[1], 'M6 vs home'),
        (None,       'M3 vs M6'),
    ]
    contrast_labels = [lbl for _, lbl in contrasts]

    n_cond = len(conditions)

    fig = make_subplots(
        rows=n_cond, cols=1,
        subplot_titles=[COND_LABELS.get(c, c) for c in conditions],
        vertical_spacing=0.20 if n_cond <= 2 else 0.10,
    )

    for row_idx, cond in enumerate(conditions):

        # ── Build z + hover matrices [n_contrasts × n_features] ──────────
        z_data     = []
        hover_data = []
        for key, c_lbl in contrasts:
            z_row     = []
            hover_row = []
            for feat in features:
                res = results_nested[feat].get(cond)
                if res is None:
                    z_row.append(np.nan)
                    hover_row.append(f'<b>{feat}</b> — {c_lbl}')
                    continue
                # permutation p + fitted beta; no Wald quantity is read here
                coef = _contrast_est_z(res, CONTRAST_WEIGHTS[c_lbl])[0]
                p    = (resample_results.get(feat, {}).get(cond, {})
                        .get(c_lbl, {}).get('p', np.nan)
                        if resample_results else np.nan)
                d     = cohen_d_lme(coef, res)
                p_fdr = (fdr_results.get(feat, {}).get(cond, {}).get(c_lbl, np.nan)
                         if fdr_results else np.nan)
                p_eff = p_fdr if not np.isnan(p_fdr) else p
                z_row.append(_cell_z(p_eff, d, alpha))
                hover_row.append(f'<b>{feat}</b> — {c_lbl}')
            z_data.append(z_row)
            hover_data.append(hover_row)

        fig.add_trace(go.Heatmap(
            x=features,
            y=contrast_labels,
            z=z_data,
            text=hover_data,
            colorscale=_lme_colorscale(cond),
            showscale=False,
            zmin=0, zmax=1,
            xgap=2, ygap=2,
            hovertemplate='%{text}<extra></extra>',
        ), row=row_idx + 1, col=1)

        # ── Per-cell text annotations ──────────────────────────────────────
        for key, c_lbl in contrasts:
            for feat in features:
                res = results_nested[feat].get(cond)
                if res is None:
                    txt = '<i>n/a</i>'
                    fc  = '#CCCCCC'
                else:
                    # beta from the fit; p and the interval from the resampling
                    coef = _contrast_est_z(res, CONTRAST_WEIGHTS[c_lbl])[0]
                    _r   = (resample_results.get(feat, {}).get(cond, {})
                            .get(c_lbl, {}) if resample_results else {})
                    p  = _r.get('p',     np.nan)
                    lo = _r.get('ci_lo', np.nan)
                    hi = _r.get('ci_hi', np.nan)

                    d     = cohen_d_lme(coef, res)
                    p_fdr = (fdr_results.get(feat, {}).get(cond, {}).get(c_lbl, np.nan)
                             if fdr_results else np.nan)
                    p_eff = p_fdr if not np.isnan(p_fdr) else p
                    z     = _cell_z(p_eff, d, alpha)
                    d_str = f'd={d:+.2f}({_d_label(d)})' if not np.isnan(d) else ''
                    p_str = (f'p={p:.3f}<br>q={p_fdr:.3f}'
                             if not np.isnan(p_fdr) else f'p={p:.3f}')

                    if np.isnan(coef):
                        txt = '<i>n/a</i>'
                        fc  = '#CCCCCC'
                    elif z == 1.0:   # sig + meaningful
                        txt = (f'<b>{coef:+.2f}*</b><br>'
                               f'[{lo:.2f}, {hi:.2f}]<br>'
                               f'{p_str}<br>'
                               f'<i>{d_str}</i>')
                        fc  = '#1E3A5F'  # navy on blue-300
                    elif z == 0.75:  # sig but trivial effect
                        txt = (f'<b>{coef:+.2f}*</b><br>'
                               f'[{lo:.2f}, {hi:.2f}]<br>'
                               f'{p_str}<br>'
                               f'<i>{d_str}</i><br>'
                               f'<i>⚠sm eff</i>')
                        fc  = '#555555'
                    elif z == 0.5:   # underpowered — real effect, missed
                        txt = (f'{coef:+.2f}<br>'
                               f'[{lo:.2f}, {hi:.2f}]<br>'
                               f'{p_str}<br>'
                               f'<b><i>⚠ underpow</i></b><br>'
                               f'<b><i>{d_str}</i></b>')
                        fc  = '#92400E'  # amber-800
                    elif z == 0.25:  # small non-sig
                        txt = (f'{coef:+.2f}<br>'
                               f'[{lo:.2f}, {hi:.2f}]<br>'
                               f'{p_str}<br>'
                               f'<i>{d_str}</i>')
                        fc  = '#888888'
                    else:            # truly null
                        txt = (f'{coef:+.2f}<br>'
                               f'[{lo:.2f}, {hi:.2f}]<br>'
                               f'{p_str}<br>'
                               f'<i>{d_str}</i>')
                        fc  = '#AAAAAA'

                    if getattr(res, '_used_slopes', False) and not np.isnan(coef):
                        txt += '<br><i>~slopes</i>'

                    if getattr(res, '_singular_re', False) and not np.isnan(coef):
                        txt += '<br><i>⚠singRE</i>'

                    if loo_results is not None:
                        _lc = loo_results.get(feat, {}).get(cond, {}).get(c_lbl, {})
                        if _lc:
                            txt += (f'<br><i>loo:{_lc["consistent_n"]}'
                                    f'/{_lc["n_loo"]}</i>')

                fig.add_annotation(
                    x=feat, y=c_lbl,
                    text=txt,
                    showarrow=False,
                    font=dict(family=FONT, size=SIZE_ANNOT, color=fc),
                    align='center',
                    row=row_idx + 1, col=1,
                )

    # X-axis: rotate feature names; only renders on bottom subplot (shared)
    fig.update_xaxes(
        tickangle=-45,
        tickfont=dict(family=FONT, size=SIZE_TICK),
        showgrid=False, zeroline=False,
    )
    fig.update_yaxes(
        autorange='reversed',
        tickfont=dict(family=FONT, size=SIZE_TICK),
        showgrid=False, zeroline=False,
    )

    # Style subplot title annotations
    cond_label_set = {COND_LABELS.get(c, c) for c in conditions}
    for ann in fig.layout.annotations:
        if ann.text in cond_label_set:
            ann.font = dict(family=FONT, size=SIZE_HEAD, color=_INK)
            ann.x = 0.0
            ann.xanchor = 'left'

    cell_h_px     = 120
    cell_w_px     = 80
    per_subplot_h = len(contrasts) * cell_h_px + 55
    total_h       = n_cond * per_subplot_h + 130
    total_w       = len(features) * cell_w_px + 180

    apply_style(fig, title=title, height=total_h,
                width=total_w, showlegend=False)
    fig.update_layout(margin=dict(l=110, r=20, t=65, b=130))

    if save_name:
        _save_fig(fig, RESULTS_DIR, save_name)

    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def _fit_tremor_features(results_percond, dfs_percond, outcome_map,
                         build_df, save_suffix, **run_kw):
    """Fit the patient-specific tremor bands into the same result dicts.

    Tremor-locked power is modelled here rather than in its own module so that
    it is corrected together with band power, aperiodic and broadband — every
    feature this script fits enters the same families, and a feature analysed
    on the side would otherwise be corrected against nothing.

    Its bands are per patient AND per hemisphere (f0 comes from the
    contralateral limb, and can differ between L and R), so a bilateral
    value mixes two integration windows on purpose: the estimand is
    'tremor-locked power', not power in a shared frequency band.

    The tremor keys live in the ordinary feature pickles; pipeline.py adds
    them via add_tremor_bands(). Missing keys are a notice, not a crash — the rest of the analysis does
    not depend on them — but every other feature's p_fdr then reflects a family
    without the tremor bands, so the run is not comparable with a complete one.
    """
    try:
        df_tr = build_df(load_all_sessions_tremor())
    except (FileNotFoundError, OSError) as exc:
        print(f'\n! Tremor bands skipped ({type(exc).__name__}: {exc}) — '
              f'run add_tremor_bands.py to include them. NOTE: the FDR family '
              f'is smaller by that many tests.', flush=True)
        return

    # Absolute power, log-transformed exactly like abs_{band}.
    df_tr['power'] = np.log(df_tr['power'].clip(lower=_CLIP))
    print(f'\nTremor band power DataFrame : {len(df_tr)} rows')
    for band in df_tr['band'].unique():
        df_band = df_tr[df_tr['band'] == band].dropna(subset=['power']).copy()
        if len(df_band) < 6:
            continue
        feat = f'abs_{band}'          # abs_tremor_f0 / abs_tremor_2f
        print(f'\n── Tremor band: {band} ──')
        results_percond[feat] = run_lme_per_condition(
            df_band, outcome='power',
            save_name_prefix=f'lme_percond_{feat}_{save_suffix}', **run_kw)
        dfs_percond[feat] = df_band
        outcome_map[feat] = 'power'


def _fit_peak_features(results_percond, dfs_percond, outcome_map,
                       build_df, save_suffix, **run_kw):
    """Fit the FOOOF beta-peak parameters into the same result dicts.

    Features are `peak_beta_cf`, `peak_beta_pw`, `peak_beta_bw` (Hz, log10
    above the aperiodic fit, and Hz respectively — no transform is applied to
    any of them, unlike the power features).  Beta is the only band with enough
    detected peaks to model.

    Two things separate these from every other feature, and neither is handled
    here — the shared drivers already behave correctly:

    * They are CONDITIONAL ON DETECTION and missing-not-at-random: a hemisphere
      with no detectable peak is plausibly a low-power one and is absent by
      construction.  `run_lme_per_condition` drops NaN per outcome, so every
      detected peak contributes and patients enter with 1-3 timepoints; n is
      smaller here than for the power features and must be read beside the
      estimate.  The detection rate itself is reported by peak_params_analysis
      and band_peak_detection, and is arguably the primary result.
    * `build_peak_params_df(side='bilateral')` averages the hemispheres that
      HAVE a peak and keeps a patient with only one, where the band-power
      builders require both channels.  Requiring both peaks would discard most
      of the sample; `n_hemi` in that frame records which rows are means.

    They are fitted here rather than in their own module so that they share one
    BH family with every other feature; a separate family corrected them
    against nothing and left every other q computed as if they had not been run.
    """
    df_pk = build_df()
    if df_pk is None or df_pk.empty:
        print('\n! Peak parameters skipped: no rows', flush=True)
        return

    print(f'\nPeak parameters ({PEAK_BAND}) DataFrame : {len(df_pk)} rows')
    for metric in PEAK_METRICS:
        df_m = df_pk.dropna(subset=[metric]).copy()
        if len(df_m) < 6:
            print(f'Skipping peak {metric}: insufficient detections')
            continue
        feat = f'peak_{PEAK_BAND}_{metric}'
        print(f'\n── Peak ({PEAK_BAND}): {metric} ──')
        results_percond[feat] = run_lme_per_condition(
            df_m, outcome=metric,
            save_name_prefix=f'lme_percond_{feat}_{save_suffix}', **run_kw)
        dfs_percond[feat] = df_m
        outcome_map[feat] = metric


def main(side='L', side_label=None):
    """
    Run all per-condition LME analyses for one electrode side.

    All feature types are fitted in a single call and collected into two dicts:
        results_percond  : {feature_label: {condition: MixedLMResults}}
        results_contrast : {feature_label: {stim_cond: MixedLMResults}}

    Feature label conventions
    -------------------------
        abs_{band}          absolute band power   (band in ANALYSIS_BANDS)
        exponent / offset   FOOOF aperiodic parameters
        broadband_power     summed aperiodic fit (linear)
        flat_abs_{band}     FOOOF-flattened absolute band power (beta only)

    `rel_` / `flat_rel_` labels are not produced: relative power is excluded
    from the statistical analysis.
    """
    _s = side_label if side_label is not None else (
        side if isinstance(side, str) else 'custom')
    print(f'\nLME Analysis  |  side={_s}\n')

    all_sessions = load_all_sessions()

    results_percond = {}   # all per-condition features
    dfs_percond     = {}   # DataFrames for LOO (keyed same as results_percond)
    outcome_map     = {}   # outcome column name per feature

    # ══════════════════════════════════════════════════════════════════════════
    # Per-condition analyses
    # ══════════════════════════════════════════════════════════════════════════

    # ── Absolute band power ───────────────────────────────────────────────────
    for pt in ('abs',):   # 'rel' excluded from statistical analysis
        df_bp = build_bandpower_df(all_sessions, side=side, power_type=pt)
        if pt == 'abs':
            df_bp['power'] = np.log(df_bp['power'].clip(lower=_CLIP))
        print(f'Band power ({pt}) DataFrame : {len(df_bp)} rows')
        for band in _analysis_bands(df_bp):
            df_band = df_bp[df_bp['band'] == band].dropna(subset=['power']).copy()
            if len(df_band) < 6:
                continue
            print(f'\n── Band ({pt}): {band} ──')
            results_percond[f'{pt}_{band}'] = run_lme_per_condition(
                df_band, outcome='power',
                save_name_prefix=f'lme_percond_{pt}_{band}_{_s}')
            dfs_percond[f'{pt}_{band}'] = df_band
            outcome_map[f'{pt}_{band}'] = 'power'

    # ── Tremor-locked bands (patient- and hemisphere-specific) ────────────────
    _fit_tremor_features(
        results_percond, dfs_percond, outcome_map,
        build_df=lambda s: build_tremor_bandpower_df(s, side=side),
        save_suffix=_s)

    # ── Aperiodic (exponent + offset) ─────────────────────────────────────────
    df_ap = build_aperiodic_df(all_sessions, side=side)
    df_ap['exponent'] = np.log(df_ap['exponent'].clip(lower=_CLIP))
    print(f'\nAperiodic DataFrame : {len(df_ap)} rows')
    for outcome in ('exponent', 'offset'):
        df_out = df_ap.dropna(subset=[outcome]).copy()
        if len(df_out) < 6:
            print(f'Skipping aperiodic {outcome}: insufficient data')
            continue
        print(f'\n── Aperiodic: {outcome} ──')
        results_percond[outcome] = run_lme_per_condition(
            df_out, outcome=outcome,
            save_name_prefix=f'lme_percond_aperiodic_{outcome}_{_s}')
        dfs_percond[outcome] = df_out
        outcome_map[outcome] = outcome

    # ── Broadband power ────────────────────────────────────────────────────────
    df_bb = build_broadband_power_df(all_sessions, side=side)
    df_bb['broadband_power'] = np.log(df_bb['broadband_power'].clip(lower=_CLIP))
    print(f'\nBroadband power DataFrame : {len(df_bb)} rows')
    if len(df_bb) >= 6:
        print('\n── Broadband power ──')
        results_percond['broadband_power'] = run_lme_per_condition(
            df_bb, outcome='broadband_power',
            save_name_prefix=f'lme_percond_broadband_{_s}')
        dfs_percond['broadband_power'] = df_bb
        outcome_map['broadband_power'] = 'broadband_power'

    # ── Flattened band power (abs, beta only) ─────────────────────────────────
    for pt in ('abs',):   # 'rel' excluded from statistical analysis
        # Not log-transformed: flattened power is the FOOOF log-space residual
        # (log PSD − log aperiodic), already a log quantity and signed.
        df_flat = build_flattened_bandpower_df(all_sessions, side=side, power_type=pt)
        print(f'\nFlattened band power ({pt}) DataFrame : {len(df_flat)} rows')
        for band in _flat_analysis_bands(df_flat):
            df_band = df_flat[df_flat['band'] == band].dropna(subset=['power']).copy()
            if len(df_band) < 6:
                continue
            print(f'\n── Flat ({pt}): {band} ──')
            results_percond[f'flat_{pt}_{band}'] = run_lme_per_condition(
                df_band, outcome='power',
                save_name_prefix=f'lme_percond_flat_{pt}_{band}_{_s}')
            dfs_percond[f'flat_{pt}_{band}'] = df_band
            outcome_map[f'flat_{pt}_{band}'] = 'power'

    # ── FOOOF beta-peak parameters ────────────────────────────────────────────
    _fit_peak_features(
        results_percond, dfs_percond, outcome_map,
        build_df=lambda: build_peak_params_df(all_sessions, side=side,
                                              band=PEAK_BAND),
        save_suffix=_s)

    # Resampling inference — the expensive step and the source of every
    # reported p and interval.  Nothing downstream reads res.pvalues.
    print(f'\nResampling ({_N_PERM} permutations + {_N_BOOT} bootstrap '
          f'draws per cell, {_N_JOBS} worker(s))…', flush=True)
    resample_percond = resample_lme(results_percond, dfs_percond, outcome_map,
                                    n_perm=_N_PERM, n_boot=_N_BOOT,
                                    n_jobs=_N_JOBS)

    fdr_percond = apply_fdr_lme(results_percond, resample_percond)

    summary_table_per_condition(results_percond, resample_percond,
                                fdr_results=fdr_percond,
                                save_name=f'pval_percond_all_{_s}')

    loo_percond = run_lme_loo(results_percond, dfs_percond, outcome_map,
                              quiet=False,
                              save_name=f'loo_percond_{_s}')

    save_m3_m6_summary(results_percond, resample_results=resample_percond,
                       fdr_results=fdr_percond,
                       loo_results=loo_percond,
                       save_name=f'lme_m3m6_summary_{_s}')

    plot_lme_table(
        results_percond,
        resample_results=resample_percond,
        title=f'Session differences — LME results  |  {_s}',
        fdr_results=fdr_percond,
        loo_results=loo_percond,
        save_name=f'table_percond_all_{_s}')

    return results_percond


def _resample_budget():
    """Resampling budget from the environment, so a fast pass needs no edit.

        ECOLOVIM_LME_NPERM=0 ECOLOVIM_LME_NBOOT=0   structure-only smoke run
        ECOLOVIM_LME_NPERM=200                      quick preview

    A budget of 0 leaves the corresponding column NaN — it does NOT fall back to
    a Wald p, because this module reports none.  Anything below the defaults is
    a preview, not a result.
    """
    return (int(os.environ.get('ECOLOVIM_LME_NPERM', N_PERM_LME)),
            int(os.environ.get('ECOLOVIM_LME_NBOOT', N_BOOT_LME)))


def _resample_jobs():
    """Worker processes for the resampling step (ECOLOVIM_LME_JOBS, default 1).

    Wall-clock only: cells are independent and each reseeds from the fixed seed,
    so the CSV does not depend on this number.  It is therefore deliberately NOT
    a run_all stage env var — making it one would put the core count into the
    stage fingerprint and force a rebuild whenever it changed.  -1 uses every
    core.
    """
    return int(os.environ.get('ECOLOVIM_LME_JOBS', 1))


_N_PERM, _N_BOOT = _resample_budget()
_N_JOBS = _resample_jobs()


if __name__ == '__main__':
    # Three analysis units, each with its own BH families: left, right, and the
    # equal-weight bilateral mean.
    _SIDES = [
        ('L',         'L',         'Left'),
        ('R',         'R',         'Right'),
        ('bilateral', 'bilateral', 'Bilateral ((L+R)/2)'),
    ]

    _all_percond = {}

    for _side, _sl, _lbl in _SIDES:
        _all_percond[_lbl] = main(side=_side, side_label=_sl)
