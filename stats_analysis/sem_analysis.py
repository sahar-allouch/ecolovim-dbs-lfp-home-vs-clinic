"""Standard error of measurement (SEM) for every ICC cell.

ICC is a ratio, sigma_b^2 / (sigma_b^2 + sigma_c^2 + sigma_e^2), so a low value
has two causes it cannot tell apart: the cohort is homogeneous (sigma_b small)
or the measurement is noisy (sigma_e large).  Separating them needs one absolute
quantity.  This module writes it.

    SEM        = sqrt(MS_error)                 within-subject SD, feature units
    sd_between = sqrt(max((MS_rows - MS_error) / k, 0))
    MDC95      = 1.96 * sqrt(2) * SEM           smallest detectable change

Same two-way ANOVA, same pivot, same log transforms as icc_analysis.py, so the
ICC recomputed here reproduces the stored one exactly (verified: max absolute
difference 4e-16 over 156 bilateral cells).

Two notes on sd_between.  It is the **variance-component** estimator, not the SD
of per-patient means that icc_analysis.py currently reports under that name --
the latter carries sigma_e^2 / k and never reaches zero, which makes
`range_flag` very unlikely to fire.  The moment estimator clamps at 0
when MS_rows <= MS_error, i.e. when the observed between-patient mean square
falls below the error mean square.  No ICC is interpretable for those cells.

Both are written even though each is recoverable from the other and the ICC,
because they answer different questions.  SEM is the one to quote in a paper --
it is a property of the measurement and transfers to another cohort -- whereas
sd_between describes these 9 patients only, and is what says whether a low ICC
means the measurement failed or the cohort did not vary.

Units are inherited from the transform icc_analysis.py applies before fitting:
natural log for abs_* / broadband / exponent, log10 for offset / flat_abs_* /
peak power (FOOOF works in log10), Hz for peak centre frequency and bandwidth.
`SEM_pct` back-transforms the log ones to a multiplicative error and
`sd_between_pct` puts the between-patient spread on that same scale, so the two
can be compared directly.  Both are left NaN for the Hz features, and neither
should be read for flat_abs_beta, a signed log-space residual centred near zero
for which a fold change is arithmetically correct and interpretively useless.

`cv_between` and `cv_sem` put both on ONE dimensionless 0-100% scale so all
thirteen features fit a single axis; see `_cv` for the two constructions and why
flat_abs_beta uses the linear one.

Output
------
    Results/stats/icc/sem_all_sides.csv      468 rows = 3 sides x 156 cells
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from stats_analysis.data_utils import (
    ANALYSIS_BANDS,
    CONDITIONS,
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

OUT_CSV = Path('Results/stats/icc/sem_all_sides.csv')
SIDES = ['bilateral', 'L', 'R']
TP_PAIRS = [('M3', 'home'), ('M3', 'M6'), ('home', 'M6')]
CLIP = 1e-12

# Which log base each feature carries, matching the transforms in icc_analysis.py.
LN_FEATURES = {'abs_delta', 'abs_theta', 'abs_alpha', 'abs_beta',
               'abs_tremor_f0', 'abs_tremor_2f', 'broadband_power', 'exponent'}
LOG10_FEATURES = {'offset', 'flat_abs_beta', 'peak_beta_pw'}

# Features whose analysis variable is the log of ONE positive quantity, so the
# variable back-transforms and its CV is the lognormal one,
# sqrt(exp(sigma^2) - 1), with sigma first put in natural-log units.
#
# flat_abs_beta is deliberately NOT here even though it is a log10 quantity: it
# is a band SUM of the FOOOF log-space residual over roughly 22 bins, so
# exponentiating it gives the product of 22 ratios rather than a value, and its
# lognormal CV runs to 1e15%.  It is an additive index -- the LME and ICC treat
# it that way and never re-transform it -- so its CV is taken linearly, like the
# Hz features.  That puts all thirteen features on one 0-100% axis.
LOGNORMAL_FEATURES = (LN_FEATURES | LOG10_FEATURES) - {'flat_abs_beta'}


def _cv(value: float, feature: str, units: str, mean_value: float) -> float:
    """Coefficient of variation: one dimensionless scale for every feature.

    SEM and sd_between are in whatever units the feature carries -- natural log,
    log10, or Hz -- which is why they cannot share an axis directly.  A CV can:
    for a log-normal quantity it is sqrt(exp(sigma^2) - 1) on the natural-log
    sigma, and for a linear one it is sigma / |mean|.  Both answer "how large is
    this spread relative to the size of the thing", so the two constructions are
    commensurable even though the arithmetic differs.
    """
    if not np.isfinite(value):
        return float('nan')
    if feature in LOGNORMAL_FEATURES:
        sigma = value * np.log(10) if units == 'log10' else value
        return float(np.sqrt(np.expm1(sigma ** 2)) * 100)
    if not np.isfinite(mean_value) or mean_value == 0:
        return float('nan')
    return float(abs(value) / abs(mean_value) * 100)


def _log_base(feature: str) -> float:
    """Base for back-transforming a SEM to a fold change; NaN for Hz features."""
    if feature in LN_FEATURES:
        return float(np.e)
    if feature in LOG10_FEATURES:
        return 10.0
    return float('nan')


def _feature_frames(sess, sess_tr, side):
    """Yield (feature_name, long frame with a 'value' column) for one side.

    Transforms mirror icc_analysis.main() exactly -- ln for absolute power,
    broadband and exponent; nothing for offset (already log10 out of FOOOF),
    flattened power (already a log-space residual) or the peak metrics.
    """
    df_abs = build_bandpower_df(sess, side=side, power_type='abs')
    df_ap = build_aperiodic_df(sess, side=side)
    df_bb = build_broadband_power_df(sess, side=side)
    df_fl = build_flattened_bandpower_df(sess, side=side, power_type='abs')
    df_tr = build_tremor_bandpower_df(sess_tr, side=side)
    df_pk = build_peak_params_df(sess, side=side, band=PEAK_BAND)

    for frame, col in [(df_abs, 'power'), (df_bb, 'broadband_power'),
                       (df_ap, 'exponent')]:
        frame[col] = np.log(frame[col].clip(lower=CLIP))
    df_tr['power'] = np.log(df_tr['power'].clip(lower=CLIP))

    df_ap = df_ap.merge(
        df_bb[['patient', 'timepoint', 'condition', 'broadband_power']],
        on=['patient', 'timepoint', 'condition'], how='left')

    for band in ANALYSIS_BANDS:
        yield f'abs_{band}', df_abs[df_abs['band'] == band].rename(columns={'power': 'value'})
    for band in sorted(df_tr['band'].unique()):
        yield f'abs_{band}', df_tr[df_tr['band'] == band].rename(columns={'power': 'value'})
    yield 'flat_abs_beta', df_fl[df_fl['band'] == 'beta'].rename(columns={'power': 'value'})
    for outcome in ('exponent', 'offset', 'broadband_power'):
        yield outcome, df_ap.rename(columns={outcome: 'value'})
    for metric in PEAK_METRICS:
        yield f'peak_{PEAK_BAND}_{metric}', df_pk.rename(columns={metric: 'value'})


def _cell(values: np.ndarray) -> dict | None:
    """Two-way ANOVA variance components for one n x k matrix of complete rows."""
    n, k = values.shape
    if n < 2:
        return None
    grand = values.mean()
    ss_rows = k * ((values.mean(axis=1) - grand) ** 2).sum()
    ss_cols = n * ((values.mean(axis=0) - grand) ** 2).sum()
    ss_err = ((values - grand) ** 2).sum() - ss_rows - ss_cols

    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (k - 1)
    ms_err = ss_err / ((n - 1) * (k - 1))
    if not np.isfinite(ms_err) or ms_err <= 0:
        return None

    return dict(
        n=n,
        mean_value=float(grand),
        ICC=(ms_rows - ms_err) / (ms_rows + (k - 1) * ms_err
                                  + k * (ms_cols - ms_err) / n),
        SEM=float(np.sqrt(ms_err)),
        sd_between=float(np.sqrt(max((ms_rows - ms_err) / k, 0.0))),
    )


def compute(sides=SIDES) -> pd.DataFrame:
    sess = load_all_sessions()
    sess_tr = load_all_sessions_tremor()

    rows = []
    for side in sides:
        for feature, frame in _feature_frames(sess, sess_tr, side):
            base = _log_base(feature)
            for condition in CONDITIONS:
                sub = frame[frame['condition'] == condition]
                total_sd = float(sub['value'].std(skipna=True))
                pivot = sub.pivot_table(index='patient', columns='timepoint',
                                        values='value', aggfunc='mean')
                for tp_a, tp_b in TP_PAIRS:
                    if not all(tp in pivot.columns for tp in (tp_a, tp_b)):
                        continue
                    stats = _cell(pivot[[tp_a, tp_b]].dropna().values)
                    if stats is None:
                        continue
                    mdc = 1.96 * np.sqrt(2) * stats['SEM']
                    units = ('ln' if feature in LN_FEATURES
                             else 'log10' if feature in LOG10_FEATURES else 'Hz')
                    rows.append(dict(
                        side=side, feature=feature, condition=condition,
                        pair=f'{tp_a} vs {tp_b}', **stats,
                        total_sd=total_sd, MDC95=mdc,
                        SEM_pct=(base ** stats['SEM'] - 1) * 100 if np.isfinite(base) else np.nan,
                        MDC95_pct=(base ** mdc - 1) * 100 if np.isfinite(base) else np.nan,
                        # Between-patient spread on the same multiplicative scale
                        # as SEM_pct, so the two are directly comparable.
                        sd_between_pct=(base ** stats['sd_between'] - 1) * 100
                                       if np.isfinite(base) else np.nan,
                        cv_between=_cv(stats['sd_between'], feature, units,
                                       stats['mean_value']),
                        cv_sem=_cv(stats['SEM'], feature, units,
                                   stats['mean_value']),
                        units=units,
                    ))
    return pd.DataFrame(rows)


# The "High SEM" bar used by classify_cells() / cell_flags() / case_flags() /
# fig_icc_sem_*, replacing the earlier per-side empirical tertile.
#
# Anchored to MDC95 >= 100% (a patient's value must at least double before a
# session-to-session change clears 95% confidence) -- see
# stats_analysis/ICC_CASE_CLASSIFICATION.md for why a fixed, criterion-derived
# bar was chosen over an empirical percentile: no clinically established
# minimal important difference exists for these features (STATISTICAL_METHODS
# Sec 4, equivalence testing withdrawn for the same reason), so "must double"
# is the smallest change-magnitude that needs no domain citation, not a number
# tuned to this dataset's distribution.
#
# Expressed on the cv_sem axis, NOT as the simpler (e^sigma - 1) fold form.
# cv_sem uses the true lognormal coefficient of variation, sqrt(exp(sigma^2)-1)
# (see _cv above); MDC95_pct uses the simpler fold form.  These are different
# functions of sigma and diverge as sigma grows -- the cv_sem-axis cutoff is
# sqrt(exp(sigma_cut^2) - 1), not exp(sigma_cut) - 1.  Checked to reproduce
# MDC95_pct >= 100 exactly on every feature where a fold change is
# well-defined.
#
# sigma_cut is in natural-log units and is base-independent: ln(2) for an
# ln-transformed feature and log10(2) for a log10-transformed one convert to
# the same natural-log sigma once cv_sem's internal ln(10) rescaling (see _cv)
# is undone, since ln(2) = log10(2) * ln(10) exactly.
_SIGMA_CUT_MDC100 = np.log(2) / (1.96 * np.sqrt(2))
SEM_CV_CUTOFF_MDC100 = float(np.sqrt(np.expm1(_SIGMA_CUT_MDC100 ** 2)) * 100)

# Features for which SEM_CV_CUTOFF_MDC100 is an exact statement about a fold
# change.  For the other three -- flat_abs_beta (a signed residual SUM, where
# "doubling" has no meaning) and peak_beta_cf / peak_beta_bw (Hz, not a ratio)
# -- applying the same numeric bar is an extension, not a derived fact: no
# separate criterion for them was established, so the alternative was to
# invent one, which is exactly the arbitrary-threshold problem this bar exists
# to avoid.  Callers that need to distinguish should check membership here.
MDC100_DEFINED_FEATURES = frozenset({
    'abs_delta', 'abs_theta', 'abs_alpha', 'abs_beta', 'abs_tremor_f0',
    'abs_tremor_2f', 'broadband_power', 'exponent', 'offset', 'peak_beta_pw',
})


def main():
    df = compute()
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f'wrote {OUT_CSV}  ({len(df)} rows; '
          f'{df["side"].nunique()} sides x {len(df) // df["side"].nunique()} cells)')

    degenerate = df[df['sd_between'] == 0]
    print(f'\nsd_between == 0 (no recoverable between-patient variance): '
          f'{len(degenerate)} cells')
    if len(degenerate):
        print(degenerate[['side', 'feature', 'condition', 'pair', 'ICC', 'SEM']]
              .round(3).to_string(index=False))

    print('\nmedian SEM per feature x side (native units):')
    print(df.pivot_table(index=['feature', 'units'], columns='side',
                         values='SEM', aggfunc='median')
            .reindex(columns=SIDES).round(3).to_string())


if __name__ == '__main__':
    main()
