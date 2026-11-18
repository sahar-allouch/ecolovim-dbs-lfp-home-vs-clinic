"""Peak detection and availability report for the FOOOF beta-peak parameters.

This module no longer fits any model
------------------------------------
ICC(2,1) and the per-condition LME for `cf`, `pw` and `bw` are run by
icc_analysis.py and lme_analysis.py, alongside band power, tremor bands,
aperiodic and broadband, and they appear in those modules' output tables as
`peak_beta_cf`, `peak_beta_pw` and `peak_beta_bw`.  They were moved out of here
so that the LME shares ONE Benjamini-Hochberg family with every other feature:
a separate family meant the peak parameters were corrected against nothing but
each other, and every other feature's p_fdr was computed as if the peak tests
had not been run.

What remains is the part no other module reports, and the part the docstring of
this module always argued was arguably the primary result: how often a peak is
detected at all.

What it describes
-----------------
The FOOOF peak parameters in `feat['band_peaks_allch'][band]`, shape
(1, n_ch, 3) = [cf, pw, bw].  Assembled by `data_utils.build_peak_params_df`:

    cf  center frequency   (Hz)
    pw  peak power         (log10 power above the aperiodic fit — already log)
    bw  bandwidth          (Hz, censored to PEAK_WIDTH_LIMITS by the fit)

One value per patient x side x condition x timepoint; the PSD is mean-reduced
across epochs before the FOOOF fit, so there is no within-session replication.

Missing-data policy
-------------------
Peak parameters exist only where a peak was detected, so every estimate derived
from them is CONDITIONAL ON DETECTION, and the sample is missing-not-at-random:
a hemisphere with no detectable peak is plausibly a low-power one, and it is
absent by construction.  Those numbers describe peaks that exist, not the
cohort.  The detection rate is what this module reports, and it is the context
in which the ICC/LME rows elsewhere must be read.

Bandwidth caveat
----------------
`bw` is hard-censored at PEAK_WIDTH_LIMITS by the fit.  A bw sitting on a bound
is not a measurement, so `bw_at_bound` counts them here and the summary warns
when any are present — treat every bw result as descriptive.

Bands
-----
Beta only by default — see DEFAULT_BANDS.  Alpha peaks are detected in 17-32 %
of hemispheres, leaving 0-4 complete pairs per cell.

Outputs
-------
    Results/stats/peak_params/peak_params_long.csv
    Results/stats/peak_params/peak_params_availability.csv

Usage
-----
    uv run python stats_analysis/peak_params_analysis.py
    uv run python stats_analysis/peak_params_analysis.py --bands beta alpha
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stats_analysis.data_utils import (  # noqa: E402
    PEAK_METRICS,
    PEAK_WIDTH_LIMITS,
    build_peak_params_df,
    load_all_sessions,
)

SIDES = ['L', 'R']
UNITS = ['L', 'R', 'bilateral']
DEFAULT_BANDS = ['beta']
METRICS       = PEAK_METRICS
METRIC_LABEL  = {'cf': 'Center frequency (Hz)',
                 'pw': 'Peak power (log10 above aperiodic)',
                 'bw': 'Bandwidth (Hz)'}

OUT_DIR = Path(__file__).resolve().parent.parent / 'Results' / 'stats' / 'peak_params'


# ── Data assembly ────────────────────────────────────────────────────────────
def build_frames(all_sessions, bands):
    """{(band, unit): DataFrame} for every unit, plus the L+R pooled frame.

    The pooled frame carries `side`, which the availability report groups by.
    """
    frames, pooled = {}, {}
    for band in bands:
        for unit in UNITS:
            frames[(band, unit)] = build_peak_params_df(all_sessions,
                                                        side=unit, band=band)
        df_l, df_r = frames[(band, 'L')].copy(), frames[(band, 'R')].copy()
        df_l['side'], df_r['side'] = 'L', 'R'
        df = pd.concat([df_l, df_r], ignore_index=True)
        df['patient_side'] = df['patient'].astype(str) + '_' + df['side']
        pooled[band] = df
    return frames, pooled


def availability_report(pooled, bands):
    """Per band x condition x side x timepoint: rows with a usable channel,
    detections, and the count of bw values pinned to a PEAK_WIDTH_LIMITS bound.

    `available` counts rows present in the frame at all — build_peak_params_df
    emits a row only when the stim channel exists, so a row is a usable
    hemisphere whether or not a peak was found in it.
    """
    lo, hi = PEAK_WIDTH_LIMITS

    def _bw_at_bound(s):
        v = s.dropna().to_numpy()
        return int((np.isclose(v, lo) | np.isclose(v, hi)).sum())

    df = pd.concat([pooled[b] for b in bands], ignore_index=True)
    out = (df.groupby(['band', 'condition', 'side', 'timepoint'],
                      sort=False, observed=True)
             .agg(available=('detected', 'size'),
                  detected=('detected', 'sum'),
                  bw_at_bound=('bw', _bw_at_bound))
             .reset_index())
    out['detect_pct'] = (100 * out.detected / out.available.replace(0, np.nan)).round(0)
    return out


# ── Console summary ──────────────────────────────────────────────────────────
def print_summary(avail):
    lo, hi = PEAK_WIDTH_LIMITS
    print('\n' + '=' * 78)
    print('DETECTION / AVAILABILITY')
    print('=' * 78)
    tot = (avail.groupby(['band', 'timepoint'], sort=False, observed=True)[
        ['available', 'detected', 'bw_at_bound']].sum().reset_index())
    tot['detect_pct'] = (100 * tot.detected / tot.available.replace(0, np.nan)).round(1)
    print(tot.to_string(index=False))

    print('\n  per condition x side:')
    print(avail.to_string(index=False))

    n_bound, n_det = int(avail.bw_at_bound.sum()), int(avail.detected.sum())
    if n_bound:
        print(f'\n!! {n_bound}/{n_det} detected peaks have BW pinned to the FOOOF '
              f'bound ({lo:g} or {hi:g} Hz). Those are censored, not measured — '
              'treat every bw result as descriptive.')

    print('\n  ICC and LME for peak_beta_cf / _pw / _bw are run by')
    print('  icc_analysis.py and lme_analysis.py, in the same tables and the')
    print('  same FDR family as every other feature. Read them against the')
    print('  detection rates above: every one is conditional on detection.')


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bands', nargs='+', default=DEFAULT_BANDS)
    ap.add_argument('--out-dir', default=str(OUT_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print('Loading sessions …')
    all_sessions   = load_all_sessions()
    frames, pooled = build_frames(all_sessions, args.bands)

    avail = availability_report(pooled, args.bands)

    long_df = pd.concat(
        [f.assign(unit=unit) for (band, unit), f in frames.items() if not f.empty],
        ignore_index=True)
    long_df.to_csv(out_dir / 'peak_params_long.csv', index=False)
    avail.to_csv(out_dir / 'peak_params_availability.csv', index=False)

    print_summary(avail)
    print(f'\nCSVs → {out_dir}')


if __name__ == '__main__':
    main()
