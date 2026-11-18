"""
Absolute band power in the patient- and hemisphere-specific tremor bands.

Re-integrates the spectra that are ALREADY stored in the feature pickles, so
nothing upstream (preprocessing, epoching, FOOOF) is re-run:

    M3 / M6   Results/features/spect_feats_dict_{M3,M6}_badremoved.pkl
              'psd_abs_allch' (n_epochs, n_ch, n_freqs) amplitudes + 'freqs'
    home      Results/features/home_psds_dict.pkl
              'fft_bin_data'  (n_epochs, n_ch, n_freqs) amplitudes + 'freqs'

Both sit on the same ~0.977 Hz grid. Band power is the sum of |PSD|^2 over the
bins inside the band, using the same `freqs >= lo & freqs <= hi` rule as
spectral_analysis.frequency_analysis.extract_bandPower — the only difference is
that the edges vary per channel, so the summation is done channel by channel.

Because a 2 Hz-wide band holds only 2-3 bins on this grid, the bin count of
every cell is recorded alongside the power and written to a report, and the
per-cell count is carried into the stats frames.

Outputs:
    Results/stats/tremor/tremor_band_bins.csv

pipeline.py calls add_tremor_bands() on each feature dict before saving it, so
the tremor keys are already in the ordinary pickles. Running this script writes
no pickle of its own — it re-derives the same numbers from those files purely to
produce the bin-count report.

Run:  uv run python add_tremor_bands.py
"""

from __future__ import annotations

import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import BROADBAND_FREQ_RANGE  # noqa: E402
from power import _fix_negative_bins  # noqa: E402
from stats_analysis.data_utils import (  # noqa: E402
    CONDITIONS,
    EXCLUDED_PATIENTS,
    NB_PATIENTS,
    STIM_CHANNELS,
)
from tremor_bands import TREMOR_BAND_NAMES, TREMOR_BANDS  # noqa: E402

FEAT_DIR = ROOT / 'Results/features'
OUT_STATS = ROOT / 'Results/stats/tremor'

# timepoint -> (feature pickle, PSD key). pipeline.py already adds the tremor
# keys to these dicts before saving them, so this script no longer writes a
# *_tremor.pkl copy — running it re-derives the same numbers from the saved
# pickles purely to produce the bin-count QC report.
SOURCES = {
    'M3':   (FEAT_DIR / 'spect_feats_dict_M3_badremoved.pkl', 'psd_abs_allch'),
    'M6':   (FEAT_DIR / 'spect_feats_dict_M6_badremoved.pkl', 'psd_abs_allch'),
    'home': (FEAT_DIR / 'home_psds_dict.pkl', 'fft_bin_data'),
}

ABS_KEY = 'power_abs_tremor_allch'
ABS_EPOCH_KEY = 'power_abs_tremor_allch_allepochs'
BINS_KEY = 'tremor_band_nbins'
EDGES_KEY = 'tremor_band_edges'


def band_bins(freqs: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Same inclusive rule as extract_bandPower."""
    return np.logical_and(freqs >= lo, freqs <= hi)


def tremor_power_for_entry(entry: dict, patient_idx: int, psd_key: str,
                           fix_negatives: bool) -> list[dict]:
    """Adds the tremor keys to `entry` in place; returns per-cell bin records."""
    freqs = np.asarray(entry['freqs'], dtype=float)
    psd = np.asarray(entry[psd_key], dtype=float)          # (n_ep, n_ch, n_freq)
    if fix_negatives:
        psd = _fix_negative_bins(psd)
    power = psd ** 2                                        # amplitude -> power

    n_ep, n_ch, _ = power.shape
    n_bands = len(TREMOR_BAND_NAMES)
    out = np.full((n_ep, n_ch, n_bands), np.nan)
    nbins = np.full((n_ch, n_bands), np.nan)
    edges = np.full((n_ch, n_bands, 2), np.nan)
    records = []

    ch_names = list(entry['ch_names'])
    for side in ('L', 'R'):
        ch_name = STIM_CHANNELS.get(patient_idx, {}).get(side)
        if ch_name is None or ch_name not in ch_names:
            continue
        ch = ch_names.index(ch_name)
        bands = TREMOR_BANDS.get((patient_idx, side))
        if bands is None:
            continue
        for b, name in enumerate(TREMOR_BAND_NAMES):
            lo, hi = bands[name]
            idx = band_bins(freqs, lo, hi)
            out[:, ch, b] = power[:, ch, :][:, idx].sum(axis=1)
            nbins[ch, b] = int(idx.sum())
            edges[ch, b] = (lo, hi)
            records.append({'patient_idx': patient_idx, 'side': side,
                            'channel': ch_name, 'band': name,
                            'lo_hz': lo, 'hi_hz': hi,
                            'n_bins': int(idx.sum()),
                            'bin_hz': ', '.join(f'{f:.2f}' for f in freqs[idx]),
                            'n_epochs': n_ep})

    entry[ABS_EPOCH_KEY] = out
    with warnings.catch_warnings():
        # channels that are not a stimulation contact stay all-NaN by design
        warnings.simplefilter('ignore', RuntimeWarning)
        entry[ABS_KEY] = np.nanmean(out, axis=0) if n_ep else out
    entry[BINS_KEY] = nbins
    entry[EDGES_KEY] = edges
    return records


def add_tremor_bands(sdict: dict, psd_key: str, fix_negatives: bool,
                     timepoint: str = '', records: list | None = None) -> dict:
    """
    Add the tremor-band keys to every entry of one feature dict, in place.

    Factored out of main() so pipeline.py can call it on the dict it already
    holds in memory, once for the clinic session and once for home — the same
    shape as get_home_band_power / get_home_aperiodic_params. main() still uses
    it to process the pickles on disk, so the two routes cannot drift.

    Parameters
    ----------
    sdict : dict
        {condition: [entry or None per patient]}, mutated in place.
    psd_key : str
        'psd_abs_allch' for clinic, 'fft_bin_data' for home.
    fix_negatives : bool
        Home only: the device censors bins above ~58 Hz and stores them as
        negatives (see power._fix_negative_bins). Clinic PSDs are computed
        here and never censored.
    timepoint : str
        Stamped onto the returned bin records for the report.

    records : list, optional
        If given, the per-cell bin records (patient, side, band, edges,
        n_bins) are appended to it — that is what the QC report is built
        from. The dict itself is what callers want back.

    Returns
    -------
    dict
        `sdict`, mutated in place and returned so callers can assign it.
        No copy is made and no separate *_tremor.pkl is written: the tremor
        keys live in the same dict as every other feature.
    """
    if records is None:
        records = []
    for condition in CONDITIONS:
        if condition not in sdict:
            continue
        for i in range(NB_PATIENTS):
            if i in EXCLUDED_PATIENTS:
                continue
            entry = sdict[condition][i]
            if entry is None or entry.get(psd_key) is None:
                continue
            recs = tremor_power_for_entry(entry, i, psd_key,
                                          fix_negatives=fix_negatives)
            for r in recs:
                r.update({'timepoint': timepoint, 'condition': condition})
            records += recs
    return sdict


def main() -> int:
    OUT_STATS.mkdir(parents=True, exist_ok=True)
    all_records = []

    for tp, (src, psd_key) in SOURCES.items():
        if not src.exists():
            sys.exit(f'FATAL: missing input {src}')
        with open(src, 'rb') as fh:
            sdict = pickle.load(fh)

        recs: list[dict] = []
        add_tremor_bands(sdict, psd_key, fix_negatives=(tp == 'home'),
                         timepoint=tp, records=recs)
        all_records += recs
        n_done = len({(r['condition'], r['patient_idx']) for r in recs})
        print(f'{tp:>5}: {n_done} entries checked ({src.name})')

    bins = pd.DataFrame(all_records)
    bins.to_csv(OUT_STATS / 'tremor_band_bins.csv', index=False)

    print('\n' + '=' * 78)
    print('FREQUENCY RESOLUTION / BIN COUNT PER BAND')
    print('=' * 78)
    if bins.empty:
        print('  no cells computed')
        return 1
    print(f'  grid spacing: {bins.bin_hz.str.split(", ").str[0].nunique()} distinct '
          f'first-bin values across cells')
    print(bins.groupby('band')['n_bins']
          .agg(['min', 'max', 'mean', 'count']).round(2).to_string())
    print('\n  bins per (patient, side, band) — one row per unique band:')
    uniq = (bins.drop_duplicates(['patient_idx', 'side', 'band'])
                .sort_values(['patient_idx', 'side', 'band']))
    for r in uniq.itertuples():
        print(f'    P{r.patient_idx + 1:<3} {r.side}  {r.band:<10} '
              f'[{r.lo_hz:5.2f}, {r.hi_hz:5.2f}]  n_bins={r.n_bins}  '
              f'bins: {r.bin_hz}')
    zero = uniq[uniq.n_bins == 0]
    if len(zero):
        print(f'\n  WARNING: {len(zero)} band(s) contain no FFT bin at all')
    print(f'\n  broadband denominator for relative power: {BROADBAND_FREQ_RANGE} Hz '
          f'(unused — absolute power only)')
    print(f'\nWrote {OUT_STATS / "tremor_band_bins.csv"}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
