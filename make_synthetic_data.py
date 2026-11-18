"""
Synthetic stand-in for the ECOLOVIM patient data, so the pipeline can be run
end to end without access to the real recordings.

The real data are not distributed. This script writes FAKE files with exactly
the layout the pipeline expects, at the paths it reads them from:

    Results/data-preprocessed/epochs_dict_M3_badremoved.pkl   clinic epochs, M3
    Results/data-preprocessed/epochs_dict_M6_badremoved.pkl   clinic epochs, M6
    Results/data-preprocessed/home_fft_dict.pkl               at-home device FFT snapshots
    Results/Tables/impedance_all_patients.csv                 contact impedances, M3 and M6

Then:

    uv run python make_synthetic_data.py
    uv run python run_all.py --group features stats figures

Every number is drawn from a seeded random generator. Nothing is derived from
patient data: the spectra are a textbook model (1/f^exponent aperiodic
component + Gaussian alpha, beta and tremor peaks), not a fit to any recording.
Results computed from these files are meaningless — they only show that the
code runs.

What the fake data mimic (layout only):
- 10 patients x 4 conditions (rest/mvt x stim OFF/ON).
- Clinic, stim OFF: 6 bipolar channels (L03, L13, L02, R03, R13, R02).
  Clinic, stim ON and home: the 2 sensing channels in STIM_CHANNELS.
- 800 ms epochs at 250 Hz, i.e. 200 samples, as MNE EpochsArray, in volts.
- Home: the device's 100-bin FFT (0-96.7 Hz, 250/256 Hz spacing), in µVp.
- Missing data where the analysis expects it: patient 3 (index 2) has M3 only,
  patient 4 (index 3) has no at-home stim-OFF recordings
  (EXCLUDED_PATIENTS / HOME_EXCLUDED_BY_COND in stats_analysis/data_utils.py).

Built-in effects, so the figures and models have something to show:
- beta power drops with stimulation ON and with movement,
- small random per-patient shifts between M3, home and M6.

Existing files are never overwritten unless --force is given, so running this
in a folder that holds the real data does nothing.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import mne
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import TREMOR_F0  # noqa: E402
from stats_analysis.data_utils import (  # noqa: E402
    CONDITIONS,
    EXCLUDED_PATIENTS,
    HOME_EXCLUDED_BY_COND,
    NB_PATIENTS,
    STIM_CHANNELS,
)

SEED = 20260101
SFREQ = 250.0
EPOCH_SAMPLES = 200                       # 800 ms
CLINIC_CHANNELS = ['L03', 'L13', 'L02', 'R03', 'R13', 'R02']
HOME_FREQS = np.round(np.arange(100) * SFREQ / 256, 2)   # device grid, 0-96.68 Hz

PREPROC_DIR = ROOT / 'Results' / 'data-preprocessed'
OUT_EPOCHS = {tp: PREPROC_DIR / f'epochs_dict_{tp}_badremoved.pkl' for tp in ('M3', 'M6')}
OUT_HOME = PREPROC_DIR / 'home_fft_dict.pkl'
OUT_IMPEDANCE = ROOT / 'Results' / 'Tables' / 'impedance_all_patients.csv'

# SenSight lead: ring 0, three segments on levels 1 and 2, ring 3.
LEAD_CONTACTS = ['0', '1a', '1b', '1c', '2a', '2b', '2c', '3']


# ── Spectral model ────────────────────────────────────────────────────────────

def _gauss(f, cf, amp, sd):
    return amp * np.exp(-0.5 * ((f - cf) / sd) ** 2)


def model_psd(f, p):
    """Power spectrum (arbitrary units) for one parameter dict."""
    f = np.maximum(f, 0.5)                # no singularity at DC
    ap = 10 ** p['offset'] / f ** p['exponent']
    return ap * (1
                 + _gauss(f, 10.0, p['alpha'], 1.5)
                 + _gauss(f, p['beta_cf'], p['beta'], 3.0)
                 + _gauss(f, p['f0'], p['tremor'], 0.7)
                 + _gauss(f, 2 * p['f0'], p['tremor'] / 3, 0.7))


def draw_patients(rng):
    """Per patient x hemisphere baseline parameters; tremor f0 from config.TREMOR_F0."""
    base = {}
    for i in range(NB_PATIENTS):
        for side in ('L', 'R'):
            base[(i, side)] = dict(
                offset=rng.normal(0.0, 0.15),
                exponent=rng.normal(1.5, 0.15),
                alpha=rng.uniform(0.2, 0.8),
                beta_cf=rng.uniform(15, 28),
                beta=rng.uniform(1.0, 3.0),
                f0=TREMOR_F0[f'P{i + 1:02d}'][side],    # same f0 the analysis uses
                tremor=rng.uniform(0.3, 1.5),
            )
    return base


def condition_params(p, condition, tp_shift, rng):
    """Apply the stim / movement / timepoint effects to one baseline."""
    q = dict(p)
    stim_on = condition.endswith('stimON')
    mvt = condition.startswith('mvt')
    q['beta'] *= (0.6 if stim_on else 1.0) * (0.75 if mvt else 1.0) * tp_shift['beta']
    q['tremor'] *= (0.4 if stim_on else 1.0) * (1.5 if mvt else 1.0)
    q['exponent'] += (0.1 if stim_on else 0.0) + tp_shift['exponent']
    q['offset'] += tp_shift['offset'] + rng.normal(0, 0.03)
    return q


def channel_side(ch):
    return ch[0]


# ── Clinic epochs ─────────────────────────────────────────────────────────────

def synth_timeseries(psd_params, n_samples, rng):
    """Gaussian noise shaped to the model spectrum, scaled to ~8 µV RMS."""
    freqs = np.fft.rfftfreq(n_samples, 1 / SFREQ)
    amp = np.sqrt(model_psd(freqs, psd_params))
    spec = (rng.standard_normal(freqs.size) + 1j * rng.standard_normal(freqs.size)) * amp
    spec[0] = 0.0
    x = np.fft.irfft(spec, n=n_samples)
    return x / x.std() * 8e-6 * 10 ** (psd_params['offset'] / 2)


def make_epochs(channels, params_by_side, n_epochs, rng):
    n = n_epochs * EPOCH_SAMPLES
    data = np.stack([synth_timeseries(params_by_side[channel_side(ch)], n, rng)
                     for ch in channels])                     # (n_ch, n_samples)
    data = data.reshape(len(channels), n_epochs, EPOCH_SAMPLES).transpose(1, 0, 2)
    info = mne.create_info(channels, SFREQ, ch_types='dbs')
    events = np.column_stack([np.arange(n_epochs) * EPOCH_SAMPLES,
                              np.zeros(n_epochs, int), np.ones(n_epochs, int)])
    return mne.EpochsArray(data, info, events=events, event_id={'1': 1},
                           tmin=0.0, verbose=False)


def make_clinic(tp, base, shifts, rng):
    epochs_dict = {c: [] for c in CONDITIONS}
    for cond in CONDITIONS:
        for i in range(NB_PATIENTS):
            if tp == 'M6' and i in EXCLUDED_PATIENTS:     # withdrew after M3
                epochs_dict[cond].append(None)
                continue
            params = {s: condition_params(base[(i, s)], cond, shifts[(i, s, tp)], rng)
                      for s in ('L', 'R')}
            channels = (CLINIC_CHANNELS if cond.endswith('stimOFF')
                        else [STIM_CHANNELS[i]['L'], STIM_CHANNELS[i]['R']])
            n_epochs = int(rng.integers(40, 260))
            epochs_dict[cond].append(make_epochs(channels, params, n_epochs, rng))
    return epochs_dict


# ── At-home device FFT ────────────────────────────────────────────────────────

def make_home(base, shifts, rng):
    home = {c: [] for c in CONDITIONS}
    for cond in CONDITIONS:
        for i in range(NB_PATIENTS):
            if i in EXCLUDED_PATIENTS or i in HOME_EXCLUDED_BY_COND.get(cond, set()):
                home[cond].append(None)
                continue
            channels = [STIM_CHANNELS[i]['L'], STIM_CHANNELS[i]['R']]
            n_snap = int(rng.integers(8, 40))
            snaps = np.empty((n_snap, 2, HOME_FREQS.size))
            for c, ch in enumerate(channels):
                q = condition_params(base[(i, channel_side(ch))], cond,
                                     shifts[(i, channel_side(ch), 'home')], rng)
                amp = np.sqrt(model_psd(HOME_FREQS, q))
                amp *= 7.6 / amp[1]           # ~7.6 µVp at 1 Hz: matches clinic delta power
                # each snapshot is an average of a few periodograms -> chi noise
                noise = np.sqrt(rng.chisquare(8, size=(n_snap, HOME_FREQS.size)) / 8)
                snaps[:, c, :] = amp * noise
            home[cond].append({'fft_bin_data': snaps,
                               'ch_names': channels,
                               'freqs': HOME_FREQS.copy()})
    return home


# ── Impedance table ───────────────────────────────────────────────────────────

def make_impedance_table(rng):
    """Same columns as the real table extracted from the session JSONs. One impedance session
    per patient x timepoint (none at M6 for EXCLUDED_PATIENTS); monopolar = each
    contact vs the case, bipolar = every contact pair. Built-in effect: a small
    rise from M3 to M6. Dates and file names are placeholders."""
    rows = []
    for i in range(NB_PATIENTS):
        for hemi in ('Left', 'Right'):
            lead = rng.normal(3200, 600)                       # per-lead level, Ω
            contact = {c: lead * rng.lognormal(0, 0.25) for c in LEAD_CONTACTS}
            for tp in ('M3', 'M6'):
                if tp == 'M6' and i in EXCLUDED_PATIENTS:
                    continue
                drift = 1.0 if tp == 'M3' else rng.lognormal(0.05, 0.08)
                common = dict(Patient=i + 1, Timepoint=tp,
                              SessionDate='2000-01-01T00:00:00Z',
                              File=f'synthetic_P{i + 1}_{tp}.json',
                              TestCurrentMA='1.0', Hemisphere=hemi)
                mono = {c: contact[c] * drift * rng.lognormal(0, 0.03)
                        for c in LEAD_CONTACTS}
                for c, val in mono.items():
                    rows.append({**common, 'ImpedanceStatus': 'GOOD',
                                 'Mode': 'Monopolar', 'Electrode1': 'Case',
                                 'Electrode2': f'SenSight_{c}',
                                 'Impedance_Ohm': round(val)})
                for a in range(len(LEAD_CONTACTS)):
                    for b in range(a + 1, len(LEAD_CONTACTS)):
                        ca, cb = LEAD_CONTACTS[a], LEAD_CONTACTS[b]
                        rows.append({**common, 'ImpedanceStatus': 'GOOD',
                                     'Mode': 'Bipolar',
                                     'Electrode1': f'SenSight_{ca}',
                                     'Electrode2': f'SenSight_{cb}',
                                     'Impedance_Ohm': round(mono[ca] + mono[cb])})
    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def _writable(path, force):
    if path.exists() and not force:
        print(f'skip   {path.relative_to(ROOT)} (exists; --force to overwrite)')
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--force', action='store_true',
                    help='overwrite existing files (destroys real data if present)')
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    base = draw_patients(rng)
    shifts = {(i, s, tp): dict(beta=rng.lognormal(0, 0.15),
                               exponent=rng.normal(0, 0.05),
                               offset=rng.normal(0, 0.05))
              for i in range(NB_PATIENTS) for s in ('L', 'R')
              for tp in ('M3', 'home', 'M6')}

    # Generate everything before writing anything, so the random stream (and
    # therefore every file) is the same whichever files get skipped.
    pickles = {OUT_EPOCHS[tp]: make_clinic(tp, base, shifts, rng) for tp in OUT_EPOCHS}
    pickles[OUT_HOME] = make_home(base, shifts, rng)

    for path, obj in pickles.items():
        if _writable(path, args.force):
            with open(path, 'wb') as f:
                pickle.dump(obj, f)
            print(f'wrote  {path.relative_to(ROOT)}')

    imp = make_impedance_table(rng)
    if _writable(OUT_IMPEDANCE, args.force):
        imp.to_csv(OUT_IMPEDANCE, index=False)
        print(f'wrote  {OUT_IMPEDANCE.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
