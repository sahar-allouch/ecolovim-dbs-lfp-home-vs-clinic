"""
Spectral features for one recording session.

    ECOLOVIM_SESSION=M3   python pipeline.py    # in-clinic, 3 months
    ECOLOVIM_SESSION=M6   python pipeline.py    # in-clinic, 6 months
    ECOLOVIM_SESSION=home python pipeline.py    # at home

The session comes from the environment so that run_all.py can run the same
script once per session.

In-clinic (M3, M6)
    reads   Results/data-preprocessed/epochs_dict_{session}_badremoved.pkl
            {condition: [one mne.Epochs per patient, or None]}
    writes  Results/features/spect_feats_dict_{session}_badremoved.pkl
    Welch PSD over the concatenated epochs, band power, FOOOF, tremor bands.

At home
    reads   Results/data-preprocessed/home_fft_dict.pkl
            {condition: [per patient: None or
             {'fft_bin_data': (n_snapshots, n_ch, 100), 'ch_names', 'freqs'}]}
    writes  Results/features/home_psds_dict.pkl
    The device computed these spectra itself, so there is no epoching and no
    Welch: band power and FOOOF are computed on the stored FFT bins.

The input pickles are written by make_synthetic_data.py when the real
recordings are not available.
"""

import os
import pickle

from add_tremor_bands import add_tremor_bands
from power import get_home_aperiodic_params, get_home_band_power
from spectral_analysis.get_spectral_feat import get_spectral_feat
from stats_analysis.data_utils import NB_PATIENTS

session = os.environ.get('ECOLOVIM_SESSION', 'M3')   # 'M3', 'M6' or 'home'

PREPROC_DIR = 'Results/data-preprocessed'
FEAT_DIR = 'Results/features'
os.makedirs(FEAT_DIR, exist_ok=True)

if session in ('M3', 'M6'):
    with open(f'{PREPROC_DIR}/epochs_dict_{session}_badremoved.pkl', 'rb') as f:
        epochs_dict = pickle.load(f)

    spect_feats_dict = {
        condition: [get_spectral_feat(ep) if ep is not None else None for ep in per_patient]
        for condition, per_patient in epochs_dict.items()
    }

    # Tremor bands — summed from the PSDs just computed over the patient- and
    # hemisphere-specific windows, so the saved pickle already carries them.
    # Clinic PSDs are never censored, hence fix_negatives=False.
    spect_feats_dict = add_tremor_bands(spect_feats_dict, 'psd_abs_allch',
                                        fix_negatives=False, timepoint=session)

    with open(f'{FEAT_DIR}/spect_feats_dict_{session}_badremoved.pkl', 'wb') as f:
        pickle.dump(spect_feats_dict, f)

else:
    with open(f'{PREPROC_DIR}/home_fft_dict.pkl', 'rb') as f:
        home_psds_dict = pickle.load(f)

    home_psds_dict = get_home_band_power(home_psds_dict, NB_PATIENTS)
    home_psds_dict = get_home_aperiodic_params(home_psds_dict, NB_PATIENTS)

    # The device censors bins above ~58 Hz in the at-home snapshots and stores
    # them as negative values; fix_negatives=True replaces them before the
    # band sum (the tremor windows sit far below that frequency).
    home_psds_dict = add_tremor_bands(home_psds_dict, 'fft_bin_data',
                                      fix_negatives=True, timepoint='home')

    with open(f'{FEAT_DIR}/home_psds_dict.pkl', 'wb') as f:
        pickle.dump(home_psds_dict, f)
