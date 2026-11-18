import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import APERIODIC_FREQ_RANGE, BROADBAND_FREQ_RANGE, FREQ_BANDS  # noqa: E402
from spectral_analysis.frequency_analysis import (  # noqa: E402
    extract_bandPower,
    extract_peak_aperiodic_params_fooof,
    extract_psd_welch,
)


def get_spectral_feat(epochs):
    """Spectral features of one recording (one patient x session x condition).

    Parameters
    ----------
    epochs : mne.Epochs
        Preprocessed 800 ms epochs at 250 Hz.

    Returns
    -------
    dict with the PSD, band power (absolute and relative), FOOOF aperiodic and
    periodic parameters, and the channel names.
    """
    bands = {k: np.array(v) for k, v in FREQ_BANDS.items()}
    data = epochs.get_data()

    # Concatenate the 800 ms epochs back into one continuous trace and run a
    # single Welch over all of it — one PSD per patient x session x condition x
    # channel. Welch already averages its segment periodograms in the power
    # domain, so no second averaging step is needed.
    #
    # The epochs are not contiguous in time (rejected segments were dropped
    # during preprocessing), so a 1 s window can straddle a splice: with
    # win_samples=250 and noverlap=200 (50-sample hop) each splice falls inside
    # five windows. Against the several hundred windows of a recording this is a
    # small fraction, and the Hann taper suppresses the step, but the result is
    # an approximation of the original trace, not a reconstruction.
    n_ep, n_ch, n_t = data.shape
    cont = data.transpose(1, 0, 2).reshape(n_ch, n_ep * n_t)   # (n_ch, n_samples)
    psd_abs_allch, _, psd_rel_allch, _, freqs = extract_psd_welch(
        cont, fs=250, n_fft=256, n_freq_bins=100, avg_epochs=False)

    # FOOOF expects power (µV²), not amplitude (µVp). Square first, then
    # average: mean(A²) is the mean power, mean(A)² underestimates it.
    psd_power_mean = np.mean(psd_abs_allch ** 2, axis=0)   # (n_ch, n_freqs)
    fooof_dict = extract_peak_aperiodic_params_fooof(psd_power_mean, freqs, APERIODIC_FREQ_RANGE)

    power_abs_allch, power_rel_allch = extract_bandPower(
        np.transpose(psd_abs_allch ** 2, (0, 2, 1)), freqs, bands, BROADBAND_FREQ_RANGE,
        epochs=True)

    return {'psd_abs_allch': psd_abs_allch,
            'psd_rel_allch': psd_rel_allch,
            'power_abs_allch': np.mean(power_abs_allch, axis=0),
            'power_rel_allch': np.mean(power_rel_allch, axis=0),
            'freqs': freqs,
            'ch_names': epochs.info['ch_names'],
            'aperiodic_exponent_allch': fooof_dict['aperiodic_exponent'],
            'aperiodic_offset_allch': fooof_dict['aperiodic_offset'],
            'aperiodic_broadband_power_allch': fooof_dict['aperiodic_broadband_power'],
            'flattened_power_abs_allch': fooof_dict['flattened_power_abs'],
            'aperiodic_fit_allch': fooof_dict['aperiodic_fit'],
            'peak_fit_allch': fooof_dict['peak_fit'],
            'error_allch': fooof_dict.get('error'),
            'r_squared_allch': fooof_dict.get('r_squared'),
            'n_peaks_allch': fooof_dict.get('n_peaks'),
            'band_peaks_allch': fooof_dict.get('band_peaks')}
