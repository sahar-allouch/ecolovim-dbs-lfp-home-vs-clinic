import numpy as np
from spectral_analysis.frequency_analysis import extract_bandPower
from spectral_analysis.frequency_analysis import extract_peak_aperiodic_params_fooof
from config import FREQ_BANDS, BAND_NAMES, APERIODIC_FREQ_RANGE, BROADBAND_FREQ_RANGE

HOME_BANDS      = {k: np.array(v) for k, v in FREQ_BANDS.items()}
HOME_BAND_ORDER = BAND_NAMES   # fixed insertion order → row index


def _fix_negative_bins(psd):
    """
    Replace negative amplitude bins with the last valid (non-negative) value,
    scanning left-to-right along the frequency axis.

    Parameters
    ----------
    psd : ndarray, shape (n_epochs, n_ch, n_freqs)

    Returns
    -------
    psd_fixed : ndarray, same shape — copy with negatives replaced.
    """
    psd_fixed = psd.copy()
    n_epochs, n_ch, n_freqs = psd_fixed.shape
    for e in range(n_epochs):
        for ch in range(n_ch):
            last_valid = 0.0
            for f in range(n_freqs):
                if psd_fixed[e, ch, f] < 0:
                    psd_fixed[e, ch, f] = last_valid
                else:
                    last_valid = psd_fixed[e, ch, f]
    return psd_fixed


def get_home_band_power(home_psds_dict, nb_patients, psd_type='fft_bin_data'):
    """
    Compute band power for at-home PSDs using the same function (extract_bandPower)
    and the same band definitions as the M3/M6 pipeline (get_spectral_feat).

    Input
    -----
    home_psds_dict[condition][patient]['fft_bin_data'] : ndarray (n_epochs, n_ch, n_freqs)
        Percept on-chip FFT magnitudes (µVp).

    Output keys added to each non-None entry
    -----------------------------------------
    power_abs_allch : ndarray (n_ch, n_bands)   — absolute band power, avg over epochs
    power_rel_allch : ndarray (n_ch, n_bands)   — relative band power, avg over epochs
    power_abs_allch_allepochs,
    power_rel_allch_allepochs : ndarray (n_epochs, n_ch, n_bands) — before the
        epoch average. Band order is HOME_BAND_ORDER (= config.BAND_NAMES).
        Shapes match M3/M6 spect_feats_dict entries exactly.

    """
    for patient in range(nb_patients):
        for condition in home_psds_dict:
            entry = home_psds_dict[condition][patient]
            if entry is None:
                continue

            freqs   = np.array(entry['freqs'])
            psd_raw = _fix_negative_bins(np.array(entry[psd_type]))   # (n_epochs, n_ch, n_freqs)


            # ── Square first, then mean (mean power approach) ─────────────────
            # Square amplitudes (µVp → µVp²) per epoch, sum within each band,
            # and only then average over epochs: mean(A²) is the mean power,
            # mean(A)² underestimates it.
            # extract_bandPower wants the frequency axis in the middle for 3-D
            # input, hence the transpose to (n_epochs, n_freqs, n_ch).
            power_abs_allch, power_rel_allch = extract_bandPower(
                np.transpose(psd_raw**2, (0, 2, 1)), freqs, HOME_BANDS, BROADBAND_FREQ_RANGE, epochs=True)
            # returns (n_epochs, n_ch, n_bands) arrays

            entry['power_abs_allch'] = np.mean(power_abs_allch, axis=0)   # (n_ch, n_bands)
            entry['power_rel_allch'] = np.mean(power_rel_allch, axis=0)   # (n_ch, n_bands)

            entry['power_abs_allch_allepochs'] = power_abs_allch   # (n_epochs, n_ch, n_bands)
            entry['power_rel_allch_allepochs'] = power_rel_allch   # (n_epochs, n_ch, n_bands)

    return home_psds_dict


def get_home_aperiodic_params(home_psds_dict, nb_patients, psd_type=['fft_bin_data']):

    for patient in range(0, nb_patients):
        for psd_key in psd_type:
            for condition in home_psds_dict.keys():
                if home_psds_dict[condition][patient] is not None:
                    freqs = home_psds_dict[condition][patient]['freqs']
                    psd = _fix_negative_bins(np.array(home_psds_dict[condition][patient][psd_key]))
                    # Square before averaging: mean(A²) gives true mean power; mean(A)² underestimates it.
                    mean_psd_power = np.mean(psd ** 2, axis=0)   # (n_ch, n_freqs) — mean power
                    # mean_psd_power = _savgol_smooth_psd(mean_psd_power)
                    fooof_dict = extract_peak_aperiodic_params_fooof(mean_psd_power, freqs, APERIODIC_FREQ_RANGE)
                    home_psds_dict[condition][patient]['aperiodic_exponent_allch'] = fooof_dict['aperiodic_exponent']
                    home_psds_dict[condition][patient]['aperiodic_offset_allch'] = fooof_dict['aperiodic_offset']
                    home_psds_dict[condition][patient]['aperiodic_broadband_power_allch'] = fooof_dict['aperiodic_broadband_power']
                    home_psds_dict[condition][patient]['flattened_power_abs_allch'] = fooof_dict['flattened_power_abs']
                    # relative flattened power is no longer produced by the FOOOF helper
                    # (rel_ / flat_rel_ are excluded from the analysis), matching
                    # get_spectral_feat.py for the in-clinic sessions:
                    # home_psds_dict[condition][patient]['flattened_power_rel_allch'] = fooof_dict['flattened_power_rel']
                    home_psds_dict[condition][patient]['aperiodic_fit_allch'] = fooof_dict['aperiodic_fit']
                    home_psds_dict[condition][patient]['peak_fit_allch'] = fooof_dict['peak_fit']
                    home_psds_dict[condition][patient]['error_allch'] = fooof_dict['error']
                    home_psds_dict[condition][patient]['r_squared_allch'] = fooof_dict['r_squared']
                    home_psds_dict[condition][patient]['n_peaks_allch'] = fooof_dict['n_peaks']
                    home_psds_dict[condition][patient]['band_peaks_allch'] = fooof_dict['band_peaks']
    
    return home_psds_dict