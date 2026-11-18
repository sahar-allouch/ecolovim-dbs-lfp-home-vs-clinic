import sys
import os
import numpy as np
from copy import deepcopy
from fooof import FOOOF
from fooof.data import FOOOFSettings
from fooof.bands import Bands
from fooof.analysis import get_band_peak_fm
from scipy.interpolate import interp1d
from scipy.signal import welch as _scipy_welch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import FREQ_BANDS


def extract_psd_welch(data, fs=250, win_samples=250, noverlap=200, n_fft=256,
                      n_freq_bins=100, avg_epochs=True, scale=1.695):
    """
    Welch PSD per input block, returned as amplitude in µVp — the unit of the
    spectra the Percept device stores.

    Per block this is the MATLAB call

        [pxx, f] = pwelch(signal, 250, 200, 256, 250)

    i.e. 1 s Hann window, 80 % overlap, 256-point FFT, averaged over all the
    segments of the block. `get_spectral_feat` passes ONE block per
    patient x session x condition (every retained 800 ms epoch concatenated
    into a single trace), so the block axis has length 1.

    n_fft = 256 puts the bins on the device's own fs/256 ≈ 0.977 Hz grid, the
    grid of the at-home spectra, so clinic and home spectra can be compared bin
    for bin without interpolation. The grid is not rounded: rounding to two
    decimals makes consecutive steps alternate between 0.97 and 0.98 Hz.

    µV²/Hz → µVp uses Medtronic's factor applied to the amplitude spectral
    density, with no bin-width term:

        mag = sqrt(pxx) * 1.695,
        1.695 = RMS(Hanning250) * sqrt(2) * 2 * 250/256

    Multiplying by sqrt(df) as well would make the result depend on n_fft,
    which a density must not. This scaling was checked against recordings that
    carry both a raw trace and the device's own PSD, and reproduces the stored
    values.

    Changing n_fft moves the frequency grid and the calibration: `freqs` is
    n_freq_bins * fs/n_fft wide, and `mag` scales as sqrt(fs/n_fft).

    `detrend=False` matches MATLAB's pwelch, which does not remove the segment
    mean; scipy defaults to doing so.

    Parameters
    ----------
    data : ndarray, shape (n_blocks, n_ch, n_times) or (n_ch, n_times)
        Time-domain signal in V (MNE convention).
    win_samples, noverlap, n_fft : int
        Welch segment length, overlap and FFT size, in samples.
    avg_epochs : bool
        Average across the leading (block) axis before returning.

    Returns
    -------
    psd_abs_allch, psd_abs_avgch, psd_rel_allch, psd_rel_avgch, freqs
    """
    data = np.array(data, dtype=float)
    if data.ndim == 2:
        data = data[np.newaxis, :]
    data = data * 1e6                       # V → µV, so the output is µVp

    n_blocks, n_ch, n_times = data.shape

    # A block shorter than one window cannot be Welch-averaged; scipy would
    # silently shrink nperseg, which would change the calibration. Pad instead.
    nperseg = min(win_samples, n_times) if n_times < win_samples else win_samples
    if n_times < win_samples:
        pad = win_samples - n_times
        data = np.pad(data, ((0, 0), (0, 0), (0, pad)))
        nperseg = win_samples
        n_times = data.shape[2]

    magnitude = np.zeros((n_blocks, n_ch, n_freq_bins))
    for b in range(n_blocks):
        freqs_full, pxx = _scipy_welch(
            data[b], fs=fs, window='hann', nperseg=nperseg,
            noverlap=min(noverlap, nperseg - 1), nfft=n_fft,
            scaling='density', detrend=False, axis=-1,
        )                                   # pxx: (n_ch, n_fft//2 + 1), µV²/Hz
        magnitude[b] = (np.sqrt(pxx) * scale)[:, :n_freq_bins]

    freqs = freqs_full[:n_freq_bins]        # not rounded — see docstring

    psd_abs_allch = magnitude.mean(axis=0) if avg_epochs else magnitude
    psd_abs_avgch = psd_abs_allch.mean(axis=-2)

    total = psd_abs_allch.sum(axis=-1, keepdims=True)
    total = np.where(total == 0, np.finfo(float).eps, total)
    psd_rel_allch = psd_abs_allch / total
    psd_rel_avgch = psd_rel_allch.mean(axis=-2)

    return psd_abs_allch, psd_abs_avgch, psd_rel_allch, psd_rel_avgch, freqs


def extract_bandPower(psd, freqs, bf_all, broadband_freq_range=None, output_dict=False, epochs=False, rel=True):

    """Extract powers of the computed PSD (power + relative power) as specified in bf_all

    Parameters
    ----------
    computed_psd : ndarray, shape=(n_freqs) or (n_freqs, n_chan)
        The computed PSD from which we want to extract the power.
    freq_range : list
        The frequency range of data, ex: [3, 45].
    bf_all : dict
        The dictionary that specifies the frequency of interests (keys: str, names of freq range ; values: list).


    Returns
    -------
    power_all : dict
        The extracted power of each freq band (keys: str, names of freq range ; values: value if psd is 1d
        or (nb_chan) if psd is 2d)
    rel_power_all : dict
        The extracted relative power of each freq band (keys: str, names of freq range ; values: value if psd is 1d
        or (nb_chan) if psd is 2d)
    """

#     # Initialize the dictionaries (power, relative power)
#     band_power = {}
#     rel_band_power = {}

#     power_total = np.sum(psd, 0)

#     # Compute 'power' then 'relative power' for each subj in each band for the specified psd input
#     for bf, val in bf_all.items():

#         # Power
#         idx = np.logical_and(freqs >= val[0], freqs <= val[1])
#         band_power[bf] = np.sum(psd[idx], axis = 0)
#         rel_band_power[bf] = band_power[bf] / power_total #!!!!!!!!!!!!!!!!!


#     if not output_dict:
#         band_power = np.array(list(band_power.values())) # DO WE NEED np ARRAY OR JUST LIST?
#         rel_band_power = np.array(list(rel_band_power.values()))

#     return band_power, rel_band_power
    
    band_power = {}
    rel_band_power = {}

    # psd is 3d (n_epochs, n_freqs, n_chan) instead of 2d (n_freqs, nchan)

    freq_axis = 1 if epochs else 0
    if broadband_freq_range is None:
        power_total = np.sum(psd, axis=freq_axis)
    else:
        idx_broad = np.logical_and(freqs >= broadband_freq_range[0], freqs <= broadband_freq_range[1])
        # index along the freq axis regardless of ndim
        slices = (slice(None),) * freq_axis + (idx_broad,)
        power_total = np.sum(psd[slices], axis=freq_axis)

    # Compute 'power' then 'relative power' for each subj in each band for the specified psd input
    for bf, val in bf_all.items():
        # Power
        idx = np.logical_and(freqs >= val[0], freqs <= val[1])

        # if psd.ndim == 2:
        #     band_power[bf] = np.sum(psd[idx], axis = 0)
        # elif psd.ndim == 3:
        #     band_power[bf] = np.sum(psd[:, idx], axis = 1)

        axis = 0 if psd.ndim == 2 else 1
        band_power[bf] = np.sum(psd[(slice(None),) * (psd.ndim - 2) + (idx,)], axis=axis)

        if rel:
            rel_band_power[bf] = band_power[bf] / power_total

    if not output_dict:
        if psd.ndim == 2:
            # pass
            band_power = np.array(list(band_power.values()))

            if rel:
                rel_band_power = np.array(list(rel_band_power.values()))
        
        elif psd.ndim == 3:
            band_power = np.transpose(np.array(list(band_power.values())), [1, 2, 0])

            if rel:
                rel_band_power = np.transpose(np.array(list(rel_band_power.values())), [1, 2, 0])

    if rel:
        return band_power, rel_band_power

    return band_power, None  # Return None for rel_band_power if rel is False


def extract_peak_aperiodic_params_fooof(psds, freqs, freq_range):
    
    if freq_range[0] > freqs[0]:
        # crop psds and freqs to match freq_range
        idx = np.where(freqs >= freq_range[0])[0]
        freqs = freqs[idx]
        psds = psds[:, :, idx] if psds.ndim == 3 else psds[:, idx] if psds.ndim == 2 else psds[idx]


    if freq_range[1] < freqs[-1]:
        # crop psds and freqs to match freq_range
        idx = np.where(freqs <= freq_range[1])[0]
        freqs = freqs[idx]
        psds = psds[:, :, idx] if psds.ndim == 3 else psds[:, idx] if psds.ndim == 2 else psds[idx]


    if psds.ndim == 2:
        psds = psds[np.newaxis, :]  # Add a dimension for epochs

    assert psds.ndim == 3, f"psds must be 3D (n_epochs, n_chans, n_freqs), got shape {psds.shape}"

    nb_epochs, nb_chans, nb_freqs = psds.shape

    if True:  # block kept for indentation; psds is guaranteed 3D by assertion above
        # NaN-initialised so failed FOOOF fits are distinguishable from a real zero value.
        exponents  = np.full((nb_epochs, nb_chans), np.nan)
        offsets    = np.full((nb_epochs, nb_chans), np.nan)
        error      = np.full((nb_epochs, nb_chans), np.nan)
        r_squared  = np.full((nb_epochs, nb_chans), np.nan)

        # --- variable-size peak params ---
        # get_params('peak_params') → (N_total_peaks, 4): [CF, PW, BW, spectrum_idx]
        # spectrum_idx runs 0 … nb_epochs*nb_chans-1; we map back to (epoch, chan)
        peak_central_freq = [[None for _ in range(nb_chans)] for _ in range(nb_epochs)]
        peak_power        = [[None for _ in range(nb_chans)] for _ in range(nb_epochs)]
        peak_bandwidth    = [[None for _ in range(nb_chans)] for _ in range(nb_epochs)]
        n_peaks           = np.full((nb_epochs, nb_chans), np.nan)

        # --- band peaks — from central config ---
        freq_bands = {k: list(v) for k, v in FREQ_BANDS.items()}
        band_defs = Bands(freq_bands)

        # band_names = ['delta', 'theta', 'alpha', 'alpha1', 'alpha2', 'beta', 'gamma1', 'gamma2']
        band_names = list(freq_bands.keys())

        peaks         = {b: np.full((nb_epochs, nb_chans, 3), np.nan) for b in band_names}
        band_peaks_cfs = {b: np.full((nb_epochs, nb_chans), np.nan)   for b in band_names}
        band_peaks_pws = {b: np.full((nb_epochs, nb_chans), np.nan)   for b in band_names}
        band_peaks_bws = {b: np.full((nb_epochs, nb_chans), np.nan)   for b in band_names}


        # NaN-initialised: bins not covered by FOOOF (masked or out of range) stay NaN
        # rather than 0, which would be misread as 10^0 = 1 when exponentiated.
        aperiodic_fit        = np.full((nb_epochs, nb_chans, nb_freqs), np.nan)
        peak_fit             = np.full((nb_epochs, nb_chans, nb_freqs), np.nan)
        flattened_spectra    = np.full((nb_epochs, nb_chans, nb_freqs), np.nan)
        # peak_removed_spectra = np.full((nb_epochs, nb_chans, nb_freqs), np.nan)

        full_spectrum        = np.full((nb_epochs, nb_chans, nb_freqs), np.nan)

        freqs_fit            = np.full((nb_epochs, nb_chans, nb_freqs), np.nan)
        aperiodic_broadband_power = np.zeros((nb_epochs, nb_chans))

        flattened_power_abs = np.zeros((nb_epochs, nb_chans, len(freq_bands)))
        # flattened_power_rel = np.zeros((nb_epochs, nb_chans, len(freq_bands)))

        # FOOOF settings
        PEAK_WIDTH_LIMITS = [2, 12]   # Hz
        MAX_N_PEAKS = 3
        MIN_PEAK_HEIGHT = 0
        PEAK_THRESHOLD = 1.5
        APERIODIC_MODE = 'fixed'

        fooof_settings = FOOOFSettings(peak_width_limits=PEAK_WIDTH_LIMITS, max_n_peaks=MAX_N_PEAKS,
                                    min_peak_height=MIN_PEAK_HEIGHT, peak_threshold=PEAK_THRESHOLD,
                                    aperiodic_mode=APERIODIC_MODE)
        
        # fooof initialization
        fm = FOOOF(*fooof_settings, verbose=False)

        fooof_obj_list = [[None for _ in range(nb_chans)] for _ in range(nb_epochs)]

        for i_epoch in range(nb_epochs):
            for i_chan in range(nb_chans):
                psd_chan = psds[i_epoch, i_chan, :]
          
                # freqs_fit, psd_fit = ensure_uniform_freqs(freqs, psd_chan)

                freqs_uni, psd_chan_uni = ensure_uniform_freqs(freqs, psd_chan)

                # Raise an error if any PSD bins are zero - debugging for FOOOF fitting issues
                mask = psd_chan_uni == 0
                if np.any(mask):
                    raise ValueError(f"Zero-valued PSD bins found for epoch {i_epoch}, channel {i_chan}.")

                # try:

                fm.fit(freqs_uni, psd_chan_uni, freq_range)
                # fm.report(freqs_fit, psd_fit, freq_range)

                # Fixed-size outputs
                
                exponents[i_epoch, i_chan] = fm.get_params('aperiodic_params', 'exponent')
                offsets[i_epoch, i_chan] = fm.get_params('aperiodic_params', 'offset')
                error[i_epoch, i_chan] = fm.get_params('error')
                r_squared[i_epoch, i_chan] = fm.get_params('r_squared')
                n_peaks[i_epoch, i_chan] = fm.get_params('n_peaks')

                # Variable-size output
                tmp = fm.get_params('peak_params')
                
                _valid = tmp is not None and tmp.size > 0 and not np.all(np.isnan(tmp))
                
                peak_central_freq[i_epoch][i_chan] = tmp[:, 0] if _valid else np.array([])
                peak_power[i_epoch][i_chan]        = tmp[:, 1] if _valid else np.array([])
                peak_bandwidth[i_epoch][i_chan]    = tmp[:, 2] if _valid else np.array([])

                # Band peaks
                for b in band_names:
                    bp = get_band_peak_fm(fm, getattr(band_defs, b))  # (3,) or None
                    if bp is not None:
                        peaks[b][i_epoch, i_chan, :] = bp
                        band_peaks_cfs[b][i_epoch, i_chan] = bp[0]
                        band_peaks_pws[b][i_epoch, i_chan] = bp[1]
                        band_peaks_bws[b][i_epoch, i_chan] = bp[2]

                # Flattened & peak-removed spectra

                aperiodic_fit[i_epoch, i_chan, :]        = fm._ap_fit
                peak_fit[i_epoch, i_chan, :]             = fm._peak_fit
                full_spectrum[i_epoch, i_chan, :]       = fm.power_spectrum
                flattened_spectra[i_epoch, i_chan, :]    = fm.power_spectrum - fm._ap_fit
                # peak_removed_spectra[i_epoch, i_chan, :] = fm.power_spectrum - fm._peak_fit
                freqs_fit[i_epoch, i_chan] = fm.freqs
                # peak_fit[i_epoch, i_chan, :] = fm._peak_fit
                # flattened_spectra[i_epoch, i_chan, :]    = fm.power_spectrum - fm._ap_fit
                # peak_removed_spectra[i_epoch, i_chan, :] = fm.power_spectrum - fm._peak_fit

                # deepcopy so each entry holds its own fit, not a shared reference
                fooof_obj_list[i_epoch][i_chan] = deepcopy(fm)

                # except Exception as e:
                #             print(f"Error processing epoch {i_epoch}, channel {i_chan}: {e}")
                            

        # aperiodic_fit is in log10 space — exponentiate then nansum to ignore unfit bins.
        # Return NaN (not 0) for channels where the fit failed entirely (all bins are NaN).
        _ap_linear = 10 ** aperiodic_fit
        _all_nan   = np.all(np.isnan(aperiodic_fit), axis=2)
        aperiodic_broadband_power = np.where(_all_nan, np.nan, np.nansum(_ap_linear, axis=2))
        # aperiodic_broadband_power = np.sum(peak_removed_spectra, axis=2)
        # (0, 2, 1)), freqs, bf_all, epochs=True)
        # flattened_spectra is in log10 space (fm.power_spectrum - fm._ap_fit).
        # Exponentiate to linear, then zero out NaN bins (unfit/masked) before summing.
        # flattened_linear = np.nan_to_num(10 ** flattened_spectra, nan=0.0)
        # Periodic power (µV²): PSD − aperiodic, both in linear space.
        # Failed-fit channels have NaN aperiodic_fit → NaN here → zero them so
        # extract_bandPower's np.sum doesn't blank the whole band.

        # flattened_linear = np.nan_to_num(10 ** full_spectrum - _ap_linear, nan=0.0)
        
        # check if freqs_fit is the same for all epochs/channels
        axes = freqs_fit.reshape(-1, nb_freqs)      # (nb_ep*nb_ch, nb_freqs)
        uniq = np.unique(axes, axis=0)              # unique AXES (rows)
        if len(uniq) > 1:
            raise ValueError("Frequency axes differ across epochs/channels.")
        common_freqs = uniq[0]                      # the shared axis (an array)
        # flattened_power_abs, _ = extract_bandPower(np.transpose(flattened_linear, (0, 2, 1)), common_freqs, freq_bands, epochs=True, rel=False)
        flattened_power_abs, _ = extract_bandPower(np.transpose(flattened_spectra, (0, 2, 1)), common_freqs, freq_bands, epochs=True, rel=False)

            
    fooof_dict = {
    'aperiodic_exponent': exponents,
        'aperiodic_offset': offsets,
        'error': error,
        'r_squared': r_squared,
        'n_peaks': n_peaks,
        'peak_central_freq': peak_central_freq,
        'peak_power': peak_power,
        'peak_bandwidth': peak_bandwidth,
        'band_peaks': peaks,
        # **{f'{b}_peaks':     peaks[b]     for b in band_names},
        # **{f'{b}_peaks_cfs': band_peaks_cfs[b] for b in band_names},
        # **{f'{b}_peaks_pws': band_peaks_pws[b] for b in band_names},
        # **{f'{b}_peaks_bws': band_peaks_bws[b] for b in band_names},
        # freqs_fit is not stored here — it varies per epoch/channel.
        # Access it per-epoch/channel via fooof_objects[i_epoch][i_chan].freqs.
        'aperiodic_fit': aperiodic_fit,
        'peak_fit': peak_fit,
        # 'flattened_spectra': flattened_spectra,
        # 'peak_removed_spectra': peak_removed_spectra,
        'aperiodic_broadband_power': aperiodic_broadband_power,
        'flattened_power_abs': flattened_power_abs,
        # 'flattened_power_rel': flattened_power_rel,
        'fooof_objects': fooof_obj_list
    }

    return fooof_dict


def ensure_uniform_freqs(freqs, psd):
    """Interpolate PSD onto a uniform frequency grid if needed."""
    step = freqs[1] - freqs[0]
    if np.allclose(np.diff(freqs), step):
        return freqs, psd  # already uniform, no-op

    if np.isclose(np.min(np.diff(freqs)), 0.97) and np.isclose(np.max(np.diff(freqs)), 0.98):
        freqs_uniform = np.arange(freqs[0], freqs[-1] + step/2, step)
        interpolator  = interp1d(freqs, psd, kind='linear', bounds_error=False,
                                fill_value='extrapolate')
        psd_uniform   = interpolator(freqs_uniform)
        return freqs_uniform, psd_uniform

    raise ValueError("Frequency difference is different than expected (!= [0.97, 0.98]).")

