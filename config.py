"""
Central configuration for the ECOLOVIM analysis.
Define shared parameters here — do not redefine them in individual scripts.
"""

# ── Frequency bands ────────────────────────────────────────────────────────────
# band name → (f_low, f_high) in Hz

FREQ_BANDS = {
    'delta':  (1,   4),
    'theta':  (4,   8),
    'alpha':  (8,  13),
    'alpha1': (8,  10),
    'alpha2': (10, 13),
    'beta':   (13, 35),
    'beta1':  (13, 20),
    'beta2':  (20, 35),
    'gamma1': (35, 55),
    'gamma2': (55, 90),
}

BAND_NAMES  = list(FREQ_BANDS.keys())
BAND_LABELS = {name: f'{lo}–{hi} Hz' for name, (lo, hi) in FREQ_BANDS.items()}

# ── Bands excluded from statistical analysis ──────────────────────────────────
# Single source of truth for which bands enter a statistical model, and hence
# which tests enter a multiple-comparison family. Defined here rather than per
# module so that every analysis corrects over the same feature set.
#
#   alpha1/alpha2 partition alpha, beta1/beta2 partition beta — testing a
#   sub-band beside its parent re-tests the same signal, adds rows that cannot
#   move independently of the parent, and pads the FDR family that the parent
#   is corrected within.
#   gamma1 (35–55) and gamma2 (55–90) are dropped: the line noise notch and the
#   aperiodic fit range ([1, 40], see APERIODIC_FREQ_RANGE) leave them without
#   an interpretable periodic signal to test.
#
# Bands still computed and stored — this governs analysis only, not feature
# extraction, so plots and PSD figures may still show the excluded bands.
EXCLUDED_BANDS = {'alpha1', 'alpha2', 'beta1', 'beta2', 'gamma1', 'gamma2'}

# Ordered band set every statistical model runs on: delta, theta, alpha, beta.
ANALYSIS_BANDS = [b for b in BAND_NAMES if b not in EXCLUDED_BANDS]

# ── PSD frequency range ─────────────────────────────────────────────
PSD_FREQ_RANGE = [1, 90]  # Hz — for band power; M3, M6 recordings

# ── FOOOF / aperiodic fitting range ───────────────────────────────────────────
APERIODIC_FREQ_RANGE = [1, 40]   # Hz — set to home recording

BROADBAND_FREQ_RANGE = [1, 90]   # Hz — for broadband power

# ── Tremor frequency bands ────────────────────────────────────────────────────
# Individual tremor frequency f0 (Hz) per patient and hemisphere, measured on
# the contralateral tremulous limb. Unlike FREQ_BANDS these are not global:
# each patient (and often each side) has its own f0, so the tremor bands belong
# to a (patient, hemisphere) pair — see tremor_bands.py.
#
#   tremor_f0 : [f0 - TREMOR_HALF_WIDTH,    f0 + TREMOR_HALF_WIDTH]     fundamental
#   tremor_2f : [2*f0 - TREMOR_HALF_WIDTH,  2*f0 + TREMOR_HALF_WIDTH]   2nd harmonic
#
# PLACEHOLDER VALUES. The real per-patient f0 are clinical data and are not
# distributed; these fake values match the synthetic data written by
# make_synthetic_data.py. Replace them locally to analyse real recordings.
TREMOR_F0 = {
    'P01': {'L': 5.0, 'R': 5.5},
    'P02': {'L': 6.0, 'R': 6.0},
    'P03': {'L': 4.5, 'R': 5.0},
    'P04': {'L': 5.5, 'R': 5.5},
    'P05': {'L': 6.5, 'R': 6.0},
    'P06': {'L': 4.0, 'R': 4.5},
    'P07': {'L': 5.0, 'R': 5.0},
    'P08': {'L': 7.0, 'R': 6.5},
    'P09': {'L': 5.5, 'R': 6.0},
    'P10': {'L': 4.5, 'R': 4.5},
}
TREMOR_HALF_WIDTH = 1.0   # Hz
