"""
Patient- and hemisphere-specific tremor frequency bands.

Source: TREMOR_F0 and TREMOR_HALF_WIDTH in config.py.

Unlike config.FREQ_BANDS, these edges are NOT global: each patient has an
individual tremor frequency f0, which can differ between the left and the
right electrode (f0 is taken from the contralateral tremulous limb). A band
therefore belongs to a (patient, hemisphere) pair, which is why these cannot
be appended to FREQ_BANDS.

    tremor_f0 : [f0 - w, f0 + w]        fundamental
    tremor_2f : [2*f0 - w, 2*f0 + w]    second harmonic      (w = TREMOR_HALF_WIDTH)
"""

from __future__ import annotations

from config import TREMOR_F0, TREMOR_HALF_WIDTH

TREMOR_BAND_NAMES = ['tremor_f0', 'tremor_2f']
TREMOR_BAND_LABELS = {'tremor_f0': f'Tremor fundamental (f0 ± {TREMOR_HALF_WIDTH:g} Hz)',
                      'tremor_2f': f'Tremor 2nd harmonic (2f0 ± {TREMOR_HALF_WIDTH:g} Hz)'}


def _patient_index(label: str) -> int:
    """'P01' -> 0, matching the 0-based patient index used across the pipeline."""
    return int(str(label).strip().upper().lstrip('P')) - 1


def load_f0() -> dict:
    """{(patient_idx, 'L'|'R'): f0_Hz}"""
    return {(_patient_index(p), side): float(f0)
            for p, sides in TREMOR_F0.items() for side, f0 in sides.items()}


def load_tremor_bands() -> dict:
    """{(patient_idx, 'L'|'R'): {'tremor_f0': (lo, hi), 'tremor_2f': (lo, hi)}}"""
    w = TREMOR_HALF_WIDTH
    return {key: {'tremor_f0': (f0 - w, f0 + w),
                  'tremor_2f': (2 * f0 - w, 2 * f0 + w)}
            for key, f0 in load_f0().items()}


TREMOR_BANDS = load_tremor_bands()


if __name__ == '__main__':
    for (p, side), bands in sorted(TREMOR_BANDS.items()):
        print(f'P{p + 1:02d} {side}  f0 band {bands["tremor_f0"]}  2f band {bands["tremor_2f"]}')
