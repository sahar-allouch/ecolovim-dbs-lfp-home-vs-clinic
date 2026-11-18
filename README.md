# ECOLOVIM — LFP analysis pipeline for VIM-DBS in essential tremor

Analysis code for local field potentials (LFP) recorded with the **Medtronic
Percept PC** neurostimulator in patients with essential tremor treated with deep
brain stimulation (DBS) of the ventral intermediate nucleus of the thalamus
(VIM).

The pipeline compares spectral features of the LFP across three recording
sessions — in clinic at **3 months (M3)**, **at home**, and in clinic at
**6 months (M6)** — under four conditions (rest / movement × stimulation
off / on). It asks:

- **Session effects:** does the at-home recording differ from each clinic visit?
  (linear mixed-effects models)
- **Reliability:** how stable are the features between sessions?
  (ICC(2,1), ICC(3,1), standard error of measurement)
- **Movement effect:** does movement change the spectrum, and does that change
  depend on the session or on stimulation?

The full statistical methods are in
[stats_analysis/STATISTICAL_METHODS.md](stats_analysis/STATISTICAL_METHODS.md).

> **Data availability**
>
> The patient recordings are **not** included in this repository.
> `make_synthetic_data.py` generates simulated data with exactly the same
> structure, so the whole pipeline can be run without access to the real data.
> **Results obtained on the simulated data are meaningless** — they only show
> that the code runs.

---

## Requirements

- Python ≥ 3.11
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

Dependencies (`pyproject.toml`): numpy, scipy, pandas, matplotlib, mne, fooof,
statsmodels, plotly, joblib, openpyxl. Exact tested versions are pinned in
`uv.lock`.

## Installation

```bash
git clone <this-repository-url>
cd <repository-folder>

# with uv (installs the exact versions from uv.lock)
uv sync

# or with pip
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install .                      # installs the dependencies
```

With uv, prefix the commands below with `uv run` (e.g. `uv run python run_all.py`),
or activate `.venv` first.

---

## Quick start (simulated data, ~10 minutes)

```bash
# 1. Generate the simulated input data
python make_synthetic_data.py

# 2. Run the whole pipeline with a small resampling budget (preview)
export ECOLOVIM_LME_NPERM=100 ECOLOVIM_LME_NBOOT=100 ECOLOVIM_LME_JOBS=4
python run_all.py -j 4 --group features                                  # spectral features
python run_all.py --only lme-rest-mvt --args --n-perm 100 --n-boot 100 --jobs 4
python run_all.py -j 4                                                   # everything else
```

Results appear in `Results/` (tables in `Results/stats/`, figures in
`Results/Figures/`).

The reduced budget is for checking that the code runs. Reported results use the
default budget (5000 permutations, 2000 bootstrap resamples), which takes
several hours — see [Full run](#full-run).

---

## How the pipeline is organised

```text
Preprocessed epochs (M3, M6) ─┐
At-home device spectra ───────┤
                              ▼
           pipeline.py              PSD (Welch), band power, FOOOF, tremor bands
                              │     → Results/features/*.pkl
                              ▼
    ┌─────────────────────────┼───────────────────────────────┐
    ▼                         ▼                               ▼
 lme_analysis.py        icc_analysis.py               lme_rest_mvt_diff.py
 session effects        sem_analysis.py               movement effect
                        reliability
    │                         │                               │
    └─────────────────────────┼───────────────────────────────┘
                              ▼
                   make_*_figure.py  → Results/Figures/
```

### `run_all.py` — one entry point

`run_all.py` declares every step as a *stage* (command, input files, output
files, source files). It runs the stages in dependency order and **re-runs a
stage only when something it depends on changed** (its code, its inputs, or an
upstream stage). State is kept in `Results/.run_all_state.json`; each stage
writes a log to `Results/logs/<stage>.log`.

```bash
python run_all.py --list              # show all stages and which are out of date
python run_all.py --dry-run           # show what would run, run nothing
python run_all.py                     # run everything that is out of date
python run_all.py --only lme icc      # run only these stages
python run_all.py --from tremor-bands # a stage and everything downstream of it
python run_all.py --group figures     # one group: features, stats, impedance, figures
python run_all.py --skip lme-rest-mvt # everything except these stages
python run_all.py --force             # rebuild even if up to date
python run_all.py -j 4                # run independent stages in parallel
python run_all.py --keep-going        # do not stop at the first failure
python run_all.py --only lme-rest-mvt --args --jobs 8   # pass arguments to the stage's script
```

### Stages

| Group | Stage | Script | What it does |
|-------|-------|--------|--------------|
| features | `features-M3`, `features-M6` | `pipeline.py` | Welch PSD, band power, FOOOF and tremor bands from the clinic epochs |
| features | `features-home` | `pipeline.py` | Band power, FOOOF and tremor bands from the device spectra |
| features | `tremor-bands` | `add_tremor_bands.py` | Report of the frequency bins in each tremor band |
| stats | `lme` | `stats_analysis/lme_analysis.py` | Session effects (M3 vs home, M6 vs home, M3 vs M6) |
| stats | `sem` | `stats_analysis/sem_analysis.py` | Standard error of measurement, minimal detectable change |
| stats | `icc` | `stats_analysis/icc_analysis.py` | ICC(2,1) and ICC(3,1) for each pair of sessions |
| stats | `lme-rest-mvt` | `stats_analysis/lme_rest_mvt_diff.py` | Movement effect and its interaction with session and stimulation |
| stats | `peak-params` | `stats_analysis/peak_params_analysis.py` | How often a beta peak is detected |
| stats | `band-peaks` | `stats_analysis/band_peak_detection.py` | Peak detection counts for every band |
| impedance | `impedance-wilcoxon` | `stats_analysis/wilcoxon_impedance.py` | Electrode impedance, M3 vs M6 |
| figures | `fig-lme-effectsize` | `make_lme_effectsize_figure.py` | Cohen's d of every session contrast |
| figures | `fig-icc-summary` | `make_icc_summary_figure.py` | ICC(2,1) per feature, session pair and stimulation state |
| figures | `fig-rest-mvt-combined` (`-L`, `-R`) | `make_rest_mvt_combined_figure.py` | Rest vs movement: spectra (panel A) and effect per feature (panel B) |
| figures | `fig-trajectory` | `make_trajectory_figure.py` | Per-patient feature values across sessions |
| figures | `fig-impedance` | `make_impedance_figure.py` | Impedance per contact, M3 vs M6 |

Each script can also be run on its own from the repository root, e.g.
`python -m stats_analysis.icc_analysis` or `python make_trajectory_figure.py`.

### Full run

The two slow stages (`lme` and `lme-rest-mvt`, permutation tests and cluster
bootstraps) take several hours at the default budget. Both run in parallel on
several cores without changing the results (every work unit is seeded).
`lme-rest-mvt` runs on its own first, because its figures depend on it and
arguments passed with `--args` reach only the selected stage:

```bash
unset ECOLOVIM_LME_NPERM ECOLOVIM_LME_NBOOT
python run_all.py --force --group features
python run_all.py --only lme-rest-mvt --force --args --jobs 8
ECOLOVIM_LME_JOBS=8 python run_all.py      # all remaining stages
python run_all.py --list                   # every stage should read "up to date"
```

Never use `ECOLOVIM_LME_NPERM` / `ECOLOVIM_LME_NBOOT` for reported numbers: a
small number of permutations raises the smallest p-value that can be obtained.

---

## Repository contents

```text
.
├── run_all.py                  # entry point: runs the pipeline stage by stage
├── make_synthetic_data.py      # writes simulated input data
├── pipeline.py                 # spectral features for one session ($ECOLOVIM_SESSION)
├── config.py                   # frequency bands, fit ranges, tremor frequencies
├── power.py                    # band power and FOOOF for the at-home spectra
├── tremor_bands.py             # patient-specific tremor bands (from config.py)
├── add_tremor_bands.py         # tremor-band power
├── spectral_analysis/
│   ├── frequency_analysis.py   # Welch PSD, band power, FOOOF
│   └── get_spectral_feat.py    # features for one recording
├── stats_analysis/
│   ├── data_utils.py           # loading, stimulation contacts, exclusions, long tables
│   ├── lme_analysis.py
│   ├── lme_rest_mvt_diff.py
│   ├── icc_analysis.py
│   ├── sem_analysis.py
│   ├── peak_params_analysis.py
│   ├── band_peak_detection.py
│   ├── wilcoxon_impedance.py
│   ├── plot_style.py
│   └── STATISTICAL_METHODS.md
├── make_*_figure.py            # figures
└── table_figure_style.py       # shared figure style
```

---

## Using your own data

The pipeline starts from preprocessed data. Filtering, epoching and artefact
rejection of the raw Percept recordings are done beforehand and are not part of
this repository. `make_synthetic_data.py` is the reference for every format
below.

| File | Content |
|------|---------|
| `Results/data-preprocessed/epochs_dict_M3_badremoved.pkl`, `..._M6_...` | `{condition: [one mne.EpochsArray per patient, or None]}`. Conditions: `rest_stimOFF`, `mvt_stimOFF`, `rest_stimON`, `mvt_stimON`. 800 ms epochs at 250 Hz, in volts, bipolar channels named like `L13`, `R02` |
| `Results/data-preprocessed/home_fft_dict.pkl` | `{condition: [per patient: None or {'fft_bin_data': array (n_snapshots, n_channels, 100), 'ch_names': [...], 'freqs': array (100,)}]}` — the device's own FFT, µVp, 0–96.7 Hz |
| `Results/Tables/impedance_all_patients.csv` | One row per impedance measurement: `Patient, Timepoint, Hemisphere, Mode, Electrode1, Electrode2, Impedance_Ohm` (and session metadata) |

Then edit:

- `config.py` → `TREMOR_F0`: each patient's tremor frequency per hemisphere
  (the shipped values are placeholders).
- `stats_analysis/data_utils.py` → `STIM_CHANNELS` (the channel analysed per
  patient and hemisphere), `NB_PATIENTS`, `EXCLUDED_PATIENTS`,
  `HOME_EXCLUDED_BY_COND`.

and run `python run_all.py`.

---

## Citation

<!-- TODO: add the article reference once published -->

## License

Released under the [MIT License](LICENSE).
