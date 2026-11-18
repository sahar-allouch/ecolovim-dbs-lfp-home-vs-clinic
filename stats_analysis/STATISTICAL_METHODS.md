# Statistical Methods — ECOLOVIM Spectral Analysis

Methods for the analysis of local field potentials (LFP) recorded with the
Medtronic Percept PC neurostimulator in patients with essential tremor treated
with deep brain stimulation (DBS) of the ventral intermediate nucleus of the
thalamus (VIM). The analysis compares spectral features across three recording
sessions and four experimental conditions.

> **Data availability.** Patient recordings are not distributed with this
> repository. `make_synthetic_data.py` writes simulated inputs with the same
> structure so that every step below can be run end to end. Results obtained on
> the simulated data are meaningless; they only demonstrate that the code runs.

---

## 1. Study design

### 1.1 Sessions

| Label | Description |
|-------|-------------|
| `M3`   | In-clinic recording, 3 months after implantation |
| `home` | At-home recordings — spectra computed on the device (Percept on-chip FFT) |
| `M6`   | In-clinic recording, 6 months after implantation |

### 1.2 Conditions

| Label | Description |
|-------|-------------|
| `rest_stimOFF` | Rest, stimulation off |
| `mvt_stimOFF`  | Movement, stimulation off |
| `rest_stimON`  | Rest, stimulation on |
| `mvt_stimON`   | Movement, stimulation on |

### 1.3 Participants and missing data

Ten patients (P01–P10). Each contributes at most one value per
session × condition × hemisphere.

- **P03** withdrew after M3 and is excluded from all statistical analyses
  (`EXCLUDED_PATIENTS` in `data_utils.py`).
- **P04** has no at-home stimulation-off recordings; it is excluded from any
  test involving the `home` session for `rest_stimOFF` and `mvt_stimOFF`, and
  included otherwise (`HOME_EXCLUDED_BY_COND`).

Hence n = 9 per cell, 8 for home stim-off cells. Further cells can drop out
when a FOOOF fit fails or (for peak parameters) when no peak is detected.

| Analysis | Missing-data policy |
|----------|---------------------|
| LME | Row-wise deletion per (feature, condition) model; models with fewer than 6 rows are skipped |
| ICC / SEM | Pairwise deletion: a patient enters a session pair only if both sessions are present |
| Rest vs movement | A patient × session × stim cell enters only if both rest and movement are present |

### 1.4 Channel and hemisphere

Each analysis reads one bipolar channel per hemisphere: the patient-specific
**stimulation contact** (`STIM_CHANNELS` in `data_utils.py`). Three analysis
units are reported, each with its own results and its own multiple-comparison
families:

| Unit | Definition |
|------|------------|
| `L` | Left-hemisphere stimulation contact |
| `R` | Right-hemisphere stimulation contact |
| `bilateral` | Equal-weight mean `(L + R) / 2` per patient, taken on the per-channel feature values |

`bilateral` is the pre-specified primary unit; `L` and `R` are supporting
detail. Because `bilateral` is a function of `L` and `R`, the three units share
data: no correction is applied across them, and agreement between them is
never counted as replication.

---

## 2. Spectral features

### 2.1 Power spectra

- **In-clinic (M3, M6).** The pipeline starts from preprocessed epochs
  (filtering, epoching and artefact rejection are done upstream and are not part
  of this repository). Epochs of 800 ms at 250 Hz are concatenated per
  patient × session × condition × channel, and one Welch PSD is computed over
  the whole trace (1 s Hann window, 80 % overlap, 256-point FFT), scaled to the
  device's µVp convention (`extract_psd_welch` in `frequency_analysis.py`).
  Welch averages in the power domain, so no further averaging across epochs is
  needed.
- **At home.** The device stores its own FFT spectra (100 bins, 0–96.7 Hz).
  Bins the device censors (stored as negative values) are replaced by the last
  valid value, amplitudes are squared to power, and snapshots are averaged
  (`power.py`).

**Clinic and home are different acquisition modalities.** In this cohort the
in-clinic spectra were flatter than the at-home spectra, an effect present in
the device's own spectra and therefore not removable by rescaling. A
`M3 vs home` or `M6 vs home` effect on `exponent`, `offset`, `broadband`,
`abs_beta` or `flat_abs_beta` may therefore reflect the modality rather than
physiology. `M3 vs M6` is unaffected.

### 2.2 Frequency bands

Defined once in `config.py` (`FREQ_BANDS`):
delta 1–4, theta 4–8, alpha 8–13 (alpha1 8–10, alpha2 10–13), beta 13–35
(beta1 13–20, beta2 20–35), gamma1 35–55, gamma2 55–90 Hz.

All bands are computed; **only delta, theta, alpha and beta are modelled**
(`ANALYSIS_BANDS`). Sub-bands partition their parent band, so testing both
would re-test the same signal and inflate the correction family. Gamma bands
are excluded because the line-noise notch and the aperiodic fit range
(1–40 Hz) leave no interpretable periodic signal there.

### 2.3 Tremor bands

Each patient has an individual tremor frequency f0 per hemisphere, measured on
the contralateral tremulous limb (`TREMOR_F0` in `config.py`). Two
patient- and hemisphere-specific bands are summed from the same PSDs
(`tremor_bands.py`, `add_tremor_bands.py`):

- `tremor_f0`: f0 ± 1 Hz (fundamental)
- `tremor_2f`: 2·f0 ± 1 Hz (second harmonic)

The values shipped in `config.py` are placeholders matching the synthetic data.

### 2.4 Aperiodic and periodic components (FOOOF)

FOOOF (Donoghue et al., 2020) is fitted on the power spectrum over 1–40 Hz
with a fixed aperiodic component (no knee), `peak_width_limits = [2, 12]` Hz,
`max_n_peaks = 3`, `peak_threshold = 1.5`, `min_peak_height = 0`.

| Quantity | Definition |
|----------|------------|
| `exponent` | Aperiodic exponent (1/f slope) |
| `offset` | Aperiodic offset (log₁₀ power) |
| `flat_abs_beta` | Beta-band sum of the log-space residual `log10(PSD) − aperiodic fit`. Negative values mean power below the aperiodic fit |
| `peak_beta_cf/pw/bw` | Centre frequency (Hz), power above the aperiodic fit (log₁₀) and bandwidth (Hz) of the largest peak in the beta band |

Peak parameters exist only when a peak is detected (see §6).

### 2.5 Modelled features and transforms

The LME, ICC, SEM and rest-vs-movement analyses all use the same 13 features.

| Feature | Units | Transform before modelling |
|---------|-------|----------------------------|
| `abs_delta`, `abs_theta`, `abs_alpha`, `abs_beta` | µVp² | natural log |
| `abs_tremor_f0`, `abs_tremor_2f` | µVp² | natural log |
| `broadband` (1–90 Hz) | µVp² | natural log |
| `exponent` | — | natural log |
| `offset` | log₁₀ power | none (already log) |
| `flat_abs_beta` | log₁₀ ratio | none (already log; may be negative) |
| `peak_beta_pw` | log₁₀ power | none (already log) |
| `peak_beta_cf`, `peak_beta_bw` | Hz | none |

Values are clipped at 1e-12 before any log. Because natural-log and log₁₀
features coexist, fold-change back-transforms use `exp(β)` for the former and
`10**β` for the latter. Relative band power is computed but not modelled.

---

## 3. Session effects — linear mixed-effects models

**Module:** `stats_analysis/lme_analysis.py`

### 3.1 Question

Does a feature differ between the at-home recording and each in-clinic visit,
within a given condition? A secondary contrast describes change between the two
clinic visits.

### 3.2 Model

One model per (feature, condition), fitted by REML (`statsmodels` `mixedlm`):

```
feature ~ C(timepoint, Treatment("home")) + (1 | patient)
```

| Contrast | Source |
|----------|--------|
| M3 vs home | model coefficient |
| M6 vs home | model coefficient |
| M3 vs M6 | post-hoc linear contrast of the two coefficients |

Each model has about 27–30 observations (≤ 9 patients × 3 sessions). Conditions
are modelled separately rather than in one factorial model, so that each
contrast answers the substitution question in one experimental context without
movement × stimulation nuisance structure. Random slopes are implemented but
disabled: three observations per patient cannot support them.

A non-significant contrast is not evidence of equivalence; no equivalence test
is run (§3.7).

### 3.3 Inference

| Quantity | Construction |
|----------|--------------|
| p | Two-sided **permutation** test on the contrast \|z\|: timepoint labels shuffled within patient, `N_PERM_LME = 5000` draws, `p = (#{\|z_perm\| ≥ \|z_obs\|} + 1) / (n + 1)` |
| 95 % CI | **Cluster bootstrap** percentile interval, resampling whole patients, `N_BOOT_LME = 2000` draws. One bootstrap per (feature, condition) serves all three contrasts |
| `p_boot` | p implied by the bootstrap draws; reported for audit only (it runs at about twice the nominal type-I error at this number of clusters) |

**No Wald quantity is reported.** `statsmodels` MixedLM computes p-values from
a normal distribution with the standard error treated as known, which is not
defensible with 9 clusters.

Two consequences:

1. The p-value and the CI are different constructions and can disagree at the
   margin; "p < .05" is not interchangeable with "CI excludes zero".
2. A permutation p has a resolution floor of `1 / (n + 1)`; smaller values are
   never quoted.

Resampling uses fixed seeds, so reruns reproduce identical tables.

### 3.4 Effect size

$$d = \frac{\hat\beta}{\sigma_{\text{total}}}, \qquad \sigma_{\text{total}} = \sqrt{\tau^2 + \sigma^2}$$

with τ² the random-intercept (between-patient) variance and σ² the residual
variance; both are written to the results tables (`group_var`, `scale`).
Standardising by the total SD keeps d on the scale Cohen's benchmarks were
defined for, and puts it on the same variance decomposition as the ICC
(τ² / (τ² + σ²)). Labels: negligible < 0.20 ≤ small < 0.50 ≤ medium < 0.80 ≤
large (Cohen, 1988).

This is not the effect size for a repeated-measures power calculation, which
needs d_z = β / (σ√2). When the random intercept is singular (τ² ≈ 0, flagged
`full_singular_re`), d reduces to β / σ.

### 3.5 Leave-one-patient-out sensitivity

Each model is refitted once per left-out patient (random intercept only).
Reported per (feature, condition, contrast):

| Field | Meaning |
|-------|---------|
| `n_loo` | Folds that converged (folds with < 6 rows are skipped) |
| `consistent_n` | Folds whose significance agrees with the full model |
| `sign_stable_n` | Folds whose coefficient sign agrees with the full model |
| `singular_re_n` | Folds with a singular random intercept |
| `fragile` | `sign_stable_n < n_loo` |

LOO is a stability diagnostic; it is not corrected for multiplicity and its
fold-level p-values are never reported.

### 3.6 Multiple comparisons

Benjamini–Hochberg FDR, **one family per contrast per analysis unit**:
13 features × 4 conditions = 52 tests, three families (M3 vs home, M6 vs home,
M3 vs M6) for each of `L`, `R` and `bilateral`. A family therefore holds
exactly one test per (feature, condition). The three contrasts are not pooled
with each other: they are linear functions of the same two coefficients.
p-values from failed fits are excluded before correction.

### 3.7 Equivalence testing — not performed

Two one-sided tests (TOST) need a pre-specified equivalence margin, which
could not be justified from prior literature for these features, and at
n = 8–9 they cannot declare equivalence even for nearly identical means. No
equivalence claim is made.

### 3.8 Outputs (`Results/stats/lme/`)

- `lme_m3m6_summary_{side}.csv` — one row per (feature, condition, contrast):
  β, 95 % CI, p, q (FDR), `p_boot`, Cohen's d, variance components. **The
  canonical results table.**
- `loo_percond_{side}.csv` — leave-one-out stability.
- `lme_percond_{feature}_{side}_{condition}.txt` — full model summaries.

---

## 4. Movement effect — rest vs movement

**Module:** `stats_analysis/lme_rest_mvt_diff.py`

Rest and movement are never a term in the models of §3 or §5, which stratify by
condition. This module is the only test of the movement effect.

### 4.1 Difference score

Within each patient × session × stimulation cell, rest and movement come from
the same recording session, so the per-cell difference

```
Diff = mvt − rest        (log-transformed features ⇒ Diff = log(mvt / rest))
```

cancels the patient's baseline, the electrode configuration and the session
scale exactly, instead of estimating them.

### 4.2 Questions and models

| Question | Model | Rows |
|----------|-------|------|
| (a) Is there a movement effect within each stimulation state? | Reduced model `Diff ~ C(timepoint)` fitted on that state's rows; estimate = its session-averaged mean | ~27 per state |
| (b) Does the movement effect differ between sessions? | Full model `Diff ~ C(timepoint, Treatment("home")) + C(stim, Treatment("stimOFF")) + (1 \| patient)`, terms `timepoint[M3]`, `timepoint[M6]` | ~53 |
| (c) Does stimulation change the movement effect? | Same full model, term `stim[stimON]` | ~53 |

In the full model every term except the intercept is an interaction with
movement. The random intercept captures between-patient differences in
movement responsiveness.

### 4.3 Inference

| Term | Permutation null |
|------|------------------|
| (a) simple effect, intercept | Sign-flip of whole patients' differences; with ≤ 13 patients all 2ⁿ sign vectors are enumerated, so the test is exact |
| `timepoint` terms | Timepoint labels permuted within patient × stimulation |
| `stim` term | Stimulation labels permuted within patient × session |

Statistic: \|z\| of the term, recomputed by refitting on each permuted data
set (`N_PERM = 5000` where enumeration is not exhaustive). Intervals: cluster
bootstrap over patients (`N_BOOT = 2000`). For (a) estimate, interval and p all
come from the same reduced model on the same rows. As in §3, p and CI are
different constructions; the inferential weight for (b) and (c) is on the
interaction terms.

### 4.4 Multiple comparisons

- Model terms: BH across the 13 features, separately for each term
  (intercept, × M3, × M6, × stim) and each analysis unit.
- Simple effects: BH across the 13 features, separately within each
  stimulation state (two families of 13 per unit).

### 4.5 Outputs (`Results/stats/lme_rest_mvt/`)

`lme_rest_mvt_diff_{side}.csv` (model terms), `simple_effects_{side}.csv`
(within-state movement effects), `diff_long_{side}.csv` (the Diff values).

---

## 5. Test–retest reliability — ICC and SEM

**Modules:** `stats_analysis/icc_analysis.py`, `stats_analysis/sem_analysis.py`

### 5.1 Session pairs

Three pairwise analyses: **M3 ↔ home**, **M3 ↔ M6**, **home ↔ M6**, each on
the patients present in both sessions (k = 2). Home ↔ clinic pairs answer the
substitution question; M3 ↔ M6 describes longitudinal stability between clinic
visits.

### 5.2 ICC(2,1) — primary

Two-way random effects, absolute agreement, single measurement (Shrout &
Fleiss, 1979; McGraw & Wong, 1996):

$$\text{ICC}(2,1) = \frac{MS_r - MS_e}{MS_r + (k-1)\,MS_e + \dfrac{k}{n}(MS_c - MS_e)}$$

with MS_r, MS_c, MS_e the mean squares for patients, sessions and error. A
systematic offset between sessions lowers ICC(2,1).

### 5.3 ICC(3,1) — consistency

$$\text{ICC}(3,1) = \frac{MS_r - MS_e}{MS_r + (k-1)\,MS_e}$$

Systematic session offsets do not penalise ICC(3,1). A high ICC(3,1) beside a
low ICC(2,1) indicates that patients keep their ranking but the session level
shifts. With these sample sizes the estimate of ICC(3,1) can fall below that of
ICC(2,1) through sampling noise.

### 5.4 Interval, test and interpretation

- **95 % CI:** percentile bootstrap over patients, `N_BOOT_ICC = 5000`, for both
  ICC(2,1) and ICC(3,1). No analytical F-based interval is reported. A
  bootstrap interval narrower than 0.05 is flagged as degenerate (mostly tied
  inputs).
- **p (H₀: ICC = 0):** F = MS_r / MS_e on (n − 1, (n − 1)(k − 1)) degrees of
  freedom. It only says whether between-patient structure is detectable; it is
  **never corrected for multiplicity** and any display of it is labelled
  uncorrected. Reliability is read from the estimate, its CI and the rating.
- **Rating** (Koo & Li, 2016): poor < 0.50 ≤ moderate < 0.75 ≤ good < 0.90 ≤
  excellent.
- **Power and minimum detectable ICC:** `icc_power` (F non-centrality) and
  `min_detectable_icc` (smallest ICC detectable with 80 % power, solved with
  Brent's method) per cell.

### 5.5 Standard error of measurement

ICC is a ratio and cannot tell a homogeneous cohort from a noisy measurement.
`sem_analysis.py` uses the same ANOVA to report absolute quantities per
(unit, feature, condition, pair):

| Quantity | Definition |
|----------|------------|
| SEM | √MS_e — within-patient SD, in feature units |
| `sd_between` | √max((MS_r − MS_e)/k, 0) — between-patient SD (variance-component estimator) |
| MDC95 | 1.96 · √2 · SEM — smallest detectable change |

SEM describes the measurement and transfers to other cohorts; `sd_between`
describes these patients only. For log-transformed features both are also
expressed as percentages (multiplicative error).

### 5.6 Outputs (`Results/stats/icc/`)

`icc_all_pairwise_{side}.csv` (ICC(2,1), ICC(3,1), CIs, p, power, MDI),
`sem_all_sides.csv`.

---

## 6. Peak detection

**Modules:** `stats_analysis/peak_params_analysis.py`,
`stats_analysis/band_peak_detection.py`

A peak is "detected" in a band when FOOOF returns a peak whose centre frequency
lies in that band. Peak parameters are therefore **conditional on detection**
and missing not at random: a hemisphere without a detectable peak is plausibly
one with low power. The beta peak parameters enter §3 and §5 with everything
else, and must be read beside their detection rates.

- `peak_params_availability.csv` — detection counts per cell, and how many
  bandwidths sit on a fit bound (`peak_width_limits`); a bandwidth at a bound
  is not a measurement, so bandwidth results are descriptive.
- `band_peaks/` — detection counts per band × condition × session × side, each
  with its own denominator (k / n).

No models are fitted here.

---

## 7. Electrode impedance

**Module:** `stats_analysis/wilcoxon_impedance.py`

Monopolar impedances at M3 and M6 are averaged over measurement sessions and
compared with the Wilcoxon signed-rank test (exact null when there are no ties
or zero differences, normal approximation otherwise; zero differences dropped).
Effect size: matched-pairs rank-biserial correlation
r_rb = (W⁺ − W⁻) / (n_eff(n_eff + 1)/2), positive when M6 > M3.

Two levels, each with its own BH family: mean over contacts per hemisphere
(2 tests) and per contact × hemisphere (16 tests). BH runs on unrounded p.

---

## 8. Multiple-comparison families — summary

A family is always *features × conditions within one question*. Contrasts of
one fitted coefficient vector are never pooled, and no family spans analysis
units.

| Analysis | Correction | Family |
|----------|------------|--------|
| Session effects (LME) | Benjamini–Hochberg | per contrast per unit: 13 features × 4 conditions = 52 |
| Movement effect, model terms | Benjamini–Hochberg | per term per unit: 13 features |
| Movement effect, simple effects | Benjamini–Hochberg | per stimulation state per unit: 13 features |
| ICC | none | p reported as uncorrected; not used for inference |
| Impedance | Benjamini–Hochberg | hemisphere level (2), contact level (16) |
| LOO, peak detection, SEM | none | descriptive |

Benjamini–Hochberg assumes independence or positive regression dependence.
Tests within a family share patients, so dependence is present and not verified
to be positive; BH is used as the field convention.

---

## 9. Key constants

| Constant | Value | Where |
|----------|-------|-------|
| `ALPHA` | 0.05 | all tests |
| `N_PERM_LME` | 5000 | `lme_analysis.py` |
| `N_BOOT_LME` | 2000 | `lme_analysis.py` |
| `N_PERM` | 5000 | `lme_rest_mvt_diff.py` |
| `N_BOOT` | 2000 | `lme_rest_mvt_diff.py` |
| `EXHAUSTIVE_MAX_PATIENTS` | 13 | `lme_rest_mvt_diff.py` — exact sign-flip enumeration below this |
| `N_BOOT_ICC` | 5000 | `icc_analysis.py` |
| `MDI_POWER` | 0.80 | `icc_analysis.py` |
| `TP_PAIRS` | (M3, home), (M3, M6), (home, M6) | `icc_analysis.py`, `sem_analysis.py` |
| `PEAK_WIDTH_LIMITS` | 2–12 Hz | FOOOF fit, `frequency_analysis.py` |
| `APERIODIC_FREQ_RANGE` | 1–40 Hz | `config.py` |
| `TREMOR_HALF_WIDTH` | 1 Hz | `config.py` |
| `_CLIP` | 1e-12 | floor before log transforms |

`ECOLOVIM_LME_NPERM` / `ECOLOVIM_LME_NBOOT` reduce the LME resampling budget
for quick previews; previews raise the permutation floor and are not results.

---

## 10. References

- Benjamini, Y. & Hochberg, Y. (1995). Controlling the false discovery rate: a
  practical and powerful approach to multiple testing. *Journal of the Royal
  Statistical Society B*, 57(1), 289–300.
- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences*
  (2nd ed.). Lawrence Erlbaum.
- Donoghue, T. et al. (2020). Parameterizing neural power spectra into periodic
  and aperiodic components. *Nature Neuroscience*, 23, 1655–1665.
- Koo, T.K. & Li, M.Y. (2016). A guideline of selecting and reporting
  intraclass correlation coefficients for reliability research. *Journal of
  Chiropractic Medicine*, 15(2), 155–163.
- McGraw, K.O. & Wong, S.P. (1996). Forming inferences about some intraclass
  correlation coefficients. *Psychological Methods*, 1(1), 30–46.
- Schuirmann, D.J. (1987). A comparison of the two one-sided tests procedure
  and the power approach for assessing the equivalence of average
  bioavailability. *Journal of Pharmacokinetics and Biopharmaceutics*, 15(6),
  657–680.
- Shrout, P.E. & Fleiss, J.L. (1979). Intraclass correlations: uses in
  assessing rater reliability. *Psychological Bulletin*, 86(2), 420–428.
